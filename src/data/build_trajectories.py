from __future__ import annotations

import argparse
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.data.build_vocab import cohort_path_from_config, load_vocab_bundle
from src.data.load_mimic import MIMICDataPaths, choose_stay_for_event, iter_table
from src.features.lab_processor import LabProcessor
from src.features.medication_history import (
    build_cumulative_history,
    dedupe_preserve_order,
    extract_medication_token,
    medication_event_time,
)
from src.features.vital_processor import VitalProcessor
from src.utils.io import (
    ensure_dir,
    hours_between,
    load_yaml_config,
    parse_datetime,
    parse_float,
    parse_int,
    read_csv_gz,
    resolve_path,
    write_json,
    write_jsonl_gz,
)


def _trajectory_root(config: dict) -> Path:
    processed_root = resolve_path(config["_project_root"], config["paths"]["processed_root"])
    return ensure_dir(Path(processed_root) / "trajectories")


def _bucket_index(stay: dict[str, object], event_time: datetime | None, bucket_hours: int) -> int:
    num_steps = int(stay["num_steps"])
    if num_steps <= 1 or event_time is None:
        return 0
    start = stay["intime_dt"]
    end = stay["outtime_dt"]
    if event_time <= start:
        return 0
    if event_time >= end:
        return num_steps - 1
    delta_hours = max((event_time - start).total_seconds() / 3600.0, 0.0)
    return min(int(delta_hours // bucket_hours), num_steps - 1)


def _load_stays(config: dict) -> tuple[list[dict[str, object]], dict[int, dict[str, object]], dict[int, list[dict[str, object]]]]:
    cohort_rows = read_csv_gz(cohort_path_from_config(config))
    stays: list[dict[str, object]] = []
    stays_by_id: dict[int, dict[str, object]] = {}
    stays_by_hadm: dict[int, list[dict[str, object]]] = defaultdict(list)
    bucket_hours = int(config.get("features", {}).get("time_bucket_hours", 24))
    for row in cohort_rows:
        stay_id = parse_int(row.get("stay_id"))
        hadm_id = parse_int(row.get("hadm_id"))
        subject_id = parse_int(row.get("subject_id"))
        intime = parse_datetime(row.get("intime"))
        outtime = parse_datetime(row.get("outtime"))
        split = str(row.get("split", "train"))
        if None in (stay_id, hadm_id, subject_id) or intime is None or outtime is None:
            continue
        num_steps = max(1, int(math.ceil(hours_between(intime, outtime) / float(bucket_hours))))
        stay = {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "stay_id": stay_id,
            "split": split,
            "intime": row.get("intime", ""),
            "outtime": row.get("outtime", ""),
            "intime_dt": intime,
            "outtime_dt": outtime,
            "num_steps": num_steps,
        }
        stays.append(stay)
        stays_by_id[stay_id] = stay
        stays_by_hadm[hadm_id].append(stay)
    for hadm_stays in stays_by_hadm.values():
        hadm_stays.sort(key=lambda item: item["intime_dt"])
    stays.sort(key=lambda item: (item["subject_id"], item["hadm_id"], item["stay_id"]))
    return stays, stays_by_id, stays_by_hadm


def build_trajectories(config_path: str | Path) -> dict[str, Path]:
    config = load_yaml_config(config_path)
    paths = MIMICDataPaths.from_config(config)
    vocab_bundle = load_vocab_bundle(config)
    trajectory_root = _trajectory_root(config)
    stays, stays_by_id, stays_by_hadm = _load_stays(config)
    feature_cfg = config.get("features", {})
    bucket_hours = int(feature_cfg.get("time_bucket_hours", 24))
    max_med_history = int(feature_cfg.get("max_med_history", 32))
    normalization_eps = float(feature_cfg.get("normalization_eps", 1e-6))

    num_lab_features = max(len(vocab_bundle["lab"]["idx_to_token"]) - 2, 0)
    num_vital_features = max(len(vocab_bundle["vital"]["idx_to_token"]) - 2, 0)
    drug_vocab_size = len(vocab_bundle["drug"]["idx_to_token"])

    lab_stats = LabProcessor.init_running_stats(num_lab_features)
    vital_stats = VitalProcessor.init_running_stats(num_vital_features)

    store: dict[int, dict[str, object]] = {}
    for stay in stays:
        store[int(stay["stay_id"])] = {
            "diagnosis_ids": set(),
            "procedure_by_bucket": defaultdict(set),
            "lab_sparse": {},
            "vital_sparse": {},
            "target_drugs_by_bucket": defaultdict(list),
        }

    for row in iter_table(paths, "diagnoses_icd", fields=["hadm_id", "icd_code", "icd_version"]):
        hadm_id = parse_int(row.get("hadm_id"))
        code = str(row.get("icd_code", "")).strip()
        version = str(row.get("icd_version", "")).strip()
        if hadm_id not in stays_by_hadm or not code or not version:
            continue
        token = f"ICD{version}:{code}"
        diagnosis_id = vocab_bundle["diagnosis"]["token_to_idx"].get(token, 1)
        for stay in stays_by_hadm[hadm_id]:
            store[int(stay["stay_id"])]["diagnosis_ids"].add(diagnosis_id)

    for row in iter_table(paths, "procedures_icd", fields=["hadm_id", "chartdate", "icd_code", "icd_version"]):
        hadm_id = parse_int(row.get("hadm_id"))
        code = str(row.get("icd_code", "")).strip()
        version = str(row.get("icd_version", "")).strip()
        if hadm_id not in stays_by_hadm or not code or not version:
            continue
        event_time = parse_datetime(row.get("chartdate"))
        stay = choose_stay_for_event(stays_by_hadm[hadm_id], event_time)
        if stay is None:
            continue
        procedure_id = vocab_bundle["procedure"]["token_to_idx"].get(f"PROC{version}:{code}", 1)
        bucket_index = _bucket_index(stay, event_time, bucket_hours)
        store[int(stay["stay_id"])]["procedure_by_bucket"][bucket_index].add(procedure_id)

    for row in iter_table(paths, "labevents", fields=["hadm_id", "itemid", "valuenum", "charttime"]):
        hadm_id = parse_int(row.get("hadm_id"))
        itemid = str(row.get("itemid", "")).strip()
        value = parse_float(row.get("valuenum"))
        if hadm_id not in stays_by_hadm or not itemid or value is None:
            continue
        vocab_index = vocab_bundle["lab"]["token_to_idx"].get(f"LAB:{itemid}")
        if vocab_index is None or vocab_index < 2:
            continue
        feature_index = vocab_index - 2
        event_time = parse_datetime(row.get("charttime"))
        stay = choose_stay_for_event(stays_by_hadm[hadm_id], event_time)
        if stay is None:
            continue
        bucket_index = _bucket_index(stay, event_time, bucket_hours)
        LabProcessor.update_latest(
            store[int(stay["stay_id"])]["lab_sparse"],
            bucket_index,
            feature_index,
            event_time,
            value,
        )
        if stay["split"] == "train":
            LabProcessor.update_running_stats(lab_stats, feature_index, value)

    for row in iter_table(paths, "chartevents", fields=["stay_id", "itemid", "valuenum", "charttime"]):
        stay_id = parse_int(row.get("stay_id"))
        itemid = str(row.get("itemid", "")).strip()
        value = parse_float(row.get("valuenum"))
        if stay_id not in stays_by_id or not itemid or value is None:
            continue
        vocab_index = vocab_bundle["vital"]["token_to_idx"].get(f"VITAL:{itemid}")
        if vocab_index is None or vocab_index < 2:
            continue
        feature_index = vocab_index - 2
        stay = stays_by_id[stay_id]
        event_time = parse_datetime(row.get("charttime"))
        bucket_index = _bucket_index(stay, event_time, bucket_hours)
        VitalProcessor.update_latest(
            store[int(stay["stay_id"])]["vital_sparse"],
            bucket_index,
            feature_index,
            event_time,
            value,
        )
        if stay["split"] == "train":
            VitalProcessor.update_running_stats(vital_stats, feature_index, value)

    for table_name, fields in (
        ("emar", ["hadm_id", "medication", "charttime", "scheduletime", "storetime"]),
        ("prescriptions", ["hadm_id", "drug", "formulary_drug_cd", "starttime", "stoptime"]),
        ("pharmacy", ["hadm_id", "medication", "starttime", "verifiedtime", "entertime"]),
    ):
        for row in iter_table(paths, table_name, fields=fields):
            hadm_id = parse_int(row.get("hadm_id"))
            if hadm_id not in stays_by_hadm:
                continue
            token = extract_medication_token(row)
            if not token:
                continue
            drug_id = vocab_bundle["drug"]["token_to_idx"].get(token, 1)
            event_time = medication_event_time(table_name, row)
            stay = choose_stay_for_event(stays_by_hadm[hadm_id], event_time)
            if stay is None:
                continue
            bucket_index = _bucket_index(stay, event_time, bucket_hours)
            store[int(stay["stay_id"])]["target_drugs_by_bucket"][bucket_index].append(drug_id)

    lab_processor = LabProcessor(
        num_lab_features,
        LabProcessor.finalize_running_stats(lab_stats, eps=normalization_eps),
        eps=normalization_eps,
    )
    vital_processor = VitalProcessor(
        num_vital_features,
        VitalProcessor.finalize_running_stats(vital_stats, eps=normalization_eps),
        eps=normalization_eps,
    )

    records_by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for stay in stays:
        stay_id = int(stay["stay_id"])
        state = store[stay_id]
        num_steps = int(stay["num_steps"])
        diagnosis_ids = sorted(int(code_id) for code_id in state["diagnosis_ids"])
        bucket_drugs = [
            dedupe_preserve_order(state["target_drugs_by_bucket"].get(step_index, []))
            for step_index in range(num_steps)
        ]
        med_history = build_cumulative_history(bucket_drugs, max_med_history)
        lab_values, lab_mask = lab_processor.build_dense_steps(state["lab_sparse"], num_steps)
        vital_values, vital_mask = vital_processor.build_dense_steps(state["vital_sparse"], num_steps)

        steps: list[dict[str, object]] = []
        for step_index in range(num_steps):
            steps.append(
                {
                    "step_index": step_index,
                    "diagnosis_ids": diagnosis_ids,
                    "procedure_ids": sorted(state["procedure_by_bucket"].get(step_index, set())),
                    "lab_values": lab_values[step_index],
                    "lab_mask": lab_mask[step_index],
                    "vital_values": vital_values[step_index],
                    "vital_mask": vital_mask[step_index],
                    "med_history_ids": med_history[step_index],
                    "delta_hours": 0.0 if step_index == 0 else float(bucket_hours),
                    "target_drugs": bucket_drugs[step_index],
                }
            )

        records_by_split[str(stay["split"])].append(
            {
                "subject_id": stay["subject_id"],
                "hadm_id": stay["hadm_id"],
                "stay_id": stay_id,
                "split": stay["split"],
                "intime": stay["intime"],
                "outtime": stay["outtime"],
                "num_steps": num_steps,
                "drug_vocab_size": drug_vocab_size,
                "lab_feature_size": num_lab_features,
                "vital_feature_size": num_vital_features,
                "steps": steps,
            }
        )

    output_paths: dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        split_dir = ensure_dir(trajectory_root / split_name)
        output_paths[split_name] = write_jsonl_gz(
            split_dir / "trajectories.jsonl.gz",
            records_by_split.get(split_name, []),
        )

    write_json(
        trajectory_root / "normalization_stats.json",
        {
            "lab": [stat.__dict__ for stat in lab_processor.stats],
            "vital": [stat.__dict__ for stat in vital_processor.stats],
            "eps": normalization_eps,
        },
    )
    write_json(
        trajectory_root / "metadata.json",
        {
            "bucket_hours": bucket_hours,
            "max_med_history": max_med_history,
            "drug_vocab_size": drug_vocab_size,
            "lab_feature_size": num_lab_features,
            "vital_feature_size": num_vital_features,
            "counts_by_split": {split: len(records) for split, records in records_by_split.items()},
        },
    )
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bucketed ICU stay trajectories.")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    args = parser.parse_args()
    build_trajectories(args.config)


if __name__ == "__main__":
    main()
