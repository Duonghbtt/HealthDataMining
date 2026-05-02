from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.io import (
    ensure_dir,
    hours_between,
    load_yaml_config,
    parse_datetime,
    read_csv_gz,
    read_json,
    resolve_path,
    write_json,
    write_jsonl_gz,
)


LOGGER = logging.getLogger(__name__)


def _normalize_free_text(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value).upper()).strip("_")


def _canonicalize_medication_text(value: str, *, prefer_code: bool = False) -> str | None:
    raw_value = str(value).strip()
    if not raw_value:
        return None
    if raw_value.startswith(("NAME:", "CODE:")):
        return raw_value
    normalized = _normalize_free_text(raw_value)
    if not normalized:
        return None
    prefix = "CODE" if prefer_code else "NAME"
    return f"{prefix}:{normalized}"


def _extract_medication_token(row: Mapping[str, Any]) -> str | None:
    for field in ("medication", "drug", "formulary_drug_cd"):
        token = _canonicalize_medication_text(
            str(row.get(field, "")),
            prefer_code=(field == "formulary_drug_cd"),
        )
        if token:
            return token
    return None


def _dedupe_preserve_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        resolved = int(value)
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
    )


def _trajectory_output_root(config: Mapping[str, Any]) -> Path:
    paths_cfg = dict(config.get("paths", {}))
    trajectory_root = paths_cfg.get("trajectory_interim_root")
    if trajectory_root:
        return ensure_dir(resolve_path(config["_project_root"], trajectory_root))
    interim_root = resolve_path(config["_project_root"], paths_cfg["interim_root"])
    return ensure_dir(Path(interim_root) / "trajectories")


def _vocab_dir_from_config(config: Mapping[str, Any]) -> Path:
    paths_cfg = dict(config.get("paths", {}))
    vocab_root = paths_cfg.get("vocab_root")
    if vocab_root:
        return Path(resolve_path(config["_project_root"], vocab_root))
    interim_root = resolve_path(config["_project_root"], paths_cfg["interim_root"])
    return Path(interim_root) / "vocab"


def _cohort_path_from_config(config: Mapping[str, Any]) -> Path:
    paths_cfg = dict(config.get("paths", {}))
    cohort_root = paths_cfg.get("cohort_root")
    if cohort_root:
        return Path(resolve_path(config["_project_root"], cohort_root)) / "cohort.csv.gz"
    interim_root = resolve_path(config["_project_root"], paths_cfg["interim_root"])
    return Path(interim_root) / "cohort" / "cohort.csv.gz"


def _trajectory_file(root: Path, split: str) -> Path:
    return root / split / "trajectories.jsonl.gz"


def _read_first_existing_json(vocab_dir: Path, candidate_names: tuple[str, ...]) -> dict[str, Any]:
    for filename in candidate_names:
        path = vocab_dir / filename
        if path.exists():
            payload = read_json(path)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid vocab artifact at {path}: expected object payload.")
            return payload
    raise FileNotFoundError(
        f"Missing expected vocab artifact under {vocab_dir}. Checked: {list(candidate_names)}"
    )


def _load_vocab_bundle_for_trajectories(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    vocab_dir = _vocab_dir_from_config(config)
    med_main_vocab = _read_first_existing_json(
        vocab_dir,
        ("med_vocab_main.json",),
    )
    return {
        "diagnosis": _read_first_existing_json(
            vocab_dir,
            ("diag_vocab.json", "diagnosis_vocab.json"),
        ),
        "procedure": _read_first_existing_json(
            vocab_dir,
            ("proc_vocab.json", "procedure_vocab.json"),
        ),
        "drug": med_main_vocab,
        "med_main": med_main_vocab,
        "lab": _read_first_existing_json(vocab_dir, ("lab_vocab.json",)),
        "vital": _read_first_existing_json(vocab_dir, ("vital_vocab.json",)),
    }


def _load_med_vocab_main_metadata(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    vocab_dir = _vocab_dir_from_config(config)
    metadata_path = vocab_dir / "med_vocab_main_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Missing medication vocab metadata required by build_trajectories: "
            f"{metadata_path}. Run build_vocab.py first."
        )
    payload = read_json(metadata_path)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid med_vocab_main_metadata artifact at {metadata_path}: expected object."
        )
    return payload


def _parse_json_list(value: str | None, *, field_name: str) -> list[Any]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse %s JSON payload: %s", field_name, exc)
        return []
    if isinstance(parsed, list):
        return parsed
    LOGGER.warning("Expected %s to decode to a list, got %s", field_name, type(parsed).__name__)
    return []


def _build_raw_medication_token_to_ids(
    vocab_bundle: Mapping[str, dict[str, Any]],
    med_vocab_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[int, ...]]:
    primary_med_vocab = vocab_bundle["med_main"]
    token_to_idx = {
        str(token): int(index)
        for token, index in primary_med_vocab["token_to_idx"].items()
        if int(index) >= 2
    }

    raw_token_to_ids: dict[str, list[int]] = defaultdict(list)
    for canonical_token, metadata_row in med_vocab_metadata.items():
        med_id = token_to_idx.get(str(canonical_token))
        if med_id is None:
            continue
        for raw_token in metadata_row.get("source_raw_tokens", []):
            raw_token_text = str(raw_token).strip()
            if not raw_token_text:
                continue
            raw_token_to_ids[raw_token_text].append(int(med_id))

    finalized: dict[str, tuple[int, ...]] = {}
    for raw_token, ids in raw_token_to_ids.items():
        finalized[raw_token] = tuple(_dedupe_preserve_order(sorted(int(value) for value in ids)))
    return finalized


def _encode_visit_row(
    row: Mapping[str, str],
    *,
    diag_token_to_idx: Mapping[str, int],
    proc_token_to_idx: Mapping[str, int],
    raw_medication_token_to_ids: Mapping[str, tuple[int, ...]],
    max_med_history: int,
    diagnostics: Counter[str],
) -> dict[str, Any]:
    diagnosis_tokens = [
        str(token).strip()
        for token in _parse_json_list(row.get("diagnosis_codes"), field_name="diagnosis_codes")
        if str(token).strip()
    ]
    procedure_tokens = [
        str(token).strip()
        for token in _parse_json_list(row.get("procedure_codes"), field_name="procedure_codes")
        if str(token).strip()
    ]
    raw_medication_records = [
        dict(record)
        for record in _parse_json_list(row.get("raw_medication_records"), field_name="raw_medication_records")
        if isinstance(record, Mapping)
    ]

    diagnosis_ids = _dedupe_preserve_order([int(diag_token_to_idx.get(token, 1)) for token in diagnosis_tokens])
    procedure_ids = _dedupe_preserve_order([int(proc_token_to_idx.get(token, 1)) for token in procedure_tokens])

    target_drugs: list[int] = []
    unmapped_med_tokens: list[str] = []
    medication_tokens_seen = 0
    mapped_medication_tokens = 0
    for medication_record in raw_medication_records:
        token = _extract_medication_token(medication_record)
        if not token:
            continue
        medication_tokens_seen += 1
        mapped_ids = raw_medication_token_to_ids.get(str(token), ())
        if not mapped_ids:
            unmapped_med_tokens.append(str(token))
            continue
        mapped_medication_tokens += 1
        target_drugs.extend(int(value) for value in mapped_ids)
    target_drugs = _dedupe_preserve_order(target_drugs)

    if not target_drugs:
        diagnostics["visits_with_empty_targets_after_normalization"] += 1
    diagnostics["diagnosis_ids_total"] += len(diagnosis_ids)
    diagnostics["procedure_ids_total"] += len(procedure_ids)
    diagnostics["raw_medication_tokens_seen"] += medication_tokens_seen
    diagnostics["raw_medication_tokens_mapped"] += mapped_medication_tokens
    diagnostics["target_medication_ids_total"] += len(target_drugs)
    diagnostics["visits_total"] += 1

    intime = str(row.get("intime", "")).strip()
    outtime = str(row.get("outtime", "")).strip()
    return {
        "subject_id": int(row["subject_id"]),
        "hadm_id": int(row["hadm_id"]),
        "stay_id": int(row["stay_id"]),
        "split": str(row.get("split", "train")).strip() or "train",
        "visit_order": int(row.get("visit_order", 0) or 0),
        "patient_num_visits": int(row.get("patient_num_visits", 0) or 0),
        "intime": intime,
        "outtime": outtime,
        "intime_dt": parse_datetime(intime),
        "outtime_dt": parse_datetime(outtime),
        "diagnosis_ids": diagnosis_ids,
        "procedure_ids": procedure_ids,
        "target_drugs": target_drugs,
        "lab_values": [],
        "lab_mask": [],
        "vital_values": [],
        "vital_mask": [],
        "unmapped_medication_tokens": unmapped_med_tokens[:10],
        "max_med_history": int(max_med_history),
    }


def _copy_step(step: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "step_index": int(step["step_index"]),
        "diagnosis_ids": [int(value) for value in step.get("diagnosis_ids", [])],
        "procedure_ids": [int(value) for value in step.get("procedure_ids", [])],
        "lab_values": list(step.get("lab_values", [])),
        "lab_mask": list(step.get("lab_mask", [])),
        "vital_values": list(step.get("vital_values", [])),
        "vital_mask": list(step.get("vital_mask", [])),
        "med_history_ids": [int(value) for value in step.get("med_history_ids", [])],
        "delta_hours": float(step.get("delta_hours", 0.0)),
        "target_drugs": [int(value) for value in step.get("target_drugs", [])],
    }


def _build_patient_prefix_trajectories(
    encoded_visits_by_subject: Mapping[int, list[dict[str, Any]]],
    *,
    drug_vocab_size: int,
    max_med_history: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    records_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diagnostics: Counter[str] = Counter()
    trajectory_lengths: list[int] = []
    visit_target_sizes: list[int] = []

    for subject_id in sorted(encoded_visits_by_subject):
        visits = sorted(
            encoded_visits_by_subject[subject_id],
            key=lambda item: (int(item["visit_order"]), str(item["intime"]), int(item["hadm_id"])),
        )
        if not visits:
            continue

        split = str(visits[0]["split"])
        patient_num_visits = int(visits[0]["patient_num_visits"] or len(visits))
        cumulative_history: list[int] = []
        trajectory_steps: list[dict[str, Any]] = []
        previous_intime_dt = None
        trajectory_start_time = str(visits[0]["intime"])

        for step_index, visit in enumerate(visits):
            current_intime_dt = visit.get("intime_dt")
            delta_hours = (
                0.0
                if step_index == 0 or previous_intime_dt is None or current_intime_dt is None
                else float(hours_between(previous_intime_dt, current_intime_dt))
            )
            step = {
                "step_index": int(step_index),
                "diagnosis_ids": list(visit["diagnosis_ids"]),
                "procedure_ids": list(visit["procedure_ids"]),
                "lab_values": [],
                "lab_mask": [],
                "vital_values": [],
                "vital_mask": [],
                "med_history_ids": cumulative_history[:max_med_history],
                "delta_hours": float(delta_hours),
                "target_drugs": list(visit["target_drugs"]),
            }
            trajectory_steps.append(step)
            trajectory_lengths.append(step_index + 1)
            visit_target_sizes.append(len(step["target_drugs"]))

            records_by_split[split].append(
                {
                    "patient_id": int(subject_id),
                    "subject_id": int(subject_id),
                    "hadm_id": int(visit["hadm_id"]),
                    "stay_id": int(visit["stay_id"]),
                    "split": split,
                    "intime": trajectory_start_time,
                    "outtime": str(visit["outtime"]),
                    "num_steps": int(step_index + 1),
                    "visit_index": int(step_index),
                    "visit_position": int(step_index + 1),
                    "history_length": int(step_index + 1),
                    "patient_num_visits": int(patient_num_visits),
                    "drug_vocab_size": int(drug_vocab_size),
                    "drug_representation": "med_vocab_main",
                    "lab_feature_size": 0,
                    "vital_feature_size": 0,
                    "steps": [_copy_step(existing_step) for existing_step in trajectory_steps],
                }
            )

            for drug_id in reversed(step["target_drugs"]):
                if drug_id in cumulative_history:
                    cumulative_history.remove(drug_id)
                cumulative_history.insert(0, int(drug_id))
            cumulative_history = cumulative_history[:max_med_history]
            previous_intime_dt = current_intime_dt
            diagnostics["num_unique_visits"] += 1

    num_trajectories = sum(len(records) for records in records_by_split.values())
    num_unique_visits = int(diagnostics["num_unique_visits"])
    empty_visit_count = sum(1 for size in visit_target_sizes if int(size) <= 0)
    summary = {
        "num_trajectories": int(num_trajectories),
        "num_unique_visits": int(num_unique_visits),
        "trajectory_length_min": int(min(trajectory_lengths)) if trajectory_lengths else 0,
        "trajectory_length_max": int(max(trajectory_lengths)) if trajectory_lengths else 0,
        "trajectory_length_avg": (
            round(sum(trajectory_lengths) / len(trajectory_lengths), 4) if trajectory_lengths else 0.0
        ),
        "avg_num_meds_per_visit": (
            round(sum(visit_target_sizes) / len(visit_target_sizes), 4) if visit_target_sizes else 0.0
        ),
        "empty_visit_rate": (
            round(empty_visit_count / len(visit_target_sizes), 6) if visit_target_sizes else 0.0
        ),
        "empty_visit_count": int(empty_visit_count),
        "counts_by_split": {split: len(records) for split, records in sorted(records_by_split.items())},
    }
    return records_by_split, summary


def build_trajectories(config_path: str | Path) -> dict[str, Path]:
    _configure_logging()
    config = load_yaml_config(config_path)
    trajectory_root = _trajectory_output_root(config)
    vocab_bundle = _load_vocab_bundle_for_trajectories(config)
    med_vocab_metadata = _load_med_vocab_main_metadata(config)
    feature_cfg = dict(config.get("features", {}))
    max_med_history = int(feature_cfg.get("max_med_history", 32))
    if max_med_history < 1:
        raise ValueError(f"features.max_med_history must be >= 1, got {max_med_history}")

    cohort_path = _cohort_path_from_config(config)
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Cohort artifact is missing at {cohort_path}. Run build_cohort.py first."
        )

    primary_med_vocab = vocab_bundle["med_main"]
    diag_token_to_idx = {str(token): int(index) for token, index in vocab_bundle["diagnosis"]["token_to_idx"].items()}
    proc_token_to_idx = {str(token): int(index) for token, index in vocab_bundle["procedure"]["token_to_idx"].items()}
    raw_medication_token_to_ids = _build_raw_medication_token_to_ids(vocab_bundle, med_vocab_metadata)

    LOGGER.info("Loading cohort rows from %s", cohort_path)
    cohort_rows = read_csv_gz(cohort_path)
    visit_encoding_diagnostics: Counter[str] = Counter()
    encoded_visits_by_subject: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in cohort_rows:
        subject_id_text = str(row.get("subject_id", "")).strip()
        hadm_id_text = str(row.get("hadm_id", "")).strip()
        stay_id_text = str(row.get("stay_id", "")).strip()
        if not subject_id_text or not hadm_id_text or not stay_id_text:
            continue
        encoded_visit = _encode_visit_row(
            row,
            diag_token_to_idx=diag_token_to_idx,
            proc_token_to_idx=proc_token_to_idx,
            raw_medication_token_to_ids=raw_medication_token_to_ids,
            max_med_history=max_med_history,
            diagnostics=visit_encoding_diagnostics,
        )
        encoded_visits_by_subject[int(encoded_visit["subject_id"])].append(encoded_visit)

    records_by_split, trajectory_summary = _build_patient_prefix_trajectories(
        encoded_visits_by_subject,
        drug_vocab_size=len(primary_med_vocab["idx_to_token"]),
        max_med_history=max_med_history,
    )

    output_paths: dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        split_path = _trajectory_file(trajectory_root, split_name)
        output_paths[split_name] = write_jsonl_gz(split_path, records_by_split.get(split_name, []))

    metadata = {
        "benchmark_unit": "patient_visit_prefix",
        "trajectory_definition": "each record is one patient prefix ending at the current visit",
        "drug_representation": "med_vocab_main",
        "drug_vocab_size": int(primary_med_vocab["size"]),
        "drug_vocab_name": str(primary_med_vocab.get("name", "med_vocab_main")),
        "lab_feature_size": 0,
        "vital_feature_size": 0,
        "max_med_history": int(max_med_history),
        "counts_by_split": {
            split_name: len(records_by_split.get(split_name, []))
            for split_name in ("train", "val", "test")
        },
    }
    summary = {
        **trajectory_summary,
        "visit_encoding": {key: int(value) for key, value in sorted(visit_encoding_diagnostics.items())},
    }

    write_json(trajectory_root / "metadata.json", metadata)
    write_json(trajectory_root / "trajectory_summary.json", summary)

    LOGGER.info("Built trajectories at %s", trajectory_root)
    LOGGER.info("Trajectories: %s", summary["num_trajectories"])
    LOGGER.info(
        "Trajectory length min/avg/max: %s / %.4f / %s",
        summary["trajectory_length_min"],
        summary["trajectory_length_avg"],
        summary["trajectory_length_max"],
    )
    LOGGER.info("Avg meds/visit: %.4f", summary["avg_num_meds_per_visit"])
    LOGGER.info("Empty visit rate: %.6f", summary["empty_visit_rate"])
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build patient-level ordered visit trajectories aligned to med_vocab_main."
    )
    parser.add_argument("--config", default="configs/data.yaml", help="Path to configs/data.yaml")
    args = parser.parse_args()
    build_trajectories(args.config)


if __name__ == "__main__":
    main()
