from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from src.data.load_mimic import MIMICDataPaths, iter_table, read_lookup
from src.utils.io import (
    assign_split,
    ensure_dir,
    hours_between,
    load_yaml_config,
    parse_datetime,
    parse_int,
    resolve_path,
    write_csv_gz,
    write_json,
)


COHORT_FIELDS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "intime",
    "outtime",
    "los_hours",
    "split",
    "admission_type",
    "insurance",
    "language",
    "marital_status",
    "race",
    "hospital_expire_flag",
    "gender",
    "anchor_age",
    "anchor_year",
]


def _collect_hadm_presence(paths: MIMICDataPaths, table_name: str) -> set[int]:
    present: set[int] = set()
    for row in iter_table(paths, table_name, fields=["hadm_id"]):
        hadm_id = parse_int(row.get("hadm_id"))
        if hadm_id is not None:
            present.add(hadm_id)
    return present


def _collect_medication_hadm(paths: MIMICDataPaths) -> set[int]:
    medication_hadm: set[int] = set()
    for table_name in ("emar", "prescriptions", "pharmacy"):
        medication_hadm.update(_collect_hadm_presence(paths, table_name))
    return medication_hadm


def _cohort_output_paths(config: dict) -> tuple[Path, Path]:
    interim_root = resolve_path(config["_project_root"], config["paths"]["interim_root"])
    cohort_dir = ensure_dir(Path(interim_root) / "cohort")
    split_dir = ensure_dir(cohort_dir / "splits")
    return cohort_dir, split_dir


def build_cohort(config_path: str | Path) -> Path:
    config = load_yaml_config(config_path)
    paths = MIMICDataPaths.from_config(config)
    cohort_dir, split_dir = _cohort_output_paths(config)

    seed = int(config.get("seed", 42))
    split_cfg = config.get("split", {})
    cohort_cfg = config.get("cohort", {})
    min_los_hours = float(cohort_cfg.get("min_los_hours", 0.0))
    require_diagnosis = bool(cohort_cfg.get("require_diagnosis", True))
    require_medication = bool(cohort_cfg.get("require_medication", True))

    diagnosis_hadm = _collect_hadm_presence(paths, "diagnoses_icd")
    medication_hadm = _collect_medication_hadm(paths)
    admissions = read_lookup(
        paths,
        "admissions",
        "hadm_id",
        [
            "admission_type",
            "insurance",
            "language",
            "marital_status",
            "race",
            "hospital_expire_flag",
        ],
    )
    patients = read_lookup(
        paths,
        "patients",
        "subject_id",
        ["gender", "anchor_age", "anchor_year"],
    )

    candidates: list[dict[str, object]] = []
    for row in iter_table(
        paths,
        "icustays",
        fields=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    ):
        subject_id = parse_int(row.get("subject_id"))
        hadm_id = parse_int(row.get("hadm_id"))
        stay_id = parse_int(row.get("stay_id"))
        intime = parse_datetime(row.get("intime"))
        outtime = parse_datetime(row.get("outtime"))
        if None in (subject_id, hadm_id, stay_id) or intime is None or outtime is None:
            continue
        if require_diagnosis and hadm_id not in diagnosis_hadm:
            continue
        if require_medication and hadm_id not in medication_hadm:
            continue

        los_hours = hours_between(intime, outtime)
        if los_hours < min_los_hours:
            continue

        admission = admissions.get(str(hadm_id), {})
        patient = patients.get(str(subject_id), {})
        candidates.append(
            {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "stay_id": stay_id,
                "intime": intime.strftime("%Y-%m-%d %H:%M:%S"),
                "outtime": outtime.strftime("%Y-%m-%d %H:%M:%S"),
                "los_hours": round(los_hours, 4),
                "admission_type": admission.get("admission_type", ""),
                "insurance": admission.get("insurance", ""),
                "language": admission.get("language", ""),
                "marital_status": admission.get("marital_status", ""),
                "race": admission.get("race", ""),
                "hospital_expire_flag": admission.get("hospital_expire_flag", ""),
                "gender": patient.get("gender", ""),
                "anchor_age": patient.get("anchor_age", ""),
                "anchor_year": patient.get("anchor_year", ""),
            }
        )

    candidates.sort(key=lambda item: (int(item["hadm_id"]), item["intime"], int(item["stay_id"])))
    cohort_rows: list[dict[str, object]] = []
    seen_stays: set[int] = set()
    seen_hadm: set[int] = set()
    split_subjects: dict[str, set[int]] = defaultdict(set)
    for row in candidates:
        stay_id = int(row["stay_id"])
        hadm_id = int(row["hadm_id"])
        subject_id = int(row["subject_id"])
        if stay_id in seen_stays or hadm_id in seen_hadm:
            continue
        split_name = assign_split(subject_id, split_cfg, seed)
        row["split"] = split_name
        seen_stays.add(stay_id)
        seen_hadm.add(hadm_id)
        split_subjects[split_name].add(subject_id)
        cohort_rows.append(row)

    cohort_path = write_csv_gz(cohort_dir / "cohort.csv.gz", cohort_rows, COHORT_FIELDS)
    split_manifest = {
        split_name: sorted(subject_ids)
        for split_name, subject_ids in split_subjects.items()
    }
    for split_name in ("train", "val", "test"):
        write_json(split_dir / f"{split_name}_subject_ids.json", split_manifest.get(split_name, []))
    write_json(
        cohort_dir / "cohort_summary.json",
        {
            "num_rows": len(cohort_rows),
            "num_subjects": len({int(row["subject_id"]) for row in cohort_rows}),
            "split_subjects": {k: len(v) for k, v in split_subjects.items()},
        },
    )
    return cohort_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ICU cohort for the drug recommendation pipeline.")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    args = parser.parse_args()
    build_cohort(args.config)


if __name__ == "__main__":
    main()
