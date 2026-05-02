from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    def tqdm(iterable: Any, *args: Any, **kwargs: Any) -> Any:
        return iterable

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
    write_parquet_pylist,
    write_json,
)


LOGGER = logging.getLogger(__name__)

COHORT_FIELDS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "stay_id_source",
    "visit_order",
    "patient_num_visits",
    "admittime",
    "dischtime",
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
    "num_diagnoses",
    "num_procedures",
    "num_medications",
    "diagnosis_codes",
    "procedure_codes",
    "raw_medication_records",
]

PRESCRIPTION_FIELDS = (
    "subject_id",
    "hadm_id",
    "starttime",
    "stoptime",
    "drug",
    "formulary_drug_cd",
    "ndc",
    "route",
    "drug_type",
)


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


def _cohort_output_paths(config: dict) -> tuple[Path, Path]:
    cohort_root = config["paths"].get("cohort_root")
    if cohort_root:
        cohort_dir = ensure_dir(resolve_path(config["_project_root"], cohort_root))
    else:
        interim_root = resolve_path(config["_project_root"], config["paths"]["interim_root"])
        cohort_dir = ensure_dir(Path(interim_root) / "cohort")
    split_dir = ensure_dir(cohort_dir / "splits")
    return cohort_dir, split_dir


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        output.append(normalized)
        seen.add(normalized)
    return output


def _collect_primary_stay_by_hadm(paths: MIMICDataPaths) -> dict[int, int]:
    icustays_path = paths.table_path("icustays")
    if not icustays_path.exists():
        return {}

    primary_stay_by_hadm: dict[int, tuple[str, int]] = {}
    for row in tqdm(
        iter_table(paths, "icustays", fields=["hadm_id", "stay_id", "intime"]),
        desc="Scan icustays for compatibility stay IDs",
        unit="row",
    ):
        hadm_id = parse_int(row.get("hadm_id"))
        stay_id = parse_int(row.get("stay_id"))
        if hadm_id is None or stay_id is None:
            continue
        intime = parse_datetime(row.get("intime"))
        sort_key = (
            intime.strftime("%Y-%m-%d %H:%M:%S")
            if intime is not None
            else "9999-12-31 23:59:59"
        )
        current = primary_stay_by_hadm.get(hadm_id)
        candidate = (sort_key, stay_id)
        if current is None or candidate < current:
            primary_stay_by_hadm[hadm_id] = candidate
    return {hadm_id: stay_id for hadm_id, (_, stay_id) in primary_stay_by_hadm.items()}


def _collect_diagnosis_codes_by_hadm(paths: MIMICDataPaths) -> dict[int, list[str]]:
    grouped: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for row in tqdm(
        iter_table(paths, "diagnoses_icd", fields=["hadm_id", "seq_num", "icd_code", "icd_version"]),
        desc="Collect diagnosis codes",
        unit="row",
    ):
        hadm_id = parse_int(row.get("hadm_id"))
        icd_code = str(row.get("icd_code", "")).strip()
        icd_version = str(row.get("icd_version", "")).strip()
        if hadm_id is None or not icd_code or not icd_version:
            continue
        seq_num = parse_int(row.get("seq_num"), default=10**9) or 10**9
        grouped[hadm_id].append((seq_num, f"ICD{icd_version}:{icd_code}"))

    output: dict[int, list[str]] = {}
    for hadm_id, records in grouped.items():
        sorted_codes = [code for _, code in sorted(records, key=lambda item: (item[0], item[1]))]
        output[hadm_id] = _dedupe_preserve_order(sorted_codes)
    return output


def _collect_procedure_codes_by_hadm(paths: MIMICDataPaths) -> dict[int, list[str]]:
    grouped: dict[int, list[tuple[str, int, str]]] = defaultdict(list)
    for row in tqdm(
        iter_table(paths, "procedures_icd", fields=["hadm_id", "chartdate", "seq_num", "icd_code", "icd_version"]),
        desc="Collect procedure codes",
        unit="row",
    ):
        hadm_id = parse_int(row.get("hadm_id"))
        icd_code = str(row.get("icd_code", "")).strip()
        icd_version = str(row.get("icd_version", "")).strip()
        if hadm_id is None or not icd_code or not icd_version:
            continue
        chartdate = parse_datetime(row.get("chartdate"))
        chartdate_key = chartdate.strftime("%Y-%m-%d %H:%M:%S") if chartdate is not None else ""
        seq_num = parse_int(row.get("seq_num"), default=10**9) or 10**9
        grouped[hadm_id].append((chartdate_key, seq_num, f"PROC{icd_version}:{icd_code}"))

    output: dict[int, list[str]] = {}
    for hadm_id, records in grouped.items():
        sorted_codes = [
            code
            for _, _, code in sorted(records, key=lambda item: (item[0], item[1], item[2]))
        ]
        output[hadm_id] = _dedupe_preserve_order(sorted_codes)
    return output


def _collect_prescriptions_by_hadm(paths: MIMICDataPaths) -> dict[int, list[dict[str, str]]]:
    prescriptions_by_hadm: dict[int, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    for row in tqdm(
        iter_table(paths, "prescriptions", fields=PRESCRIPTION_FIELDS),
        desc="Collect raw medication records",
        unit="row",
    ):
        hadm_id = parse_int(row.get("hadm_id"))
        if hadm_id is None:
            continue

        medication_record = {
            "subject_id": str(row.get("subject_id", "")).strip(),
            "starttime": str(row.get("starttime", "")).strip(),
            "stoptime": str(row.get("stoptime", "")).strip(),
            "drug": str(row.get("drug", "")).strip(),
            "formulary_drug_cd": str(row.get("formulary_drug_cd", "")).strip(),
            "ndc": str(row.get("ndc", "")).strip(),
            "route": str(row.get("route", "")).strip(),
            "drug_type": str(row.get("drug_type", "")).strip(),
        }
        sort_time = parse_datetime(medication_record["starttime"]) or parse_datetime(medication_record["stoptime"])
        sort_key = sort_time.strftime("%Y-%m-%d %H:%M:%S") if sort_time is not None else ""
        prescriptions_by_hadm[hadm_id].append((sort_key, medication_record))

    output: dict[int, list[dict[str, str]]] = {}
    for hadm_id, records in prescriptions_by_hadm.items():
        output[hadm_id] = [
            record
            for _, record in sorted(
                records,
                key=lambda item: (
                    item[0],
                    item[1]["drug"],
                    item[1]["formulary_drug_cd"],
                    item[1]["ndc"],
                ),
            )
        ]
    return output


def _write_cohort_keys(path: Path, cohort_rows: list[dict[str, object]]) -> None:
    records = [
        {
            "subject_id": int(row["subject_id"]),
            "hadm_id": int(row["hadm_id"]),
            "stay_id": int(row["stay_id"]),
            "split": str(row["split"]),
            "visit_order": int(row["visit_order"]),
            "patient_num_visits": int(row["patient_num_visits"]),
            "intime": str(row["intime"]),
            "outtime": str(row["outtime"]),
            "admittime": str(row["admittime"]),
            "dischtime": str(row["dischtime"]),
        }
        for row in cohort_rows
    ]
    if records:
        write_parquet_pylist(path, records)
        return

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_arrays(
        [
            pa.array([], type=pa.int64()),
            pa.array([], type=pa.int64()),
            pa.array([], type=pa.int64()),
            pa.array([], type=pa.string()),
            pa.array([], type=pa.int64()),
            pa.array([], type=pa.int64()),
            pa.array([], type=pa.string()),
            pa.array([], type=pa.string()),
            pa.array([], type=pa.string()),
            pa.array([], type=pa.string()),
        ],
        names=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "split",
            "visit_order",
            "patient_num_visits",
            "intime",
            "outtime",
            "admittime",
            "dischtime",
        ],
    )
    ensure_dir(path.parent)
    pq.write_table(table, path, compression="snappy")


def build_cohort(config_path: str | Path) -> Path:
    _configure_logging()
    config = load_yaml_config(config_path)
    paths = MIMICDataPaths.from_config(config)
    cohort_dir, split_dir = _cohort_output_paths(config)

    seed = int(config.get("seed", 42))
    split_cfg = config.get("split", {})
    cohort_cfg = config.get("cohort", {})
    min_los_hours = float(cohort_cfg.get("min_los_hours", 0.0))
    require_diagnosis = bool(cohort_cfg.get("require_diagnosis", True))
    require_medication = bool(cohort_cfg.get("require_medication", True))
    min_patient_visits = int(cohort_cfg.get("min_patient_visits", 2))
    if min_patient_visits < 1:
        raise ValueError(f"cohort.min_patient_visits must be >= 1, got {min_patient_visits}")

    admissions = read_lookup(
        paths,
        "admissions",
        "hadm_id",
        [
            "subject_id",
            "admittime",
            "dischtime",
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
    primary_stay_by_hadm = _collect_primary_stay_by_hadm(paths)
    diagnosis_codes_by_hadm = _collect_diagnosis_codes_by_hadm(paths)
    procedure_codes_by_hadm = _collect_procedure_codes_by_hadm(paths)
    prescriptions_by_hadm = _collect_prescriptions_by_hadm(paths)

    stats: Counter[str] = Counter()
    candidate_rows_by_subject: dict[int, list[dict[str, object]]] = defaultdict(list)
    for hadm_id, admission in tqdm(
        sorted(((parse_int(key), value) for key, value in admissions.items()), key=lambda item: (item[0] or 0)),
        desc="Build admission-level cohort visits",
        unit="visit",
    ):
        if hadm_id is None:
            stats["visits_skipped_missing_hadm_id"] += 1
            continue

        subject_id = parse_int(admission.get("subject_id"))
        admittime = parse_datetime(admission.get("admittime"))
        dischtime = parse_datetime(admission.get("dischtime")) or admittime
        if subject_id is None:
            stats["visits_skipped_missing_subject_id"] += 1
            continue
        if admittime is None:
            stats["visits_skipped_missing_admittime"] += 1
            continue

        diagnosis_codes = list(diagnosis_codes_by_hadm.get(hadm_id, []))
        procedure_codes = list(procedure_codes_by_hadm.get(hadm_id, []))
        raw_medication_records = list(prescriptions_by_hadm.get(hadm_id, []))
        if not diagnosis_codes:
            stats["visits_without_diagnosis"] += 1
        if not raw_medication_records:
            stats["visits_without_medications"] += 1

        if require_diagnosis and not diagnosis_codes:
            stats["visits_filtered_missing_diagnosis"] += 1
            continue
        if require_medication and not raw_medication_records:
            stats["visits_filtered_missing_medications"] += 1
            continue

        los_hours = hours_between(admittime, dischtime)
        if los_hours < min_los_hours:
            stats["visits_filtered_min_los"] += 1
            continue

        patient = patients.get(str(subject_id), {})
        primary_stay_id = primary_stay_by_hadm.get(hadm_id)
        stay_id = int(primary_stay_id) if primary_stay_id is not None else -int(hadm_id)
        stay_id_source = "icustays" if primary_stay_id is not None else "synthetic_hadm"
        candidate_rows_by_subject[subject_id].append(
            {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "stay_id": stay_id,
                "stay_id_source": stay_id_source,
                "admittime_dt": admittime,
                "dischtime_dt": dischtime,
                "admittime": admittime.strftime("%Y-%m-%d %H:%M:%S"),
                "dischtime": dischtime.strftime("%Y-%m-%d %H:%M:%S"),
                "intime": admittime.strftime("%Y-%m-%d %H:%M:%S"),
                "outtime": dischtime.strftime("%Y-%m-%d %H:%M:%S"),
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
                "num_diagnoses": int(len(diagnosis_codes)),
                "num_procedures": int(len(procedure_codes)),
                "num_medications": int(len(raw_medication_records)),
                "diagnosis_codes": _json_dumps(diagnosis_codes),
                "procedure_codes": _json_dumps(procedure_codes),
                "raw_medication_records": _json_dumps(raw_medication_records),
            }
        )
        stats["candidate_visits"] += 1

    cohort_rows: list[dict[str, object]] = []
    split_subjects: dict[str, set[int]] = defaultdict(set)
    split_visit_counts: Counter[str] = Counter()
    patient_visit_counts: list[int] = []
    dropped_patient_count = 0
    for subject_id in tqdm(
        sorted(candidate_rows_by_subject),
        desc="Finalize patient trajectories",
        unit="patient",
    ):
        visits = sorted(
            candidate_rows_by_subject[subject_id],
            key=lambda row: (row["admittime_dt"], int(row["hadm_id"]), int(row["stay_id"])),
        )
        if len(visits) < min_patient_visits:
            dropped_patient_count += 1
            stats["patients_filtered_min_visits"] += 1
            stats["visits_filtered_by_patient_min_visits"] += len(visits)
            continue

        split_name = assign_split(subject_id, split_cfg, seed)
        patient_visit_counts.append(len(visits))
        split_subjects[split_name].add(subject_id)
        split_visit_counts[split_name] += len(visits)
        for visit_order, row in enumerate(visits, start=1):
            row["visit_order"] = visit_order
            row["patient_num_visits"] = len(visits)
            row["split"] = split_name
            row.pop("admittime_dt", None)
            row.pop("dischtime_dt", None)
            cohort_rows.append(row)

    cohort_rows.sort(key=lambda row: (int(row["subject_id"]), int(row["visit_order"]), int(row["hadm_id"])))
    cohort_path = write_csv_gz(cohort_dir / "cohort.csv.gz", cohort_rows, COHORT_FIELDS)
    _write_cohort_keys(cohort_dir / "cohort_keys.parquet", cohort_rows)

    split_manifest = {
        split_name: sorted(subject_ids)
        for split_name, subject_ids in split_subjects.items()
    }
    for split_name in ("train", "val", "test"):
        write_json(split_dir / f"{split_name}_subject_ids.json", split_manifest.get(split_name, []))

    num_patients = len(patient_visit_counts)
    num_visits = len(cohort_rows)
    avg_visits_per_patient = round(float(num_visits) / float(num_patients), 4) if num_patients > 0 else 0.0
    summary = {
        "benchmark_unit": "patient_ordered_admissions",
        "num_patients": int(num_patients),
        "num_visits": int(num_visits),
        "avg_visits_per_patient": avg_visits_per_patient,
        "num_patients_excluded_min_visits": int(dropped_patient_count),
        "min_patient_visits": int(min_patient_visits),
        "num_visits_without_medications": int(stats["visits_without_medications"]),
        "num_candidate_visits_before_patient_filter": int(stats["candidate_visits"]),
        "num_patients_before_patient_filter": int(len(candidate_rows_by_subject)),
        "split_subjects": {split_name: len(subject_ids) for split_name, subject_ids in split_subjects.items()},
        "split_visits": {split_name: int(count) for split_name, count in split_visit_counts.items()},
        "filter_stats": {key: int(value) for key, value in sorted(stats.items())},
    }
    write_json(cohort_dir / "cohort_summary.json", summary)

    LOGGER.info("Built benchmark cohort at %s", cohort_path)
    LOGGER.info("Patients: %s", summary["num_patients"])
    LOGGER.info("Visits: %s", summary["num_visits"])
    LOGGER.info("Avg visits/patient: %.4f", summary["avg_visits_per_patient"])
    LOGGER.info("Patients excluded by min visits: %s", summary["num_patients_excluded_min_visits"])
    LOGGER.info("Visits without medications: %s", summary["num_visits_without_medications"])
    return cohort_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an admission-level patient trajectory cohort for the drug recommendation benchmark."
    )
    parser.add_argument("--config", default="configs/data.yaml", help="Path to configs/data.yaml")
    args = parser.parse_args()
    build_cohort(args.config)


if __name__ == "__main__":
    main()
