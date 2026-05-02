from __future__ import annotations

import csv
import gzip
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from src.utils.io import _set_max_csv_field_size, load_yaml_config, parse_datetime, resolve_path


TABLE_PATHS = {
    "patients": ("hosp", "patients.csv.gz"),
    "admissions": ("hosp", "admissions.csv.gz"),
    "transfers": ("hosp", "transfers.csv.gz"),
    "diagnoses_icd": ("hosp", "diagnoses_icd.csv.gz"),
    "d_icd_diagnoses": ("hosp", "d_icd_diagnoses.csv.gz"),
    "procedures_icd": ("hosp", "procedures_icd.csv.gz"),
    "d_icd_procedures": ("hosp", "d_icd_procedures.csv.gz"),
    "labevents": ("hosp", "labevents.csv.gz"),
    "d_labitems": ("hosp", "d_labitems.csv.gz"),
    "prescriptions": ("hosp", "prescriptions.csv.gz"),
    "emar": ("hosp", "emar.csv.gz"),
    "emar_detail": ("hosp", "emar_detail.csv.gz"),
    "pharmacy": ("hosp", "pharmacy.csv.gz"),
    "icustays": ("icu", "icustays.csv.gz"),
    "chartevents": ("icu", "chartevents.csv.gz"),
    "d_items": ("icu", "d_items.csv.gz"),
}

SPARK_DEFAULTS = {
    "enabled": True,
    "master": "local[4]",
    "driver_memory": "3g",
    "default_parallelism": 8,
    "sql_shuffle_partitions": 24,
    "adaptive_enabled": True,
    "adaptive_coalesce_enabled": True,
    "files_max_partition_bytes": "64m",
    "local_dir": "/tmp/healthdm-spark",
    "stage_cache_dir": "data/interim/spark_cache",
    "cache_codec": "snappy",
    "trajectory_rows_per_file": 2048,
    "max_open_shards_per_dataset": 2,
}

SPARK_TABLE_COLUMNS = {
    "icustays": ["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    "patients": ["subject_id", "gender", "anchor_age", "anchor_year"],
    "admissions": [
        "subject_id",
        "hadm_id",
        "admission_type",
        "insurance",
        "language",
        "marital_status",
        "race",
        "hospital_expire_flag",
    ],
    "diagnoses_icd": ["hadm_id", "icd_code", "icd_version"],
    "procedures_icd": ["hadm_id", "chartdate", "icd_code", "icd_version"],
    "labevents": ["hadm_id", "charttime", "itemid", "valuenum"],
    "d_labitems": ["itemid", "label", "category", "fluid"],
    "prescriptions": ["hadm_id", "drug", "formulary_drug_cd", "starttime", "stoptime"],
    "emar": ["hadm_id", "medication", "charttime", "scheduletime", "storetime"],
    "pharmacy": ["hadm_id", "medication", "starttime", "verifiedtime", "entertime"],
    "chartevents": ["stay_id", "charttime", "itemid", "valuenum"],
    "d_items": ["itemid", "label", "category", "unitname"],
}


@dataclass(frozen=True)
class MIMICDataPaths:
    raw_root: Path

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "MIMICDataPaths":
        raw_root = resolve_path(config["_project_root"], config["paths"]["raw_root"])
        return cls(raw_root=Path(raw_root))

    @classmethod
    def from_config_path(cls, config_path: str | Path) -> "MIMICDataPaths":
        return cls.from_config(load_yaml_config(config_path))

    def table_path(self, table_name: str) -> Path:
        if table_name not in TABLE_PATHS:
            raise KeyError(f"Unknown MIMIC table: {table_name}")
        group, filename = TABLE_PATHS[table_name]
        return self.raw_root / group / filename


def open_csv(path: str | Path):
    csv_path = Path(path)
    if csv_path.suffix == ".gz":
        return gzip.open(csv_path, "rt", encoding="utf-8", newline="")
    return csv_path.open("r", encoding="utf-8", newline="")


def iter_csv_rows(
    path: str | Path,
    *,
    fields: Iterable[str] | None = None,
) -> Iterator[dict[str, str]]:
    selected = list(fields) if fields else None
    with open_csv(path) as handle:
        _set_max_csv_field_size()
        reader = csv.DictReader(handle)
        for row in reader:
            if selected is None:
                yield row
            else:
                yield {field: row.get(field, "") for field in selected}


def iter_table(
    paths: MIMICDataPaths,
    table_name: str,
    *,
    fields: Iterable[str] | None = None,
) -> Iterator[dict[str, str]]:
    yield from iter_csv_rows(paths.table_path(table_name), fields=fields)


def read_lookup(
    paths: MIMICDataPaths,
    table_name: str,
    key_field: str,
    value_fields: Iterable[str],
) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    requested = [key_field, *list(value_fields)]
    for row in iter_table(paths, table_name, fields=requested):
        key = str(row.get(key_field, "")).strip()
        if key:
            lookup[key] = {field: row.get(field, "") for field in value_fields}
    return lookup


def choose_stay_for_event(
    stays: list[dict[str, object]],
    event_time,
) -> dict[str, object] | None:
    if not stays:
        return None
    if len(stays) == 1 or event_time is None:
        return stays[0]

    containing: list[dict[str, object]] = []
    for stay in stays:
        if stay["intime_dt"] <= event_time <= stay["outtime_dt"]:
            containing.append(stay)
    if containing:
        containing.sort(key=lambda item: item["intime_dt"])
        return containing[0]

    def distance_hours(stay: Mapping[str, object]) -> float:
        start = stay["intime_dt"]
        end = stay["outtime_dt"]
        if event_time < start:
            return (start - event_time).total_seconds() / 3600.0
        return (event_time - end).total_seconds() / 3600.0

    return min(stays, key=distance_hours)


def coerce_event_time(row: Mapping[str, str], candidate_fields: Iterable[str]):
    for field in candidate_fields:
        value = row.get(field, "")
        dt_value = parse_datetime(value)
        if dt_value is not None:
            return dt_value
    return None


def spark_config(config: Mapping[str, object]) -> dict[str, object]:
    merged = dict(SPARK_DEFAULTS)
    merged.update(config.get("spark", {}) or {})
    return merged


def spark_enabled(config: Mapping[str, object]) -> bool:
    return bool(spark_config(config).get("enabled", True))


def build_spark_session(config: Mapping[str, object], *, app_name: str):
    spark_cfg = spark_config(config)
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise RuntimeError(
            "PySpark is required for Spark preprocessing. Install requirements.txt first."
        ) from exc

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")

    builder = (
        SparkSession.builder.appName(app_name)
        .master(str(spark_cfg["master"]))
        .config("spark.driver.memory", str(spark_cfg["driver_memory"]))
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.default.parallelism", str(spark_cfg["default_parallelism"]))
        .config("spark.sql.shuffle.partitions", str(spark_cfg["sql_shuffle_partitions"]))
        .config("spark.sql.adaptive.enabled", str(spark_cfg["adaptive_enabled"]).lower())
        .config(
            "spark.sql.adaptive.coalescePartitions.enabled",
            str(spark_cfg["adaptive_coalesce_enabled"]).lower(),
        )
        .config("spark.sql.files.maxPartitionBytes", str(spark_cfg["files_max_partition_bytes"]))
        .config("spark.local.dir", str(spark_cfg["local_dir"]))
        .config("spark.sql.parquet.compression.codec", str(spark_cfg["cache_codec"]))
    )
    return builder.getOrCreate()


def read_table_spark(
    spark,
    paths: MIMICDataPaths,
    table_name: str,
    *,
    columns: Iterable[str] | None = None,
):
    requested = list(columns) if columns else list(SPARK_TABLE_COLUMNS.get(table_name, ()))
    dataframe = (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .csv(str(paths.table_path(table_name)))
    )
    if requested:
        missing = [column for column in requested if column not in dataframe.columns]
        if missing:
            raise KeyError(f"Missing columns in {table_name}: {missing}")
        dataframe = dataframe.select(*requested)
    return dataframe
