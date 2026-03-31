from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.data.load_mimic import (
    MIMICDataPaths,
    build_spark_session,
    read_table_spark,
    spark_config,
)
from src.utils.io import (
    ensure_dir,
    fingerprint_path,
    fingerprint_payload,
    load_yaml_config,
    read_json,
    resolve_path,
    write_json,
)


SCHEMA_VERSION = 1
STAGED_TABLES = ("diagnoses_icd", "procedures_icd", "labevents", "chartevents", "medications")


def _cohort_csv_path(config: dict) -> Path:
    interim_root = resolve_path(config["_project_root"], config["paths"]["interim_root"])
    return Path(interim_root) / "cohort" / "cohort.csv.gz"


def stage_cache_dir_from_config(config: dict) -> Path:
    spark_cfg = spark_config(config)
    return ensure_dir(resolve_path(config["_project_root"], spark_cfg["stage_cache_dir"]))


def cache_manifest_path_from_config(config: dict) -> Path:
    return stage_cache_dir_from_config(config) / "cache_manifest.json"


def _cohort_keys_path(config: dict) -> Path:
    interim_root = resolve_path(config["_project_root"], config["paths"]["interim_root"])
    return Path(interim_root) / "cohort" / "cohort_keys.parquet"


def _config_fingerprint(config: dict) -> str:
    relevant = {
        "paths": {"raw_root": config.get("paths", {}).get("raw_root", "")},
        "cohort": config.get("cohort", {}),
        "spark": spark_config(config),
    }
    return fingerprint_payload(relevant)


def _cohort_fingerprint(config: dict) -> str:
    cohort_csv = _cohort_csv_path(config)
    cohort_keys = _cohort_keys_path(config)
    return fingerprint_payload(
        {
            "cohort_csv": fingerprint_path(cohort_csv),
            "cohort_keys": fingerprint_path(cohort_keys),
        }
    )


def _table_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.rglob("*.parquet") if item.is_file())


def _table_manifest_entry(path: Path, rows: int) -> dict[str, object]:
    return {
        "path": path.name,
        "rows": int(rows),
        "files": [item.name for item in _table_files(path)],
    }


def cache_is_complete(config: dict, manifest: dict | None = None) -> bool:
    manifest_payload = manifest or (
        read_json(cache_manifest_path_from_config(config))
        if cache_manifest_path_from_config(config).exists()
        else None
    )
    if not manifest_payload:
        return False
    if int(manifest_payload.get("schema_version", -1)) != SCHEMA_VERSION:
        return False
    if manifest_payload.get("config_fingerprint") != _config_fingerprint(config):
        return False
    if manifest_payload.get("cohort_fingerprint") != _cohort_fingerprint(config):
        return False
    tables = manifest_payload.get("tables", {})
    cache_dir = stage_cache_dir_from_config(config)
    for table_name in STAGED_TABLES:
        entry = tables.get(table_name)
        table_dir = cache_dir / table_name
        if not entry or not table_dir.exists():
            return False
        if not _table_files(table_dir):
            return False
    return True


def require_stage_cache(config: dict) -> tuple[Path, dict]:
    cache_dir = stage_cache_dir_from_config(config)
    manifest_path = cache_manifest_path_from_config(config)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Spark cache manifest is missing at {manifest_path}. "
            f"Run `python -m src.data.stage_filtered_tables --config {config['_config_path']}` first."
        )
    manifest = read_json(manifest_path)
    if not cache_is_complete(config, manifest):
        raise FileNotFoundError(
            f"Spark cache at {cache_dir} is missing, stale, or incomplete. "
            f"Run `python -m src.data.stage_filtered_tables --config {config['_config_path']}` first."
        )
    return cache_dir, manifest


def _canonicalize_text_expr(F, column, prefix: str):
    cleaned = F.regexp_replace(F.upper(F.trim(column)), r"[^A-Z0-9]+", "_")
    cleaned = F.regexp_replace(cleaned, r"^_+|_+$", "")
    return F.when(F.length(cleaned) > 0, F.concat(F.lit(prefix), F.lit(":"), cleaned))


def _write_table(dataframe, path: Path, *, coalesce_to: int | None = None, repartition_to: tuple[int, str] | None = None) -> int:
    staged = dataframe
    if repartition_to is not None:
        partitions, column_name = repartition_to
        staged = staged.repartition(int(partitions), column_name)
    elif coalesce_to is not None:
        staged = staged.coalesce(int(coalesce_to))
    row_count = int(staged.count())
    (
        staged.write.mode("overwrite")
        .option("compression", "snappy")
        .option("maxRecordsPerFile", 2_000_000)
        .parquet(str(path))
    )
    return row_count


def stage_filtered_tables(config_path: str | Path, *, overwrite: bool = False) -> Path:
    config = load_yaml_config(config_path)
    cache_dir = stage_cache_dir_from_config(config)
    manifest_path = cache_manifest_path_from_config(config)
    if not overwrite and manifest_path.exists():
        manifest = read_json(manifest_path)
        if cache_is_complete(config, manifest):
            return cache_dir

    cohort_keys_path = _cohort_keys_path(config)
    if not cohort_keys_path.exists():
        raise FileNotFoundError(
            f"Cohort keys parquet is missing at {cohort_keys_path}. "
            f"Run `python -m src.data.build_cohort --config {config['_config_path']}` first."
        )

    if cache_dir.exists():
        for table_name in STAGED_TABLES:
            shutil.rmtree(cache_dir / table_name, ignore_errors=True)
        if manifest_path.exists():
            manifest_path.unlink()

    spark = build_spark_session(config, app_name="stage-filtered-mimic")
    try:
        from pyspark.sql import functions as F

        paths = MIMICDataPaths.from_config(config)
        cohort = (
            spark.read.parquet(str(cohort_keys_path))
            .select("subject_id", "hadm_id", "stay_id", "split", "intime", "outtime")
            .withColumn("subject_id", F.col("subject_id").cast("long"))
            .withColumn("hadm_id", F.col("hadm_id").cast("long"))
            .withColumn("stay_id", F.col("stay_id").cast("long"))
            .withColumn("intime", F.to_timestamp("intime"))
            .withColumn("outtime", F.to_timestamp("outtime"))
        )
        hadm_keys = F.broadcast(cohort.select("hadm_id").dropDuplicates())
        stay_keys = F.broadcast(cohort.select("stay_id").dropDuplicates())

        diagnoses = (
            read_table_spark(spark, paths, "diagnoses_icd")
            .select(
                F.col("hadm_id").cast("long").alias("hadm_id"),
                F.trim(F.col("icd_code")).alias("icd_code"),
                F.trim(F.col("icd_version")).alias("icd_version"),
            )
            .join(hadm_keys, "hadm_id", "inner")
            .filter(F.col("icd_code") != "")
            .filter(F.col("icd_version") != "")
            .withColumn(
                "diagnosis_token",
                F.concat(F.lit("ICD"), F.col("icd_version"), F.lit(":"), F.col("icd_code")),
            )
        )

        procedures = (
            read_table_spark(spark, paths, "procedures_icd")
            .select(
                F.col("hadm_id").cast("long").alias("hadm_id"),
                F.trim(F.col("icd_code")).alias("icd_code"),
                F.trim(F.col("icd_version")).alias("icd_version"),
                F.to_timestamp("chartdate").alias("event_time"),
            )
            .join(hadm_keys, "hadm_id", "inner")
            .filter(F.col("icd_code") != "")
            .filter(F.col("icd_version") != "")
            .withColumn(
                "procedure_token",
                F.concat(F.lit("PROC"), F.col("icd_version"), F.lit(":"), F.col("icd_code")),
            )
        )

        labevents = (
            read_table_spark(spark, paths, "labevents")
            .select(
                F.col("hadm_id").cast("long").alias("hadm_id"),
                F.trim(F.col("itemid")).alias("itemid"),
                F.col("valuenum").cast("double").alias("valuenum"),
                F.to_timestamp("charttime").alias("event_time"),
            )
            .join(hadm_keys, "hadm_id", "inner")
            .filter(F.col("itemid") != "")
            .filter(F.col("valuenum").isNotNull())
        )

        chartevents = (
            read_table_spark(spark, paths, "chartevents")
            .select(
                F.col("stay_id").cast("long").alias("stay_id"),
                F.trim(F.col("itemid")).alias("itemid"),
                F.col("valuenum").cast("double").alias("valuenum"),
                F.to_timestamp("charttime").alias("event_time"),
            )
            .join(stay_keys, "stay_id", "inner")
            .filter(F.col("itemid") != "")
            .filter(F.col("valuenum").isNotNull())
        )

        emar = (
            read_table_spark(spark, paths, "emar")
            .select(
                F.col("hadm_id").cast("long").alias("hadm_id"),
                F.coalesce(
                    F.to_timestamp("charttime"),
                    F.to_timestamp("scheduletime"),
                    F.to_timestamp("storetime"),
                ).alias("event_time"),
                _canonicalize_text_expr(F, F.col("medication"), "NAME").alias("drug_token"),
                F.lit("emar").alias("source_table"),
            )
            .join(hadm_keys, "hadm_id", "inner")
        )
        prescriptions = (
            read_table_spark(spark, paths, "prescriptions")
            .select(
                F.col("hadm_id").cast("long").alias("hadm_id"),
                F.coalesce(F.to_timestamp("starttime"), F.to_timestamp("stoptime")).alias("event_time"),
                F.coalesce(
                    _canonicalize_text_expr(F, F.col("drug"), "NAME"),
                    _canonicalize_text_expr(F, F.col("formulary_drug_cd"), "CODE"),
                ).alias("drug_token"),
                F.lit("prescriptions").alias("source_table"),
            )
            .join(hadm_keys, "hadm_id", "inner")
        )
        pharmacy = (
            read_table_spark(spark, paths, "pharmacy")
            .select(
                F.col("hadm_id").cast("long").alias("hadm_id"),
                F.coalesce(
                    F.to_timestamp("starttime"),
                    F.to_timestamp("verifiedtime"),
                    F.to_timestamp("entertime"),
                ).alias("event_time"),
                _canonicalize_text_expr(F, F.col("medication"), "NAME").alias("drug_token"),
                F.lit("pharmacy").alias("source_table"),
            )
            .join(hadm_keys, "hadm_id", "inner")
        )
        medications = (
            emar.unionByName(prescriptions).unionByName(pharmacy)
            .filter(F.col("drug_token").isNotNull())
            .filter(F.col("event_time").isNotNull())
        )

        table_rows = {
            "diagnoses_icd": _write_table(diagnoses, cache_dir / "diagnoses_icd", coalesce_to=4),
            "procedures_icd": _write_table(procedures, cache_dir / "procedures_icd", coalesce_to=4),
            "labevents": _write_table(labevents, cache_dir / "labevents", repartition_to=(8, "hadm_id")),
            "chartevents": _write_table(chartevents, cache_dir / "chartevents", repartition_to=(8, "stay_id")),
            "medications": _write_table(medications, cache_dir / "medications", coalesce_to=4),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "config_fingerprint": _config_fingerprint(config),
            "cohort_fingerprint": _cohort_fingerprint(config),
            "tables": {
                table_name: _table_manifest_entry(cache_dir / table_name, row_count)
                for table_name, row_count in table_rows.items()
            },
        }
        write_json(manifest_path, manifest)
        return cache_dir
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage cohort-filtered MIMIC tables into parquet cache.")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild the spark cache even if it looks valid.")
    args = parser.parse_args()
    stage_filtered_tables(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
