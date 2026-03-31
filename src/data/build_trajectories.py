from __future__ import annotations

import argparse
import math
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.data.build_vocab import cohort_path_from_config, load_vocab_bundle
from src.data.load_mimic import (
    MIMICDataPaths,
    build_spark_session,
    choose_stay_for_event,
    iter_table,
    spark_config,
    spark_enabled,
)
from src.data.stage_filtered_tables import require_stage_cache
from src.features.lab_processor import LabProcessor
from src.features.medication_history import (
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
    write_parquet_pylist,
)


def _trajectory_root(config: dict) -> Path:
    processed_root = resolve_path(config["_project_root"], config["paths"]["processed_root"])
    return ensure_dir(Path(processed_root) / "trajectories")


def _legacy_trajectory_file(config: dict, split: str) -> Path:
    return _trajectory_root(config) / split / "trajectories.jsonl.gz"


def _cohort_keys_path(config: dict) -> Path:
    interim_root = resolve_path(config["_project_root"], config["paths"]["interim_root"])
    return Path(interim_root) / "cohort" / "cohort_keys.parquet"


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


def _build_trajectories_python(config: dict) -> dict[str, Path]:
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
        history: list[int] = []
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
                    "med_history_ids": history[:max_med_history],
                    "delta_hours": 0.0 if step_index == 0 else float(bucket_hours),
                    "target_drugs": bucket_drugs[step_index],
                }
            )
            for drug_id in reversed(bucket_drugs[step_index]):
                if drug_id in history:
                    history.remove(drug_id)
                history.insert(0, drug_id)
            history = history[:max_med_history]

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


def _feature_mapping_rows(vocab: dict[str, object], prefix: str) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for token, index in vocab["token_to_idx"].items():
        if token in {"PAD", "UNK"}:
            continue
        rows.append(
            {
                "token": token,
                "itemid": token.split(":", 1)[1] if ":" in token else token.removeprefix(prefix),
                "index": int(index),
                "feature_index": int(index) - 2,
            }
        )
    return rows


def _create_mapping_df(spark, rows: list[dict[str, object]], schema: str):
    if rows:
        return spark.createDataFrame(rows)
    return spark.createDataFrame([], schema=schema)


def _step_index_expr(F, event_col, start_col, end_col, num_steps_col, bucket_hours: int):
    bucket_seconds = float(bucket_hours) * 3600.0
    delta = (F.col(event_col).cast("long") - F.col(start_col).cast("long")) / F.lit(bucket_seconds)
    return (
        F.when(F.col(event_col).isNull(), F.lit(0))
        .when(F.col(event_col) <= F.col(start_col), F.lit(0))
        .when(F.col(event_col) >= F.col(end_col), F.col(num_steps_col) - F.lit(1))
        .otherwise(F.least(F.floor(delta).cast("int"), F.col(num_steps_col) - F.lit(1)))
    ).cast("int")


def _collect_numeric_stats(dataframe, feature_size: int) -> list[dict[str, float]]:
    stats = [{"mean": 0.0, "std": 1.0, "count": 0} for _ in range(feature_size)]
    for row in dataframe.collect():
        feature_index = int(row["feature_index"])
        if not 0 <= feature_index < feature_size:
            continue
        std = float(row["std"]) if row["std"] is not None and float(row["std"]) > 0.0 else 1.0
        stats[feature_index] = {
            "mean": float(row["mean"] or 0.0),
            "std": std,
            "count": int(row["count"] or 0),
        }
    return stats


def _sparse_pairs_to_dense(
    pairs: list[dict[str, object]] | None,
    feature_size: int,
    stats: list[dict[str, float]],
    eps: float,
) -> tuple[list[float], list[int]]:
    dense = [0.0] * feature_size
    mask = [0] * feature_size
    for pair in pairs or []:
        if pair is None:
            continue
        feature_index = int(pair["feature_index"])
        value = float(pair["value"])
        if not 0 <= feature_index < feature_size:
            continue
        stat = stats[feature_index]
        dense[feature_index] = (value - float(stat["mean"])) / max(float(stat["std"]), eps)
        mask[feature_index] = 1
    return dense, mask


def _write_shard(split_dir: Path, shard_index: int, rows: list[dict[str, object]]) -> Path:
    shard_path = split_dir / f"part-{shard_index:05d}.parquet"
    return write_parquet_pylist(shard_path, rows)


def _flush_record(
    split: str,
    buffer: list[dict[str, object]],
    shard_counts: dict[str, int],
    manifests: dict[str, list[dict[str, object]]],
    trajectory_root: Path,
) -> None:
    if not buffer:
        return
    split_dir = ensure_dir(trajectory_root / split)
    shard_index = shard_counts[split]
    shard_path = _write_shard(split_dir, shard_index, buffer)
    manifests[split].append(
        {
            "path": str(shard_path.relative_to(trajectory_root)),
            "rows": len(buffer),
        }
    )
    shard_counts[split] += 1
    buffer.clear()


def _finalize_split_from_step_rows(
    split: str,
    source_dir: Path,
    trajectory_root: Path,
    *,
    bucket_hours: int,
    max_med_history: int,
    trajectory_rows_per_file: int,
    lab_feature_size: int,
    vital_feature_size: int,
    drug_vocab_size: int,
    lab_stats: list[dict[str, float]],
    vital_stats: list[dict[str, float]],
    normalization_eps: float,
    manifests: dict[str, list[dict[str, object]]],
    record_counts: dict[str, int],
) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for parquet trajectory finalization. Install requirements.txt first."
        ) from exc

    split_dir = trajectory_root / split
    shutil.rmtree(split_dir, ignore_errors=True)
    ensure_dir(split_dir)
    shard_counts = {split: 0}
    trajectory_buffer: list[dict[str, object]] = []

    parquet_files = sorted(item for item in source_dir.rglob("*.parquet") if item.is_file()) if source_dir.exists() else []
    for parquet_file in parquet_files:
        current_record: dict[str, object] | None = None
        history: list[int] = []
        parquet_reader = pq.ParquetFile(parquet_file)
        for batch in parquet_reader.iter_batches(batch_size=256, use_threads=False):
            for row in batch.to_pylist():
                stay_id = int(row["stay_id"])
                if current_record is None or int(current_record["stay_id"]) != stay_id:
                    if current_record is not None:
                        trajectory_buffer.append(current_record)
                        record_counts[split] += 1
                        if len(trajectory_buffer) >= trajectory_rows_per_file:
                            _flush_record(split, trajectory_buffer, shard_counts, manifests, trajectory_root)
                    history = []
                    current_record = {
                        "subject_id": int(row["subject_id"]),
                        "hadm_id": int(row["hadm_id"]),
                        "stay_id": stay_id,
                        "split": split,
                        "intime": str(row["intime"]),
                        "outtime": str(row["outtime"]),
                        "num_steps": int(row["num_steps"]),
                        "drug_vocab_size": drug_vocab_size,
                        "lab_feature_size": lab_feature_size,
                        "vital_feature_size": vital_feature_size,
                        "steps": [],
                    }

                target_drugs = dedupe_preserve_order(int(value) for value in (row.get("target_drugs") or []) if value is not None)
                lab_values, lab_mask = _sparse_pairs_to_dense(
                    row.get("lab_pairs"),
                    lab_feature_size,
                    lab_stats,
                    normalization_eps,
                )
                vital_values, vital_mask = _sparse_pairs_to_dense(
                    row.get("vital_pairs"),
                    vital_feature_size,
                    vital_stats,
                    normalization_eps,
                )
                current_record["steps"].append(
                    {
                        "step_index": int(row["step_index"]),
                        "diagnosis_ids": [int(value) for value in (row.get("diagnosis_ids") or [])],
                        "procedure_ids": [int(value) for value in (row.get("procedure_ids") or [])],
                        "lab_values": lab_values,
                        "lab_mask": lab_mask,
                        "vital_values": vital_values,
                        "vital_mask": vital_mask,
                        "med_history_ids": history[:max_med_history],
                        "delta_hours": 0.0 if int(row["step_index"]) == 0 else float(bucket_hours),
                        "target_drugs": target_drugs,
                    }
                )
                for drug_id in reversed(target_drugs):
                    if drug_id in history:
                        history.remove(drug_id)
                    history.insert(0, drug_id)
                history = history[:max_med_history]
        if current_record is not None:
            trajectory_buffer.append(current_record)
            record_counts[split] += 1
            if len(trajectory_buffer) >= trajectory_rows_per_file:
                _flush_record(split, trajectory_buffer, shard_counts, manifests, trajectory_root)

    _flush_record(split, trajectory_buffer, shard_counts, manifests, trajectory_root)


def _build_trajectories_spark(config: dict) -> dict[str, Path]:
    cache_dir, _ = require_stage_cache(config)
    cohort_keys_path = _cohort_keys_path(config)
    if not cohort_keys_path.exists():
        raise FileNotFoundError(
            f"Cohort keys parquet is missing at {cohort_keys_path}. "
            f"Run `python -m src.data.build_cohort --config {config['_config_path']}` first."
        )

    vocab_bundle = load_vocab_bundle(config)
    feature_cfg = config.get("features", {})
    bucket_hours = int(feature_cfg.get("time_bucket_hours", 24))
    max_med_history = int(feature_cfg.get("max_med_history", 32))
    normalization_eps = float(feature_cfg.get("normalization_eps", 1e-6))
    spark_cfg = spark_config(config)
    trajectory_rows_per_file = int(spark_cfg.get("trajectory_rows_per_file", 2048))
    trajectory_root = _trajectory_root(config)
    step_rows_root = trajectory_root / "_step_rows"

    shutil.rmtree(step_rows_root, ignore_errors=True)
    for split_name in ("train", "val", "test"):
        shutil.rmtree(trajectory_root / split_name, ignore_errors=True)
    manifest_path = trajectory_root / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    spark = build_spark_session(config, app_name="build-trajectories")
    try:
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        cohort = (
            spark.read.parquet(str(cohort_keys_path))
            .select("subject_id", "hadm_id", "stay_id", "split", "intime", "outtime")
            .withColumn("subject_id", F.col("subject_id").cast("long"))
            .withColumn("hadm_id", F.col("hadm_id").cast("long"))
            .withColumn("stay_id", F.col("stay_id").cast("long"))
            .withColumn("intime_ts", F.to_timestamp("intime"))
            .withColumn("outtime_ts", F.to_timestamp("outtime"))
            .withColumn(
                "num_steps",
                F.greatest(
                    F.lit(1),
                    F.ceil(
                        (F.col("outtime_ts").cast("long") - F.col("intime_ts").cast("long"))
                        / F.lit(float(bucket_hours) * 3600.0)
                    ).cast("int"),
                ),
            )
        )

        diag_map = F.broadcast(
            _create_mapping_df(
                spark,
                [
                    {"diagnosis_token": token, "diagnosis_id": int(index)}
                    for token, index in vocab_bundle["diagnosis"]["token_to_idx"].items()
                    if token not in {"PAD", "UNK"}
                ],
                "diagnosis_token string, diagnosis_id int",
            )
        )
        proc_map = F.broadcast(
            _create_mapping_df(
                spark,
                [
                    {"procedure_token": token, "procedure_id": int(index)}
                    for token, index in vocab_bundle["procedure"]["token_to_idx"].items()
                    if token not in {"PAD", "UNK"}
                ],
                "procedure_token string, procedure_id int",
            )
        )
        drug_map = F.broadcast(
            _create_mapping_df(
                spark,
                [
                    {"drug_token": token, "drug_id": int(index)}
                    for token, index in vocab_bundle["drug"]["token_to_idx"].items()
                    if token not in {"PAD", "UNK"}
                ],
                "drug_token string, drug_id int",
            )
        )
        lab_map = F.broadcast(
            _create_mapping_df(
                spark,
                [
                    {"itemid": row["itemid"], "feature_index": int(row["feature_index"])}
                    for row in _feature_mapping_rows(vocab_bundle["lab"], "LAB:")
                ],
                "itemid string, feature_index int",
            )
        )
        vital_map = F.broadcast(
            _create_mapping_df(
                spark,
                [
                    {"itemid": row["itemid"], "feature_index": int(row["feature_index"])}
                    for row in _feature_mapping_rows(vocab_bundle["vital"], "VITAL:")
                ],
                "itemid string, feature_index int",
            )
        )

        diagnosis_rows = spark.read.parquet(str(cache_dir / "diagnoses_icd")).select("hadm_id", "diagnosis_token")
        diagnosis_by_stay = (
            diagnosis_rows.join(diag_map, "diagnosis_token", "inner")
            .join(F.broadcast(cohort.select("hadm_id", "stay_id")), "hadm_id", "inner")
            .groupBy("stay_id")
            .agg(F.sort_array(F.collect_set("diagnosis_id")).alias("diagnosis_ids"))
        )

        procedure_rows = (
            spark.read.parquet(str(cache_dir / "procedures_icd"))
            .select("hadm_id", "event_time", "procedure_token")
            .join(proc_map, "procedure_token", "inner")
            .join(
                F.broadcast(
                    cohort.select("hadm_id", "stay_id", "intime_ts", "outtime_ts", "num_steps")
                ),
                "hadm_id",
                "inner",
            )
            .withColumn(
                "step_index",
                _step_index_expr(F, "event_time", "intime_ts", "outtime_ts", "num_steps", bucket_hours),
            )
            .groupBy("stay_id", "step_index")
            .agg(F.sort_array(F.collect_set("procedure_id")).alias("procedure_ids"))
        )

        lab_events = (
            spark.read.parquet(str(cache_dir / "labevents"))
            .select("hadm_id", F.trim(F.col("itemid")).alias("itemid"), "valuenum", "event_time")
            .join(lab_map, "itemid", "inner")
            .join(
                F.broadcast(
                    cohort.select("hadm_id", "stay_id", "split", "intime_ts", "outtime_ts", "num_steps")
                ),
                "hadm_id",
                "inner",
            )
            .withColumn(
                "step_index",
                _step_index_expr(F, "event_time", "intime_ts", "outtime_ts", "num_steps", bucket_hours),
            )
        )
        lab_stats = _collect_numeric_stats(
            lab_events.filter(F.col("split") == "train")
            .groupBy("feature_index")
            .agg(
                F.count("*").alias("count"),
                F.avg("valuenum").alias("mean"),
                F.stddev_pop("valuenum").alias("std"),
            ),
            max(len(vocab_bundle["lab"]["idx_to_token"]) - 2, 0),
        )
        lab_window = Window.partitionBy("stay_id", "step_index", "feature_index").orderBy(F.col("event_time").desc_nulls_last())
        lab_pairs = (
            lab_events.withColumn("row_num", F.row_number().over(lab_window))
            .filter(F.col("row_num") == 1)
            .groupBy("stay_id", "step_index")
            .agg(
                F.sort_array(
                    F.collect_list(F.struct(F.col("feature_index"), F.col("valuenum").alias("value")))
                ).alias("lab_pairs")
            )
        )

        vital_events = (
            spark.read.parquet(str(cache_dir / "chartevents"))
            .select("stay_id", F.trim(F.col("itemid")).alias("itemid"), "valuenum", "event_time")
            .join(vital_map, "itemid", "inner")
            .join(
                F.broadcast(
                    cohort.select("stay_id", "split", "intime_ts", "outtime_ts", "num_steps")
                ),
                "stay_id",
                "inner",
            )
            .withColumn(
                "step_index",
                _step_index_expr(F, "event_time", "intime_ts", "outtime_ts", "num_steps", bucket_hours),
            )
        )
        vital_stats = _collect_numeric_stats(
            vital_events.filter(F.col("split") == "train")
            .groupBy("feature_index")
            .agg(
                F.count("*").alias("count"),
                F.avg("valuenum").alias("mean"),
                F.stddev_pop("valuenum").alias("std"),
            ),
            max(len(vocab_bundle["vital"]["idx_to_token"]) - 2, 0),
        )
        vital_window = Window.partitionBy("stay_id", "step_index", "feature_index").orderBy(F.col("event_time").desc_nulls_last())
        vital_pairs = (
            vital_events.withColumn("row_num", F.row_number().over(vital_window))
            .filter(F.col("row_num") == 1)
            .groupBy("stay_id", "step_index")
            .agg(
                F.sort_array(
                    F.collect_list(F.struct(F.col("feature_index"), F.col("valuenum").alias("value")))
                ).alias("vital_pairs")
            )
        )

        medication_rows = (
            spark.read.parquet(str(cache_dir / "medications"))
            .select("hadm_id", "event_time", "drug_token")
            .join(drug_map, "drug_token", "inner")
            .join(
                F.broadcast(
                    cohort.select("hadm_id", "stay_id", "intime_ts", "outtime_ts", "num_steps")
                ),
                "hadm_id",
                "inner",
            )
            .withColumn(
                "step_index",
                _step_index_expr(F, "event_time", "intime_ts", "outtime_ts", "num_steps", bucket_hours),
            )
            .groupBy("stay_id", "step_index")
            .agg(
                F.expr(
                    "transform(sort_array(collect_list(named_struct('event_time', event_time, 'drug_id', drug_id))), x -> x.drug_id)"
                ).alias("target_drugs")
            )
        )

        step_rows = (
            cohort.select(
                "split",
                "subject_id",
                "hadm_id",
                "stay_id",
                "intime",
                "outtime",
                "num_steps",
                F.explode(F.sequence(F.lit(0), F.col("num_steps") - F.lit(1))).alias("step_index"),
            )
            .join(diagnosis_by_stay, "stay_id", "left")
            .join(procedure_rows, ["stay_id", "step_index"], "left")
            .join(lab_pairs, ["stay_id", "step_index"], "left")
            .join(vital_pairs, ["stay_id", "step_index"], "left")
            .join(medication_rows, ["stay_id", "step_index"], "left")
            .select(
                "split",
                "subject_id",
                "hadm_id",
                "stay_id",
                "intime",
                "outtime",
                "num_steps",
                "step_index",
                "diagnosis_ids",
                "procedure_ids",
                "lab_pairs",
                "vital_pairs",
                "target_drugs",
            )
        )

        for split_name, partitions in (("train", 4), ("val", 2), ("test", 2)):
            split_df = step_rows.filter(F.col("split") == split_name).drop("split")
            split_output = step_rows_root / split_name
            if split_df.rdd.isEmpty():
                ensure_dir(split_output)
                continue
            (
                split_df.repartition(partitions, "stay_id")
                .sortWithinPartitions("stay_id", "step_index")
                .write.mode("overwrite")
                .option("compression", "snappy")
                .parquet(str(split_output))
            )

        lab_feature_size = max(len(vocab_bundle["lab"]["idx_to_token"]) - 2, 0)
        vital_feature_size = max(len(vocab_bundle["vital"]["idx_to_token"]) - 2, 0)
        drug_vocab_size = len(vocab_bundle["drug"]["idx_to_token"])
        manifests: dict[str, list[dict[str, object]]] = {split: [] for split in ("train", "val", "test")}
        record_counts: dict[str, int] = {split: 0 for split in ("train", "val", "test")}
        output_paths: dict[str, Path] = {}
        for split_name in ("train", "val", "test"):
            output_paths[split_name] = trajectory_root / split_name
            _finalize_split_from_step_rows(
                split_name,
                step_rows_root / split_name,
                trajectory_root,
                bucket_hours=bucket_hours,
                max_med_history=max_med_history,
                trajectory_rows_per_file=trajectory_rows_per_file,
                lab_feature_size=lab_feature_size,
                vital_feature_size=vital_feature_size,
                drug_vocab_size=drug_vocab_size,
                lab_stats=lab_stats,
                vital_stats=vital_stats,
                normalization_eps=normalization_eps,
                manifests=manifests,
                record_counts=record_counts,
            )

        write_json(
            trajectory_root / "normalization_stats.json",
            {
                "lab": lab_stats,
                "vital": vital_stats,
                "eps": normalization_eps,
            },
        )
        write_json(
            trajectory_root / "metadata.json",
            {
                "bucket_hours": bucket_hours,
                "max_med_history": max_med_history,
                "drug_vocab_size": drug_vocab_size,
                "lab_feature_size": lab_feature_size,
                "vital_feature_size": vital_feature_size,
                "counts_by_split": record_counts,
                "processed_format": "parquet",
            },
        )
        write_json(
            manifest_path,
            {
                "format": "parquet",
                "schema_version": 1,
                "trajectory_rows_per_file": trajectory_rows_per_file,
                "counts_by_split": record_counts,
                "splits": {
                    split_name: {
                        "rows": record_counts[split_name],
                        "shards": manifests[split_name],
                    }
                    for split_name in ("train", "val", "test")
                },
            },
        )
        return output_paths
    finally:
        spark.stop()


def build_trajectories(config_path: str | Path) -> dict[str, Path]:
    config = load_yaml_config(config_path)
    if spark_enabled(config):
        return _build_trajectories_spark(config)
    return _build_trajectories_python(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bucketed ICU stay trajectories.")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    args = parser.parse_args()
    build_trajectories(args.config)


if __name__ == "__main__":
    main()
