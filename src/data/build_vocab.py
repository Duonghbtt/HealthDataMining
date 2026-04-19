from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

from src.data.load_mimic import (
    MIMICDataPaths,
    build_spark_session,
    iter_table,
    read_lookup,
    spark_enabled,
)
from src.data.stage_filtered_tables import require_stage_cache
from src.features.medication_history import extract_medication_token
from src.utils.io import (
    ensure_dir,
    load_yaml_config,
    parse_float,
    parse_int,
    read_csv_gz,
    read_json,
    resolve_path,
    write_json,
)


def vocab_dir_from_config(config: dict) -> Path:
    interim_root = resolve_path(config["_project_root"], config["paths"]["interim_root"])
    return ensure_dir(Path(interim_root) / "vocab")


def cohort_path_from_config(config: dict) -> Path:
    interim_root = resolve_path(config["_project_root"], config["paths"]["interim_root"])
    return Path(interim_root) / "cohort" / "cohort.csv.gz"


def _sorted_counter_items(counter: Counter[str], top_k: int | None = None) -> list[str]:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    tokens = [token for token, _ in items]
    if top_k is not None and top_k > 0:
        return tokens[:top_k]
    return tokens


def _build_vocab_payload(tokens: Iterable[str], vocab_name: str) -> dict[str, object]:
    idx_to_token = ["PAD", "UNK", *list(tokens)]
    token_to_idx = {token: index for index, token in enumerate(idx_to_token)}
    return {
        "name": vocab_name,
        "size": len(idx_to_token),
        "pad_idx": 0,
        "unk_idx": 1,
        "idx_to_token": idx_to_token,
        "token_to_idx": token_to_idx,
    }


def load_vocab_bundle(config_path: str | Path | dict) -> dict[str, dict]:
    config = config_path if isinstance(config_path, dict) else load_yaml_config(config_path)
    vocab_dir = vocab_dir_from_config(config)
    bundle: dict[str, dict] = {}
    for name in ("diagnosis", "procedure", "drug", "lab", "vital"):
        bundle[name] = read_json(vocab_dir / f"{name}_vocab.json")
    return bundle


def _write_vocab_outputs(
    config: dict,
    *,
    diagnosis_tokens: list[str],
    procedure_tokens: list[str],
    drug_tokens: list[str],
    lab_tokens: list[str],
    vital_tokens: list[str],
    built_from_split: str,
) -> Path:
    vocab_dir = vocab_dir_from_config(config)
    paths = MIMICDataPaths.from_config(config)

    diagnosis_vocab = _build_vocab_payload(diagnosis_tokens, "diagnosis")
    procedure_vocab = _build_vocab_payload(procedure_tokens, "procedure")
    drug_vocab = _build_vocab_payload(drug_tokens, "drug")
    lab_vocab = _build_vocab_payload(lab_tokens, "lab")
    vital_vocab = _build_vocab_payload(vital_tokens, "vital")

    lab_lookup = read_lookup(paths, "d_labitems", "itemid", ["label", "category", "fluid"])
    vital_lookup = read_lookup(paths, "d_items", "itemid", ["label", "category", "unitname"])

    lab_metadata = {
        token: {
            "index": index,
            "label": lab_lookup.get(token.split(":", 1)[1], {}).get("label", ""),
            "category": lab_lookup.get(token.split(":", 1)[1], {}).get("category", ""),
            "fluid": lab_lookup.get(token.split(":", 1)[1], {}).get("fluid", ""),
        }
        for token, index in lab_vocab["token_to_idx"].items()
        if token not in {"PAD", "UNK"}
    }
    vital_metadata = {
        token: {
            "index": index,
            "label": vital_lookup.get(token.split(":", 1)[1], {}).get("label", ""),
            "category": vital_lookup.get(token.split(":", 1)[1], {}).get("category", ""),
            "unitname": vital_lookup.get(token.split(":", 1)[1], {}).get("unitname", ""),
        }
        for token, index in vital_vocab["token_to_idx"].items()
        if token not in {"PAD", "UNK"}
    }

    write_json(vocab_dir / "diagnosis_vocab.json", diagnosis_vocab)
    write_json(vocab_dir / "procedure_vocab.json", procedure_vocab)
    write_json(vocab_dir / "drug_vocab.json", drug_vocab)
    write_json(vocab_dir / "lab_vocab.json", lab_vocab)
    write_json(vocab_dir / "vital_vocab.json", vital_vocab)
    write_json(vocab_dir / "lab_metadata.json", lab_metadata)
    write_json(vocab_dir / "vital_metadata.json", vital_metadata)
    write_json(
        vocab_dir / "vocab_summary.json",
        {
            "diagnosis_size": diagnosis_vocab["size"],
            "procedure_size": procedure_vocab["size"],
            "drug_size": drug_vocab["size"],
            "lab_size": lab_vocab["size"],
            "vital_size": vital_vocab["size"],
            "built_from_split": built_from_split,
        },
    )
    return vocab_dir


def _train_cohort_ids(config: dict) -> tuple[set[int], set[int]]:
    cohort_rows = read_csv_gz(cohort_path_from_config(config))
    train_rows = [row for row in cohort_rows if str(row.get("split", "")).strip().lower() == "train"]
    if not train_rows:
        raise ValueError(
            "Vocab building requires at least one train cohort row. "
            f"Check split assignments in {cohort_path_from_config(config)}."
        )

    hadm_ids = {parse_int(row["hadm_id"]) for row in train_rows if row.get("hadm_id")}
    stay_ids = {parse_int(row["stay_id"]) for row in train_rows if row.get("stay_id")}
    hadm_ids.discard(None)
    stay_ids.discard(None)
    if not hadm_ids and not stay_ids:
        raise ValueError("Train cohort rows did not contain any usable hadm_id or stay_id keys for vocab building.")
    return hadm_ids, stay_ids


def _build_vocab_python(config: dict) -> Path:
    paths = MIMICDataPaths.from_config(config)
    hadm_ids, stay_ids = _train_cohort_ids(config)

    feature_cfg = config.get("features", {})
    top_k_labs = int(feature_cfg.get("top_k_labs", 64))
    top_k_vitals = int(feature_cfg.get("top_k_vitals", 64))

    diagnosis_counter: Counter[str] = Counter()
    procedure_counter: Counter[str] = Counter()
    drug_counter: Counter[str] = Counter()
    lab_counter: Counter[str] = Counter()
    vital_counter: Counter[str] = Counter()

    for row in iter_table(paths, "diagnoses_icd", fields=["hadm_id", "icd_code", "icd_version"]):
        hadm_id = parse_int(row.get("hadm_id"))
        code = str(row.get("icd_code", "")).strip()
        version = str(row.get("icd_version", "")).strip()
        if hadm_id in hadm_ids and code and version:
            diagnosis_counter[f"ICD{version}:{code}"] += 1

    for row in iter_table(paths, "procedures_icd", fields=["hadm_id", "icd_code", "icd_version"]):
        hadm_id = parse_int(row.get("hadm_id"))
        code = str(row.get("icd_code", "")).strip()
        version = str(row.get("icd_version", "")).strip()
        if hadm_id in hadm_ids and code and version:
            procedure_counter[f"PROC{version}:{code}"] += 1

    for table_name, fields in (
        ("emar", ["hadm_id", "medication"]),
        ("prescriptions", ["hadm_id", "drug", "formulary_drug_cd"]),
        ("pharmacy", ["hadm_id", "medication"]),
    ):
        for row in iter_table(paths, table_name, fields=fields):
            hadm_id = parse_int(row.get("hadm_id"))
            if hadm_id not in hadm_ids:
                continue
            token = extract_medication_token(row)
            if token:
                drug_counter[token] += 1

    for row in iter_table(paths, "labevents", fields=["hadm_id", "itemid", "valuenum"]):
        hadm_id = parse_int(row.get("hadm_id"))
        itemid = str(row.get("itemid", "")).strip()
        value = parse_float(row.get("valuenum"))
        if hadm_id in hadm_ids and itemid and value is not None:
            lab_counter[f"LAB:{itemid}"] += 1

    for row in iter_table(paths, "chartevents", fields=["stay_id", "itemid", "valuenum"]):
        stay_id = parse_int(row.get("stay_id"))
        itemid = str(row.get("itemid", "")).strip()
        value = parse_float(row.get("valuenum"))
        if stay_id in stay_ids and itemid and value is not None:
            vital_counter[f"VITAL:{itemid}"] += 1

    return _write_vocab_outputs(
        config,
        diagnosis_tokens=_sorted_counter_items(diagnosis_counter),
        procedure_tokens=_sorted_counter_items(procedure_counter),
        drug_tokens=_sorted_counter_items(drug_counter),
        lab_tokens=_sorted_counter_items(lab_counter, top_k=top_k_labs),
        vital_tokens=_sorted_counter_items(vital_counter, top_k=top_k_vitals),
        built_from_split="train",
    )


def _collect_tokens(dataframe, token_column: str, *, top_k: int | None = None) -> list[str]:
    from pyspark.sql import functions as F

    ranked = (
        dataframe.groupBy(token_column)
        .count()
        .orderBy(F.desc("count"), F.asc(token_column))
        .select(token_column)
    )
    if top_k is not None and top_k > 0:
        ranked = ranked.limit(int(top_k))
    return [str(row[token_column]) for row in ranked.collect() if row[token_column]]


def _build_vocab_spark(config: dict) -> Path:
    cache_dir, _ = require_stage_cache(config)
    feature_cfg = config.get("features", {})
    top_k_labs = int(feature_cfg.get("top_k_labs", 64))
    top_k_vitals = int(feature_cfg.get("top_k_vitals", 64))
    spark = build_spark_session(config, app_name="build-vocab")
    try:
        from pyspark.sql import functions as F

        interim_root = resolve_path(config["_project_root"], config["paths"]["interim_root"])
        train_keys = (
            spark.read.parquet(str(Path(interim_root) / "cohort" / "cohort_keys.parquet"))
            .filter(F.lower(F.trim(F.col("split"))) == F.lit("train"))
            .select("hadm_id", "stay_id")
        )
        if int(train_keys.limit(1).count()) <= 0:
            raise ValueError("Vocab building requires at least one train key in cohort_keys.parquet.")

        train_hadm = F.broadcast(train_keys.select("hadm_id").dropna().dropDuplicates())
        train_stay = F.broadcast(train_keys.select("stay_id").dropna().dropDuplicates())

        diagnosis_df = (
            spark.read.parquet(str(cache_dir / "diagnoses_icd"))
            .join(train_hadm, "hadm_id", "inner")
            .select("diagnosis_token")
        )
        procedure_df = (
            spark.read.parquet(str(cache_dir / "procedures_icd"))
            .join(train_hadm, "hadm_id", "inner")
            .select("procedure_token")
        )
        drug_df = (
            spark.read.parquet(str(cache_dir / "medications"))
            .join(train_hadm, "hadm_id", "inner")
            .select("drug_token")
        )
        lab_df = (
            spark.read.parquet(str(cache_dir / "labevents"))
            .join(train_hadm, "hadm_id", "inner")
            .select(F.trim(F.col("itemid")).alias("itemid"))
        )
        vital_df = (
            spark.read.parquet(str(cache_dir / "chartevents"))
            .join(train_stay, "stay_id", "inner")
            .select(F.trim(F.col("itemid")).alias("itemid"))
        )

        diagnosis_tokens = _collect_tokens(diagnosis_df, "diagnosis_token")
        procedure_tokens = _collect_tokens(procedure_df, "procedure_token")
        drug_tokens = _collect_tokens(drug_df, "drug_token")
        lab_tokens = [f"LAB:{itemid}" for itemid in _collect_tokens(lab_df, "itemid", top_k=top_k_labs)]
        vital_tokens = [f"VITAL:{itemid}" for itemid in _collect_tokens(vital_df, "itemid", top_k=top_k_vitals)]
        return _write_vocab_outputs(
            config,
            diagnosis_tokens=diagnosis_tokens,
            procedure_tokens=procedure_tokens,
            drug_tokens=drug_tokens,
            lab_tokens=lab_tokens,
            vital_tokens=vital_tokens,
            built_from_split="train",
        )
    finally:
        spark.stop()


def build_vocab(config_path: str | Path) -> Path:
    config = load_yaml_config(config_path)
    if spark_enabled(config):
        return _build_vocab_spark(config)
    return _build_vocab_python(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vocabularies from cohort-filtered MIMIC tables.")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    args = parser.parse_args()
    build_vocab(args.config)


if __name__ == "__main__":
    main()
