from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.load_mimic import (
    MIMICDataPaths,
    build_spark_session,
    iter_csv_rows,
    iter_table,
    read_lookup,
    read_table_spark,
    spark_enabled,
)
from src.data.medication_vocab_utils import (
    MedicationNormalizationLookup,
    PrescriptionNormalizationEvidence,
    build_medication_normalization_lookup,
    build_prescription_token_normalization_map,
    is_plausible_raw_to_ingredient_match,
)
from src.data.rxnorm_utils import is_suspicious_ingredient_name, normalize_ndc
from src.features.medication_history import extract_medication_token
from src.utils.io import (
    ensure_dir,
    load_yaml_config,
    parse_float,
    parse_int,
    read_json,
    resolve_path,
    write_json,
)


LOGGER = logging.getLogger(__name__)

DRUG_MIN_FREQ = 10
MED_VOCAB_MAIN_WARN_RANGE = (100, 400)
MEDICATION_SOURCE_SPECS = (("prescriptions", ["hadm_id", "drug", "formulary_drug_cd"]),)


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")


def vocab_dir_from_config(config: Mapping[str, object]) -> Path:
    paths_cfg = dict(config.get("paths", {}))
    if paths_cfg.get("vocab_root"):
        return ensure_dir(resolve_path(config["_project_root"], paths_cfg["vocab_root"]))
    interim_root = resolve_path(config["_project_root"], paths_cfg["interim_root"])
    return ensure_dir(Path(interim_root) / "vocab")


def cohort_path_from_config(config: Mapping[str, object]) -> Path:
    cohort_root = dict(config.get("paths", {})).get("cohort_root")
    if cohort_root:
        return Path(resolve_path(config["_project_root"], cohort_root)) / "cohort.csv.gz"
    interim_root = resolve_path(config["_project_root"], dict(config.get("paths", {}))["interim_root"])
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


def _log_phase_timing(phase_name: str, phase_start: float, **stats: Any) -> None:
    elapsed = perf_counter() - phase_start
    stat_text = ", ".join(f"{key}={value}" for key, value in stats.items())
    if stat_text:
        LOGGER.info("Phase `%s` finished in %.2fs (%s)", phase_name, elapsed, stat_text)
    else:
        LOGGER.info("Phase `%s` finished in %.2fs", phase_name, elapsed)


def _read_first_existing_json(vocab_dir: Path, candidate_names: tuple[str, ...]) -> dict:
    for filename in candidate_names:
        path = vocab_dir / filename
        if path.exists():
            return read_json(path)
    raise FileNotFoundError(
        f"Missing expected vocab artifact under {vocab_dir}. Checked: {list(candidate_names)}"
    )


def load_vocab_bundle(config_path: str | Path | Mapping[str, object]) -> dict[str, dict]:
    """Load canonical benchmark vocabularies.

    `med_vocab_main.json` is the only medication representation for the
    benchmark. The `drug` key is kept as a compatibility alias for older model
    and training code that still expects that bundle name.
    """

    config = config_path if isinstance(config_path, Mapping) else load_yaml_config(config_path)
    vocab_dir = vocab_dir_from_config(config)
    med_main_vocab = _read_first_existing_json(vocab_dir, ("med_vocab_main.json",))
    bundle: dict[str, dict] = {
        "diagnosis": _read_first_existing_json(vocab_dir, ("diag_vocab.json", "diagnosis_vocab.json")),
        "procedure": _read_first_existing_json(vocab_dir, ("proc_vocab.json", "procedure_vocab.json")),
        "drug": med_main_vocab,
        "med_main": med_main_vocab,
        "lab": read_json(vocab_dir / "lab_vocab.json"),
        "vital": read_json(vocab_dir / "vital_vocab.json"),
    }
    bundle["diag"] = bundle["diagnosis"]
    bundle["proc"] = bundle["procedure"]
    return bundle


def _resolve_required_source_path(
    config: Mapping[str, object],
    *,
    key: str,
    description: str,
    defaults: tuple[str, ...],
) -> Path:
    project_root = Path(config["_project_root"])
    paths_cfg = dict(config.get("paths", {}))
    raw_value = paths_cfg.get(key) or config.get(key)

    candidates: list[Path] = []
    if raw_value:
        candidates.append(Path(resolve_path(project_root, raw_value)))
    for default in defaults:
        candidates.append(Path(resolve_path(project_root, default)))

    seen: set[str] = set()
    for candidate in candidates:
        marker = str(candidate.resolve(strict=False))
        if marker in seen:
            continue
        seen.add(marker)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Missing required {description}. Checked: {[str(path) for path in candidates]}. "
        f"Place the file there or update `{key}` in configs/data.yaml."
    )


def _resolve_rxnorm_root(config: Mapping[str, object]) -> Path:
    return _resolve_required_source_path(
        config,
        key="rxnorm_root",
        description="RxNorm release directory",
        defaults=(
            "data/processed/ddi/RxNorm_full_04062026",
            "data/processed/ddi/RxNorm_full_prescribe_04062026",
            "data/raw/ddi/RxNorm_full_04062026",
        ),
    )


def _spark_stage_cache_root(config: Mapping[str, object]) -> Path:
    spark_cfg = dict(config.get("spark", {}))
    stage_cache_dir = spark_cfg.get("stage_cache_dir", "data/interim/spark_cache")
    return Path(resolve_path(config["_project_root"], stage_cache_dir))


def _stage_cache_dir(config: Mapping[str, object], table_name: str) -> Path:
    return _spark_stage_cache_root(config) / table_name


def _has_stage_cache(cache_dir: Path) -> bool:
    return cache_dir.exists() and any(cache_dir.glob("part-*"))


def _ensure_stage_cache(
    spark,
    config: Mapping[str, object],
    *,
    paths: MIMICDataPaths,
    table_name: str,
    columns: tuple[str, str, str],
) -> Path:
    cache_dir = _stage_cache_dir(config, table_name)
    if _has_stage_cache(cache_dir):
        return cache_dir

    LOGGER.info("Stage cache for `%s` missing; creating parquet cache at %s", table_name, cache_dir)
    ensure_dir(cache_dir.parent)
    dataframe = read_table_spark(spark, paths, table_name, columns=columns)
    dataframe.write.mode("overwrite").parquet(str(cache_dir))
    return cache_dir


def _load_cohort_ids_and_splits(cohort_path: Path) -> tuple[set[int], set[int], dict[int, str]]:
    hadm_ids: set[int] = set()
    stay_ids: set[int] = set()
    hadm_splits: dict[int, str] = {}

    for row in iter_csv_rows(cohort_path, fields=("hadm_id", "stay_id", "split")):
        hadm_id = parse_int(row.get("hadm_id"))
        stay_id = parse_int(row.get("stay_id"))
        split = str(row.get("split", "")).strip()

        if hadm_id is not None:
            resolved_hadm_id = int(hadm_id)
            hadm_ids.add(resolved_hadm_id)
            if split:
                hadm_splits[resolved_hadm_id] = split
        if stay_id is not None:
            stay_ids.add(int(stay_id))

    return hadm_ids, stay_ids, hadm_splits


def _scan_prescription_tokens(
    paths: MIMICDataPaths,
    *,
    hadm_ids: set[int],
    hadm_splits: Mapping[int, str],
    lookup: MedicationNormalizationLookup,
) -> tuple[Counter[str], Counter[str], PrescriptionNormalizationEvidence]:
    raw_medication_token_counter: Counter[str] = Counter()
    train_raw_medication_token_counter: Counter[str] = Counter()
    token_to_rxcui_counts: dict[str, Counter[str]] = {}
    rows_scanned = 0
    rows_with_target_token = 0
    rows_with_non_empty_ndc = 0
    rows_with_normalized_ndc = 0
    rows_with_mapped_ndc = 0
    unmatched_ndc_examples: list[dict[str, str]] = []

    for row in iter_table(
        paths,
        "prescriptions",
        fields=("hadm_id", "drug", "formulary_drug_cd", "ndc"),
    ):
        hadm_id = parse_int(row.get("hadm_id"))
        if hadm_id not in hadm_ids:
            continue

        rows_scanned += 1
        token = extract_medication_token(row)
        if not token:
            continue

        rows_with_target_token += 1
        raw_medication_token_counter[token] += 1
        if hadm_splits.get(int(hadm_id)) == "train":
            train_raw_medication_token_counter[token] += 1

        raw_ndc = str(row.get("ndc", "")).strip()
        if raw_ndc:
            rows_with_non_empty_ndc += 1
        ndc = normalize_ndc(raw_ndc)
        if not ndc:
            continue
        rows_with_normalized_ndc += 1
        resolved_rxcui = lookup.rxnorm_index.ndc_index.ndc_to_rxcui.get(ndc)
        if not resolved_rxcui:
            if len(unmatched_ndc_examples) < 20:
                unmatched_ndc_examples.append(
                    {
                        "raw_token": token,
                        "raw_ndc": raw_ndc,
                        "normalized_ndc": ndc,
                        "drug": str(row.get("drug", "")).strip(),
                        "formulary_drug_cd": str(row.get("formulary_drug_cd", "")).strip(),
                    }
                )
            continue
        rows_with_mapped_ndc += 1
        token_to_rxcui_counts.setdefault(token, Counter())[resolved_rxcui] += 1

    evidence = PrescriptionNormalizationEvidence(
        token_counter=Counter(raw_medication_token_counter),
        token_to_rxcui_counts={token: Counter(counter) for token, counter in token_to_rxcui_counts.items()},
        rows_scanned=int(rows_scanned),
        rows_with_target_token=int(rows_with_target_token),
        rows_with_non_empty_ndc=int(rows_with_non_empty_ndc),
        rows_with_normalized_ndc=int(rows_with_normalized_ndc),
        rows_with_mapped_ndc=int(rows_with_mapped_ndc),
        sample_unmatched_ndc_examples=tuple(unmatched_ndc_examples),
    )
    return raw_medication_token_counter, train_raw_medication_token_counter, evidence


def _scan_numeric_vocab_tokens_python(
    paths: MIMICDataPaths,
    *,
    table_name: str,
    id_column: str,
    scoped_ids: set[int],
    token_prefix: str,
) -> Counter[str]:
    if not scoped_ids:
        LOGGER.info("%s scan mode: python csv path (empty cohort scope)", table_name)
        return Counter()
    LOGGER.info("%s scan mode: python csv path", table_name)
    counter: Counter[str] = Counter()
    for row in iter_table(paths, table_name, fields=[id_column, "itemid", "valuenum"]):
        scoped_id = parse_int(row.get(id_column))
        itemid = str(row.get("itemid", "")).strip()
        value = parse_float(row.get("valuenum"))
        if scoped_id in scoped_ids and itemid and value is not None:
            counter[f"{token_prefix}:{itemid}"] += 1
    return counter


def _scan_numeric_vocab_tokens_spark(
    spark,
    config: Mapping[str, object],
    *,
    paths: MIMICDataPaths,
    table_name: str,
    id_column: str,
    scoped_ids: set[int],
    token_prefix: str,
) -> Counter[str]:
    from pyspark.sql import functions as F

    if not scoped_ids:
        LOGGER.info("%s scan mode: spark parquet path skipped (empty cohort scope)", table_name)
        return Counter()

    cache_dir = _ensure_stage_cache(
        spark,
        config,
        paths=paths,
        table_name=table_name,
        columns=(id_column, "itemid", "valuenum"),
    )
    LOGGER.info("%s scan mode: spark parquet path (cache=%s)", table_name, cache_dir)

    table_df = (
        spark.read.parquet(str(cache_dir))
        .select(
            F.col(id_column).cast("long").alias(id_column),
            F.col("itemid").cast("string").alias("itemid"),
            F.col("valuenum").cast("string").alias("valuenum"),
        )
        .filter(F.col(id_column).isNotNull())
        .filter(F.length(F.trim(F.col("itemid"))) > 0)
        .filter(F.col("valuenum").cast("double").isNotNull())
    )
    scoped_ids_df = spark.createDataFrame(
        [(int(value),) for value in sorted(scoped_ids)],
        schema=[id_column],
    )
    filtered_df = table_df.join(F.broadcast(scoped_ids_df), on=id_column, how="inner")
    rows = (
        filtered_df.groupBy("itemid")
        .count()
        .collect()
    )
    counter = Counter(
        {
            f"{token_prefix}:{str(row['itemid']).strip()}": int(row["count"])
            for row in rows
            if str(row["itemid"]).strip()
        }
    )
    return counter


def _scan_numeric_vocab_tokens(
    config: Mapping[str, object],
    *,
    spark,
    paths: MIMICDataPaths,
    table_name: str,
    id_column: str,
    scoped_ids: set[int],
    token_prefix: str,
) -> Counter[str]:
    if not spark_enabled(config):
        LOGGER.info("%s scan mode: disabled; using python csv path", table_name)
        return _scan_numeric_vocab_tokens_python(
            paths,
            table_name=table_name,
            id_column=id_column,
            scoped_ids=scoped_ids,
            token_prefix=token_prefix,
        )
    if spark is None:
        LOGGER.info("%s scan mode: python csv path", table_name)
        return _scan_numeric_vocab_tokens_python(
            paths,
            table_name=table_name,
            id_column=id_column,
            scoped_ids=scoped_ids,
            token_prefix=token_prefix,
        )
    try:
        return _scan_numeric_vocab_tokens_spark(
            spark,
            config,
            paths=paths,
            table_name=table_name,
            id_column=id_column,
            scoped_ids=scoped_ids,
            token_prefix=token_prefix,
        )
    except Exception as exc:
        LOGGER.warning(
            "Spark fast path for %s failed; falling back to python csv path. Error: %s",
            table_name,
            exc,
        )
        return _scan_numeric_vocab_tokens_python(
            paths,
            table_name=table_name,
            id_column=id_column,
            scoped_ids=scoped_ids,
            token_prefix=token_prefix,
        )


def _build_med_vocab_main(
    config: Mapping[str, object],
    *,
    raw_token_counter: Counter[str],
    train_raw_token_counter: Counter[str],
    lookup: MedicationNormalizationLookup,
    prescriptions_path: Path,
    hadm_ids: set[int],
    normalization_evidence: PrescriptionNormalizationEvidence | None = None,
) -> tuple[list[str], dict[str, dict[str, object]], dict[str, object]]:
    raw_tokens = _sorted_counter_items(raw_token_counter)
    token_matches, normalization_report = build_prescription_token_normalization_map(
        prescriptions_path,
        lookup=lookup,
        raw_tokens=raw_tokens,
        hadm_ids=hadm_ids,
        evidence=normalization_evidence,
    )

    canonical_counter: Counter[str] = Counter()
    train_canonical_counter: Counter[str] = Counter()
    med_vocab_metadata: dict[str, dict[str, object]] = {}
    raw_tokens_with_rxcui = 0
    raw_tokens_with_canonical_token = 0
    raw_tokens_collapsed_to_multiple_ingredients = 0

    for raw_token in raw_tokens:
        match = token_matches.get(raw_token)
        if match is None:
            continue
        if match.resolved_product_rxcui:
            raw_tokens_with_rxcui += 1
        if match.canonical_tokens:
            raw_tokens_with_canonical_token += 1
        if len(match.canonical_tokens) > 1:
            raw_tokens_collapsed_to_multiple_ingredients += 1

        raw_count = int(raw_token_counter.get(raw_token, 0))
        train_count = int(train_raw_token_counter.get(raw_token, 0))
        for canonical_token, canonical_rxcui, canonical_name in zip(
            match.canonical_tokens,
            match.canonical_rxcuis,
            match.canonical_names,
        ):
            canonical_counter[canonical_token] += raw_count
            train_canonical_counter[canonical_token] += train_count
            metadata_row = med_vocab_metadata.setdefault(
                canonical_token,
                {
                    "canonical_token": canonical_token,
                    "canonical_rxcui": canonical_rxcui,
                    "canonical_name": canonical_name,
                    "source_raw_tokens": [],
                    "source_match_sources": set(),
                    "raw_frequency": 0,
                    "train_frequency": 0,
                    "lexical_mismatch_name_token_count": 0,
                    "lexical_mismatch_name_examples": [],
                },
            )
            source_tokens = set(metadata_row["source_raw_tokens"])
            source_tokens.add(raw_token)
            metadata_row["source_raw_tokens"] = sorted(source_tokens)
            source_match_sources = set(metadata_row["source_match_sources"])
            if match.match_source:
                source_match_sources.add(str(match.match_source))
            metadata_row["source_match_sources"] = source_match_sources
            metadata_row["raw_frequency"] = int(metadata_row["raw_frequency"]) + raw_count
            metadata_row["train_frequency"] = int(metadata_row["train_frequency"]) + train_count
            if raw_token.startswith("NAME:") and not is_plausible_raw_to_ingredient_match(
                match.raw_token_body,
                canonical_rxcui,
                lookup=lookup,
            ):
                metadata_row["lexical_mismatch_name_token_count"] = int(
                    metadata_row["lexical_mismatch_name_token_count"]
                ) + 1
                mismatch_examples = list(metadata_row["lexical_mismatch_name_examples"])
                if len(mismatch_examples) < 10:
                    mismatch_examples.append(raw_token)
                metadata_row["lexical_mismatch_name_examples"] = mismatch_examples

    final_counter = Counter(
        {
            canonical_token: int(train_count)
            for canonical_token, train_count in train_canonical_counter.items()
            if int(train_count) >= DRUG_MIN_FREQ
        }
    )
    med_vocab_tokens = _sorted_counter_items(final_counter)
    med_vocab_file_size = len(med_vocab_tokens) + 2
    suspicious_canonical_examples = [
        {
            "canonical_token": canonical_token,
            "canonical_rxcui": metadata_row.get("canonical_rxcui", ""),
            "canonical_name": metadata_row.get("canonical_name", ""),
            "train_frequency": int(metadata_row.get("train_frequency", 0)),
            "source_raw_tokens": list(metadata_row.get("source_raw_tokens", []))[:10],
            "distinct_source_raw_token_count": int(len(metadata_row.get("source_raw_tokens", []))),
            "lexical_mismatch_name_token_count": int(metadata_row.get("lexical_mismatch_name_token_count", 0)),
            "lexical_mismatch_name_examples": list(metadata_row.get("lexical_mismatch_name_examples", []))[:10],
        }
        for canonical_token, metadata_row in sorted(
            med_vocab_metadata.items(),
            key=lambda item: (
                -int(item[1].get("lexical_mismatch_name_token_count", 0)),
                -len(item[1].get("source_raw_tokens", [])),
                -int(item[1].get("train_frequency", 0)),
                item[0],
            ),
        )
        if is_suspicious_ingredient_name(metadata_row.get("canonical_name", ""))
        or int(metadata_row.get("lexical_mismatch_name_token_count", 0)) > 0
    ][:20]

    for canonical_token, metadata_row in med_vocab_metadata.items():
        metadata_row["source_match_sources"] = sorted(
            str(value) for value in metadata_row["source_match_sources"]
        )
        metadata_row["in_final_vocab"] = canonical_token in final_counter
        metadata_row["distinct_source_raw_token_count"] = int(len(metadata_row.get("source_raw_tokens", [])))
        metadata_row["canonical_name_suspicious"] = is_suspicious_ingredient_name(
            metadata_row.get("canonical_name", "")
        )

    warn_min, warn_max = MED_VOCAB_MAIN_WARN_RANGE
    if not warn_min <= len(med_vocab_tokens) <= warn_max:
        LOGGER.warning(
            "Canonical med_vocab_main has %s labels (excluding PAD/UNK), outside expected range [%s, %s].",
            len(med_vocab_tokens),
            warn_min,
            warn_max,
        )
    if len(med_vocab_tokens) > 2000:
        LOGGER.warning(
            "Strong sanity warning: final med_vocab_main size is still very large (%s labels excluding PAD/UNK).",
            len(med_vocab_tokens),
        )
    if int(normalization_report.get("rows_with_mapped_ndc", 0)) == 0:
        LOGGER.warning(
            "Strong sanity warning: NDC path matched 0 prescription rows. Check prescriptions.ndc normalization and RXNSAT mapping."
        )
    if suspicious_canonical_examples:
        LOGGER.warning(
            "Strong sanity warning: %s canonical medication names still look product-like or dose/form-specific.",
            len(suspicious_canonical_examples),
        )

    LOGGER.info("Medication label space is true ingredient-level RxNorm canonical medication tokens.")
    LOGGER.info("Unique raw prescription medication tokens: %s", len(raw_tokens))
    LOGGER.info("Raw tokens matched to RxCUI: %s", raw_tokens_with_rxcui)
    LOGGER.info("Raw tokens matched to canonical medication tokens: %s", raw_tokens_with_canonical_token)
    LOGGER.info(
        "Final med_vocab_main size: %s labels (%s including PAD/UNK)",
        len(med_vocab_tokens),
        med_vocab_file_size,
    )

    report = {
        "representation": "rxnorm_true_ingredient_medication",
        "label_level": "ingredient_rxcui_true",
        "label_token_format": "MEDRX:<ingredient_rxcui>",
        "source_tables": [table_name for table_name, _ in MEDICATION_SOURCE_SPECS],
        "raw_token_count_total_rows": int(sum(raw_token_counter.values())),
        "raw_token_count_unique": int(len(raw_tokens)),
        "matched_rxcui_unique_token_count": int(raw_tokens_with_rxcui),
        "matched_canonical_token_unique_token_count": int(raw_tokens_with_canonical_token),
        "num_unique_canonical_ingredients": int(len(med_vocab_tokens)),
        "num_unique_canonical_ingredients_before_frequency_filter": int(len(canonical_counter)),
        "num_raw_tokens_collapsed_to_multiple_ingredients": int(raw_tokens_collapsed_to_multiple_ingredients),
        "matched_rxcui_total_rows": int(
            sum(
                int(raw_token_counter.get(token, 0))
                for token, match in token_matches.items()
                if match.resolved_product_rxcui
            )
        ),
        "matched_canonical_total_rows": int(
            sum(
                int(raw_token_counter.get(token, 0))
                for token, match in token_matches.items()
                if match.canonical_tokens
            )
        ),
        "unmatched_top_examples": list(normalization_report.get("unmatched_top_examples", [])),
        "match_source_counter": dict(normalization_report.get("match_source_counter", {})),
        "ambiguous_name_match_count": int(normalization_report.get("ambiguous_name_match_count", 0)),
        "suspicious_component_count": int(
            normalization_report.get("rxnorm_ingredient_meta", {}).get("suspicious_component_count", 0)
        ),
        "suspicious_component_examples": list(
            normalization_report.get("rxnorm_ingredient_meta", {}).get("suspicious_component_examples", [])
        )[:10],
        "suspicious_name_match_examples": list(
            normalization_report.get("suspicious_name_match_examples", [])
        )[:20],
        "suspicious_ndc_match_examples": list(
            normalization_report.get("suspicious_ndc_match_examples", [])
        )[:20],
        "final_med_vocab_main_size": int(len(med_vocab_tokens)),
        "final_vocab_file_size_including_specials": int(med_vocab_file_size),
        "suspicious_canonical_examples": suspicious_canonical_examples,
        "top_canonical_tokens_by_distinct_source_raw_tokens": [
            {
                "canonical_token": canonical_token,
                "canonical_name": metadata_row.get("canonical_name", ""),
                "canonical_rxcui": metadata_row.get("canonical_rxcui", ""),
                "distinct_source_raw_token_count": int(len(metadata_row.get("source_raw_tokens", []))),
                "source_raw_tokens": list(metadata_row.get("source_raw_tokens", []))[:20],
            }
            for canonical_token, metadata_row in sorted(
                med_vocab_metadata.items(),
                key=lambda item: (
                    -len(item[1].get("source_raw_tokens", [])),
                    -int(item[1].get("train_frequency", 0)),
                    item[0],
                ),
            )[:20]
        ],
        "top_canonical_tokens_by_lexical_mismatch_suspicion": [
            {
                "canonical_token": canonical_token,
                "canonical_name": metadata_row.get("canonical_name", ""),
                "canonical_rxcui": metadata_row.get("canonical_rxcui", ""),
                "lexical_mismatch_name_token_count": int(
                    metadata_row.get("lexical_mismatch_name_token_count", 0)
                ),
                "distinct_source_raw_token_count": int(len(metadata_row.get("source_raw_tokens", []))),
                "lexical_mismatch_name_examples": list(
                    metadata_row.get("lexical_mismatch_name_examples", [])
                )[:20],
            }
            for canonical_token, metadata_row in sorted(
                med_vocab_metadata.items(),
                key=lambda item: (
                    -int(item[1].get("lexical_mismatch_name_token_count", 0)),
                    -len(item[1].get("source_raw_tokens", [])),
                    item[0],
                ),
            )
            if int(metadata_row.get("lexical_mismatch_name_token_count", 0)) > 0
        ][:20],
        "top_30_final_med_tokens_by_train_frequency": [
            {
                "canonical_token": token,
                "canonical_name": med_vocab_metadata.get(token, {}).get("canonical_name", ""),
                "canonical_rxcui": med_vocab_metadata.get(token, {}).get("canonical_rxcui", ""),
                "train_frequency": int(final_counter[token]),
            }
            for token in med_vocab_tokens[:30]
        ],
        "normalization_report": normalization_report,
    }
    write_json(vocab_dir_from_config(config) / "med_vocab_main_build_report.json", report)
    return med_vocab_tokens, med_vocab_metadata, report


def _write_vocab_outputs(
    config: Mapping[str, object],
    *,
    diag_tokens: list[str],
    proc_tokens: list[str],
    med_main_tokens: list[str],
    med_main_metadata: dict[str, dict[str, object]],
    lab_tokens: list[str],
    vital_tokens: list[str],
) -> Path:
    vocab_dir = vocab_dir_from_config(config)
    paths = MIMICDataPaths.from_config(config)

    diag_vocab = _build_vocab_payload(diag_tokens, "diag_vocab")
    proc_vocab = _build_vocab_payload(proc_tokens, "proc_vocab")
    med_vocab_main = _build_vocab_payload(med_main_tokens, "med_vocab_main")
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

    write_json(vocab_dir / "diag_vocab.json", diag_vocab)
    write_json(vocab_dir / "proc_vocab.json", proc_vocab)
    write_json(vocab_dir / "med_vocab_main.json", med_vocab_main)
    write_json(vocab_dir / "med_vocab_main_metadata.json", med_main_metadata)
    write_json(vocab_dir / "diagnosis_vocab.json", diag_vocab)
    write_json(vocab_dir / "procedure_vocab.json", proc_vocab)
    write_json(vocab_dir / "lab_vocab.json", lab_vocab)
    write_json(vocab_dir / "vital_vocab.json", vital_vocab)
    write_json(vocab_dir / "lab_metadata.json", lab_metadata)
    write_json(vocab_dir / "vital_metadata.json", vital_metadata)
    write_json(
        vocab_dir / "vocab_summary.json",
        {
            "diag_size": diag_vocab["size"],
            "proc_size": proc_vocab["size"],
            "med_vocab_main_size": med_vocab_main["size"],
            "diagnosis_size": diag_vocab["size"],
            "procedure_size": proc_vocab["size"],
            "lab_size": lab_vocab["size"],
            "vital_size": vital_vocab["size"],
        },
    )
    return vocab_dir


def build_vocab(config_path: str | Path) -> Path:
    _configure_logging()
    build_start = perf_counter()
    config = load_yaml_config(config_path)
    if spark_enabled(config):
        LOGGER.info(
            "build_vocab will use the Spark fast path only for labevents and chartevents when parquet cache is available."
        )

    paths = MIMICDataPaths.from_config(config)
    cohort_path = cohort_path_from_config(config)
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Cohort artifact is missing at {cohort_path}. Run build_cohort.py first."
        )
    cohort_phase_start = perf_counter()
    hadm_ids, stay_ids, hadm_splits = _load_cohort_ids_and_splits(cohort_path)
    _log_phase_timing(
        "load cohort ids",
        cohort_phase_start,
        hadm_ids=len(hadm_ids),
        stay_ids=len(stay_ids),
        split_hadm_ids=len(hadm_splits),
    )

    feature_cfg = config.get("features", {})
    top_k_labs = int(feature_cfg.get("top_k_labs", 64))
    top_k_vitals = int(feature_cfg.get("top_k_vitals", 64))
    rxnorm_root = _resolve_rxnorm_root(config)
    rxnorm_use_cache = bool(config.get("rxnorm_use_cache", True))
    rxnorm_force_rebuild = bool(config.get("rxnorm_force_rebuild", False))
    rxnorm_cache_dir_raw = config.get("rxnorm_cache_dir")
    rxnorm_cache_dir = (
        Path(resolve_path(config["_project_root"], rxnorm_cache_dir_raw))
        if rxnorm_cache_dir_raw
        else None
    )
    if rxnorm_force_rebuild:
        LOGGER.info("RxNorm cache mode: force rebuild")
    elif rxnorm_use_cache:
        LOGGER.info("RxNorm cache mode: enabled")
    else:
        LOGGER.info("RxNorm cache mode: disabled")
    medication_lookup = build_medication_normalization_lookup(
        rxnorm_root,
        use_cache=rxnorm_use_cache,
        force_rebuild=rxnorm_force_rebuild,
        cache_dir=rxnorm_cache_dir,
    )

    diagnosis_counter: Counter[str] = Counter()
    procedure_counter: Counter[str] = Counter()
    lab_counter: Counter[str] = Counter()
    vital_counter: Counter[str] = Counter()
    vocab_spark = None

    if spark_enabled(config):
        try:
            vocab_spark = build_spark_session(config, app_name="ClinRecBuildVocab")
            vocab_spark.sparkContext.setLogLevel("WARN")
        except Exception as exc:
            LOGGER.warning(
                "Unable to initialize Spark fast path for numeric vocab scans; using python csv path. Error: %s",
                exc,
            )
            vocab_spark = None

    try:
        diagnoses_phase_start = perf_counter()
        for row in iter_table(paths, "diagnoses_icd", fields=["hadm_id", "icd_code", "icd_version"]):
            hadm_id = parse_int(row.get("hadm_id"))
            code = str(row.get("icd_code", "")).strip()
            version = str(row.get("icd_version", "")).strip()
            if hadm_id in hadm_ids and code and version:
                diagnosis_counter[f"ICD{version}:{code}"] += 1
        _log_phase_timing(
            "diagnoses scan",
            diagnoses_phase_start,
            unique_diag_tokens=len(diagnosis_counter),
            matched_rows=sum(diagnosis_counter.values()),
        )

        procedures_phase_start = perf_counter()
        for row in iter_table(paths, "procedures_icd", fields=["hadm_id", "icd_code", "icd_version"]):
            hadm_id = parse_int(row.get("hadm_id"))
            code = str(row.get("icd_code", "")).strip()
            version = str(row.get("icd_version", "")).strip()
            if hadm_id in hadm_ids and code and version:
                procedure_counter[f"PROC{version}:{code}"] += 1
        _log_phase_timing(
            "procedures scan",
            procedures_phase_start,
            unique_proc_tokens=len(procedure_counter),
            matched_rows=sum(procedure_counter.values()),
        )

        prescriptions_phase_start = perf_counter()
        (
            raw_medication_token_counter,
            train_raw_medication_token_counter,
            prescription_evidence,
        ) = _scan_prescription_tokens(
            paths,
            hadm_ids=hadm_ids,
            hadm_splits=hadm_splits,
            lookup=medication_lookup,
        )
        _log_phase_timing(
            "prescriptions raw token scan",
            prescriptions_phase_start,
            rows_scanned=prescription_evidence.rows_scanned,
            token_rows=prescription_evidence.rows_with_target_token,
            rows_with_non_empty_ndc=prescription_evidence.rows_with_non_empty_ndc,
            rows_with_normalized_ndc=prescription_evidence.rows_with_normalized_ndc,
            rows_with_mapped_ndc=prescription_evidence.rows_with_mapped_ndc,
            unique_raw_tokens=len(raw_medication_token_counter),
        )

        medication_normalization_phase_start = perf_counter()
        med_main_tokens, med_main_metadata, _ = _build_med_vocab_main(
            config,
            raw_token_counter=raw_medication_token_counter,
            train_raw_token_counter=train_raw_medication_token_counter,
            lookup=medication_lookup,
            prescriptions_path=paths.table_path("prescriptions"),
            hadm_ids=hadm_ids,
            normalization_evidence=prescription_evidence,
        )
        _log_phase_timing(
            "medication normalization",
            medication_normalization_phase_start,
            med_vocab_main_size=len(med_main_tokens),
            metadata_rows=len(med_main_metadata),
        )

        labevents_phase_start = perf_counter()
        lab_counter = _scan_numeric_vocab_tokens(
            config,
            spark=vocab_spark,
            paths=paths,
            table_name="labevents",
            id_column="hadm_id",
            scoped_ids=hadm_ids,
            token_prefix="LAB",
        )
        _log_phase_timing(
            "labevents scan",
            labevents_phase_start,
            unique_lab_tokens=len(lab_counter),
            matched_rows=sum(lab_counter.values()),
        )

        chartevents_phase_start = perf_counter()
        vital_counter = _scan_numeric_vocab_tokens(
            config,
            spark=vocab_spark,
            paths=paths,
            table_name="chartevents",
            id_column="stay_id",
            scoped_ids=stay_ids,
            token_prefix="VITAL",
        )
        _log_phase_timing(
            "chartevents scan",
            chartevents_phase_start,
            unique_vital_tokens=len(vital_counter),
            matched_rows=sum(vital_counter.values()),
        )
    finally:
        if vocab_spark is not None:
            vocab_spark.stop()

    vocab_dir = _write_vocab_outputs(
        config,
        diag_tokens=_sorted_counter_items(diagnosis_counter),
        proc_tokens=_sorted_counter_items(procedure_counter),
        med_main_tokens=med_main_tokens,
        med_main_metadata=med_main_metadata,
        lab_tokens=_sorted_counter_items(lab_counter, top_k=top_k_labs),
        vital_tokens=_sorted_counter_items(vital_counter, top_k=top_k_vitals),
    )
    LOGGER.info("Built vocab artifacts at %s", vocab_dir)
    LOGGER.info("build_vocab finished in %.2fs", perf_counter() - build_start)
    return vocab_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vocabularies from cohort-filtered MIMIC tables.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to configs/data.yaml")
    args = parser.parse_args()
    build_vocab(args.config)


if __name__ == "__main__":
    main()
