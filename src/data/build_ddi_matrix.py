from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.ddinter_utils import (
    DEFAULT_DDINTER_GLOB,
    LEVEL_SEVERITY_PRIORITY,
    load_ddinter_dataset,
    match_names_to_ddinter_ids,
    normalize_ddinter_text,
    project_ddinter_pairs_to_vocab,
)
from src.data.drugbank_vocab_utils import load_drugbank_vocabulary_index
from src.data.rxnorm_utils import build_minimal_rxnorm_index, normalize_rxcui
from src.utils.io import (
    ensure_dir,
    load_yaml_config,
    read_json,
    resolve_path,
    save_pt,
    write_json,
)


LOGGER = logging.getLogger(__name__)

SPECIAL_VOCAB_TOKENS = {"PAD", "UNK"}
CANONICAL_DDI_REPRESENTATION = "med_vocab_main"
CURRENT_REPRESENTATION_ALIASES = {
    "",
    CANONICAL_DDI_REPRESENTATION,
    "rxnorm_canonical_medication",
    "rxnorm_ingredient",
}


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
    )


def _ddi_output_paths(config: Mapping[str, Any]) -> tuple[Path, Path]:
    paths_cfg = dict(config.get("paths", {}))
    ddi_root = paths_cfg.get("ddi_root")
    if ddi_root:
        ddi_dir = ensure_dir(resolve_path(config["_project_root"], ddi_root))
    else:
        processed_root = resolve_path(config["_project_root"], paths_cfg["processed_root"])
        ddi_dir = ensure_dir(Path(processed_root) / "ddi")
    return ddi_dir / "drug_ddi.pt", ddi_dir / "drug_ddi_report.json"


def _vocab_dir_from_config(config: Mapping[str, Any]) -> Path:
    paths_cfg = dict(config.get("paths", {}))
    vocab_root = paths_cfg.get("vocab_root")
    if vocab_root:
        return Path(resolve_path(config["_project_root"], vocab_root))
    interim_root = resolve_path(config["_project_root"], paths_cfg["interim_root"])
    return Path(interim_root) / "vocab"


def _resolve_required_source_path(
    config: Mapping[str, Any],
    *,
    key: str,
    description: str,
    defaults: Iterable[str] = (),
) -> Path:
    paths_cfg = dict(config.get("paths", {}))
    candidate_paths: list[Path] = []

    configured = paths_cfg.get(key)
    if configured:
        candidate_paths.append(Path(resolve_path(config["_project_root"], configured)))
    for default in defaults:
        candidate_paths.append(Path(resolve_path(config["_project_root"], default)))

    seen: set[Path] = set()
    deduped_candidates: list[Path] = []
    for candidate in candidate_paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped_candidates.append(candidate)

    for candidate in deduped_candidates:
        if candidate.exists():
            return candidate

    checked = [str(path) for path in deduped_candidates]
    raise FileNotFoundError(f"Unable to locate {description}. Checked: {checked}")


def _resolve_ddi_representation(config: Mapping[str, Any], warnings: list[str]) -> str:
    raw_value = str(config.get("ddi_representation", "")).strip().lower()
    if raw_value in CURRENT_REPRESENTATION_ALIASES:
        return CANONICAL_DDI_REPRESENTATION
    warning = (
        "Ignoring ddi_representation="
        f"{raw_value!r}; build_ddi_matrix always projects DDInter to med_vocab_main."
    )
    LOGGER.warning(warning)
    warnings.append(warning)
    return CANONICAL_DDI_REPRESENTATION


def _load_med_vocab_main(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str]]:
    vocab_dir = _vocab_dir_from_config(config)
    med_vocab_path = vocab_dir / "med_vocab_main.json"
    med_vocab_metadata_path = vocab_dir / "med_vocab_main_metadata.json"

    if not med_vocab_path.exists():
        raise FileNotFoundError(
            "Missing med_vocab_main artifact. Expected "
            f"{med_vocab_path}. Run build_vocab.py first."
        )

    med_vocab = read_json(med_vocab_path)
    med_vocab_metadata = (
        read_json(med_vocab_metadata_path) if med_vocab_metadata_path.exists() else {}
    )
    if "idx_to_token" not in med_vocab:
        raise ValueError(
            f"Invalid med_vocab_main artifact at {med_vocab_path}: missing `idx_to_token`."
        )
    if "pad_idx" not in med_vocab or "unk_idx" not in med_vocab:
        raise ValueError(
            f"Invalid med_vocab_main artifact at {med_vocab_path}: missing PAD/UNK indexes."
        )

    return med_vocab, med_vocab_metadata, {
        "med_vocab_main_path": str(med_vocab_path),
        "med_vocab_main_metadata_path": str(med_vocab_metadata_path),
    }


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _medication_token_body(token: str) -> str:
    return str(token).split(":", 1)[1] if ":" in str(token) else str(token)


def _resolve_canonical_rxcui(token: str, metadata_row: Mapping[str, Any]) -> str | None:
    metadata_rxcui = normalize_rxcui(str(metadata_row.get("canonical_rxcui", "")))
    if metadata_rxcui:
        return metadata_rxcui

    token_text = str(token).strip()
    if token_text.startswith("MEDRX:"):
        return normalize_rxcui(_medication_token_body(token_text))
    return None


def _candidate_name_from_raw_token(raw_token: str | None) -> str | None:
    text = str(raw_token or "").strip()
    if not text:
        return None

    prefix = ""
    body = text
    if ":" in text:
        prefix, body = text.split(":", 1)
        if str(prefix).strip().upper() != "NAME":
            return None

    candidate = body.replace("_", " ").strip()
    if not candidate or not any(character.isalpha() for character in candidate):
        return None
    return candidate


def _build_vocab_candidate_index(
    med_vocab: Mapping[str, Any],
    med_vocab_metadata: Mapping[str, Mapping[str, Any]],
    *,
    rxnorm_index: Any,
) -> tuple[dict[int, tuple[str, ...]], dict[int, dict[str, Any]], dict[str, Any]]:
    vocab_item_to_candidate_names: dict[int, tuple[str, ...]] = {}
    candidate_profiles: dict[int, dict[str, Any]] = {}

    num_real_vocab_items = 0
    num_items_with_canonical_rxcui = 0
    num_items_with_metadata_name = 0
    num_items_with_rxnorm_name = 0
    num_items_with_rxnorm_synonyms = 0
    num_items_with_raw_name_candidates = 0

    for item_index, token in enumerate(med_vocab["idx_to_token"]):
        token_text = str(token).strip()
        if token_text in SPECIAL_VOCAB_TOKENS:
            continue

        num_real_vocab_items += 1
        metadata_row = dict(med_vocab_metadata.get(token_text, {}))
        canonical_rxcui = _resolve_canonical_rxcui(token_text, metadata_row)
        if canonical_rxcui:
            num_items_with_canonical_rxcui += 1

        candidate_names: list[str] = []
        canonical_name = str(metadata_row.get("canonical_name", "")).strip()
        if canonical_name:
            candidate_names.append(canonical_name)
            num_items_with_metadata_name += 1

        if canonical_rxcui:
            name_record = rxnorm_index.name_index.rxcui_to_names.get(canonical_rxcui)
            if name_record is not None:
                preferred_name = str(name_record.preferred_name).strip()
                if preferred_name:
                    candidate_names.append(preferred_name)
                    num_items_with_rxnorm_name += 1
                if name_record.synonyms:
                    candidate_names.extend(
                        str(name).strip()
                        for name in name_record.synonyms
                        if str(name).strip()
                    )
                    num_items_with_rxnorm_synonyms += 1

            ingredient_name = str(
                rxnorm_index.ingredient_index.ingredient_rxcui_to_name.get(canonical_rxcui, "")
            ).strip()
            if ingredient_name:
                candidate_names.append(ingredient_name)

            ingredient_names = rxnorm_index.ingredient_index.rxcui_to_ingredient_names.get(
                canonical_rxcui,
                (),
            )
            candidate_names.extend(
                str(name).strip()
                for name in ingredient_names
                if str(name).strip()
            )

        raw_name_candidates = [
            candidate
            for raw_token in metadata_row.get("source_raw_tokens", [])
            if (candidate := _candidate_name_from_raw_token(str(raw_token)))
        ]
        if raw_name_candidates:
            candidate_names.extend(raw_name_candidates)
            num_items_with_raw_name_candidates += 1

        fallback_name = _medication_token_body(token_text).replace("_", " ").strip()
        if fallback_name and any(character.isalpha() for character in fallback_name):
            candidate_names.append(fallback_name)

        finalized_candidate_names = tuple(_dedupe_preserve_order(candidate_names))
        normalized_candidate_names = tuple(
            _dedupe_preserve_order(
                normalized_name
                for candidate_name in finalized_candidate_names
                if (normalized_name := normalize_ddinter_text(candidate_name))
            )
        )
        canonical_match_key = normalized_candidate_names[0] if normalized_candidate_names else None

        vocab_item_to_candidate_names[item_index] = finalized_candidate_names
        candidate_profiles[item_index] = {
            "item_index": int(item_index),
            "token": token_text,
            "canonical_rxcui": canonical_rxcui,
            "canonical_name": canonical_name or None,
            "candidate_names": list(finalized_candidate_names[:20]),
            "normalized_candidate_names": list(normalized_candidate_names[:20]),
            "canonical_match_key": canonical_match_key,
            "source_raw_tokens": list(metadata_row.get("source_raw_tokens", []))[:20],
        }

    meta = {
        "num_real_vocab_items": int(num_real_vocab_items),
        "num_items_with_canonical_rxcui": int(num_items_with_canonical_rxcui),
        "num_items_with_metadata_name": int(num_items_with_metadata_name),
        "num_items_with_rxnorm_name": int(num_items_with_rxnorm_name),
        "num_items_with_rxnorm_synonyms": int(num_items_with_rxnorm_synonyms),
        "num_items_with_raw_name_candidates": int(num_items_with_raw_name_candidates),
        "sample_candidate_profiles": [
            candidate_profiles[item_index]
            for item_index in sorted(candidate_profiles)[:20]
        ],
    }
    return vocab_item_to_candidate_names, candidate_profiles, meta


def _match_vocab_items_to_ddinter(
    vocab_item_to_candidate_names: Mapping[int, Iterable[str]],
    candidate_profiles: Mapping[int, Mapping[str, Any]],
    *,
    ddinter_dataset: Any,
    drugbank_index: Any,
    example_limit: int = 20,
) -> tuple[dict[int, tuple[str, ...]], dict[str, Any]]:
    vocab_item_to_ddinter_ids: dict[int, tuple[str, ...]] = {}
    matched_ddinter_ids: set[str] = set()
    unmatched_examples: list[dict[str, Any]] = []
    sample_matches: list[dict[str, Any]] = []

    direct_name_match_count = 0
    drugbank_bridge_match_count = 0
    unmatched_name_count = 0

    for item_index in sorted(vocab_item_to_candidate_names):
        candidate_name_list = [
            str(value).strip()
            for value in vocab_item_to_candidate_names[item_index]
            if str(value).strip()
        ]
        resolved_ids, match_report = match_names_to_ddinter_ids(
            candidate_name_list,
            ddinter_dataset=ddinter_dataset,
            drugbank_index=drugbank_index,
        )

        direct_name_match_count += len(match_report["direct_name_matches"])
        drugbank_bridge_match_count += len(match_report["drugbank_bridge_matches"])
        unmatched_name_count += len(match_report["unmatched_names"])

        if resolved_ids:
            sorted_ids = tuple(sorted(resolved_ids))
            vocab_item_to_ddinter_ids[item_index] = sorted_ids
            matched_ddinter_ids.update(sorted_ids)
            if len(sample_matches) < example_limit:
                sample_matches.append(
                    {
                        **dict(candidate_profiles.get(item_index, {})),
                        "ddinter_ids": list(sorted_ids),
                        "matched_by_direct_names": sorted(
                            str(value) for value in match_report["direct_name_matches"]
                        )[:10],
                        "matched_by_drugbank_bridge": sorted(
                            str(value) for value in match_report["drugbank_bridge_matches"]
                        )[:10],
                    }
                )
            continue

        if len(unmatched_examples) < example_limit:
            unmatched_examples.append(
                {
                    **dict(candidate_profiles.get(item_index, {})),
                    "unmatched_names": list(match_report["unmatched_names"])[:10],
                }
            )

    meta = {
        "num_vocab_items": int(len(vocab_item_to_candidate_names)),
        "num_vocab_items_matched": int(len(vocab_item_to_ddinter_ids)),
        "num_vocab_items_unmatched": int(
            len(vocab_item_to_candidate_names) - len(vocab_item_to_ddinter_ids)
        ),
        "matched_ddinter_id_count": int(len(matched_ddinter_ids)),
        "direct_name_match_count": int(direct_name_match_count),
        "drugbank_bridge_match_count": int(drugbank_bridge_match_count),
        "unmatched_name_count": int(unmatched_name_count),
        "sample_matches": sample_matches,
        "unmatched_examples": unmatched_examples,
    }
    LOGGER.info(
        "Matched %s/%s medication vocab items to DDInter IDs",
        len(vocab_item_to_ddinter_ids),
        len(vocab_item_to_candidate_names),
    )
    return vocab_item_to_ddinter_ids, meta


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "build_ddi_matrix requires `torch` to serialize tensor outputs."
        ) from exc
    return torch


def _build_ddi_tensors(
    projected_pairs: Iterable[Any],
    *,
    vocab_size: int,
    example_limit: int = 20,
) -> tuple[Any, Any, dict[str, Any]]:
    torch = _require_torch()
    matrix = torch.zeros((vocab_size, vocab_size), dtype=torch.uint8)
    severity_matrix = torch.zeros((vocab_size, vocab_size), dtype=torch.uint8)

    severity_counter: Counter[str] = Counter()
    unknown_severity_levels: set[str] = set()
    pair_metadata_examples: list[dict[str, Any]] = []

    for pair in projected_pairs:
        left_index = int(pair.left_item_id)
        right_index = int(pair.right_item_id)
        if left_index == right_index:
            continue

        matrix[left_index, right_index] = 1
        matrix[right_index, left_index] = 1

        severity_level = str(pair.max_severity_level or "unknown").strip().lower()
        if severity_level not in LEVEL_SEVERITY_PRIORITY:
            unknown_severity_levels.add(severity_level)
        severity_weight = int(LEVEL_SEVERITY_PRIORITY.get(severity_level, 0))
        current_weight = int(severity_matrix[left_index, right_index].item())
        if severity_weight > current_weight:
            severity_matrix[left_index, right_index] = severity_weight
            severity_matrix[right_index, left_index] = severity_weight

        severity_counter[severity_level] += 1

        if len(pair_metadata_examples) < example_limit:
            pair_metadata_examples.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "max_severity_level": pair.max_severity_level,
                    "dominant_level": pair.dominant_level,
                    "level_counts": list(pair.level_counts),
                    "row_count": int(pair.row_count),
                    "ddinter_pairs": [list(ddinter_pair) for ddinter_pair in pair.ddinter_pairs[:10]],
                }
            )

    matrix.fill_diagonal_(0)
    severity_matrix.fill_diagonal_(0)

    unique_pair_count = int(torch.triu(matrix, diagonal=1).sum().item())
    matrix_nonzero = int(matrix.sum().item())
    severity_nonzero = int((severity_matrix > 0).sum().item())

    meta = {
        "num_projected_vocab_pairs": int(unique_pair_count),
        "matrix_nonzero": int(matrix_nonzero),
        "severity_nonzero": int(severity_nonzero),
        "severity_counts": {key: int(value) for key, value in sorted(severity_counter.items())},
        "unknown_severity_levels": sorted(
            level for level in unknown_severity_levels if str(level).strip()
        ),
        "pair_metadata_examples": pair_metadata_examples,
    }
    return matrix, severity_matrix, meta


def build_ddi_matrix(config_path: str | Path) -> Path:
    """
    Build a DDInter-backed DDI matrix aligned to ``med_vocab_main``.

    The pipeline is:
    1. load ``med_vocab_main`` and its metadata
    2. derive candidate drug names from canonical RxNorm ingredient labels
    3. use DrugBank Vocabulary as an alias bridge
    4. match onto DDInter entities
    5. project DDInter pairs back to the benchmark medication vocabulary
    6. persist binary and severity-aware tensor artifacts plus a coverage report
    """

    _configure_logging()
    config = load_yaml_config(config_path)
    warnings: list[str] = []
    representation = _resolve_ddi_representation(config, warnings)

    matrix_path, report_path = _ddi_output_paths(config)
    med_vocab, med_vocab_metadata, med_vocab_paths = _load_med_vocab_main(config)
    vocab_tokens = [str(token) for token in med_vocab["idx_to_token"]]
    vocab_size = len(vocab_tokens)
    real_vocab_tokens = [token for token in vocab_tokens if token not in SPECIAL_VOCAB_TOKENS]

    rxnorm_root = _resolve_required_source_path(
        config,
        key="rxnorm_root",
        description="RxNorm release directory",
        defaults=(
            "data/processed/ddi/RxNorm_full_04062026",
            "data/raw/ddi/RxNorm_full_04062026",
        ),
    )
    drugbank_vocab_path = _resolve_required_source_path(
        config,
        key="drugbank_vocab_path",
        description="DrugBank vocabulary CSV",
        defaults=("data/processed/ddi/drugbank vocabulary.csv",),
    )
    ddinter_root = _resolve_required_source_path(
        config,
        key="ddinter_root",
        description="DDInter directory or CSV",
        defaults=("data/processed/ddi",),
    )
    ddinter_glob = (
        str(dict(config.get("paths", {})).get("ddinter_glob", DEFAULT_DDINTER_GLOB)).strip()
        or DEFAULT_DDINTER_GLOB
    )
    include_unknown = bool(config.get("ddinter_include_unknown", True))

    LOGGER.info("Loading RxNorm tables from %s", rxnorm_root)
    rxnorm_index = build_minimal_rxnorm_index(rxnorm_root)

    LOGGER.info("Loading DrugBank Vocabulary from %s", drugbank_vocab_path)
    drugbank_index = load_drugbank_vocabulary_index(drugbank_vocab_path)

    LOGGER.info(
        "Loading DDInter from %s using glob=%s (include_unknown=%s)",
        ddinter_root,
        ddinter_glob,
        include_unknown,
    )
    ddinter_dataset = load_ddinter_dataset(
        ddinter_root,
        glob_pattern=ddinter_glob,
        include_unknown=include_unknown,
    )

    vocab_item_to_candidate_names, candidate_profiles, candidate_meta = _build_vocab_candidate_index(
        med_vocab,
        med_vocab_metadata,
        rxnorm_index=rxnorm_index,
    )
    vocab_item_to_ddinter_ids, vocab_match_meta = _match_vocab_items_to_ddinter(
        vocab_item_to_candidate_names,
        candidate_profiles,
        ddinter_dataset=ddinter_dataset,
        drugbank_index=drugbank_index,
    )
    projected_pairs, pair_projection_meta = project_ddinter_pairs_to_vocab(
        vocab_item_to_ddinter_ids,
        ddinter_dataset=ddinter_dataset,
    )
    matrix, severity_matrix, matrix_meta = _build_ddi_tensors(projected_pairs, vocab_size=vocab_size)

    num_vocab_drugs = int(len(real_vocab_tokens))
    num_vocab_drugs_matched = int(vocab_match_meta["num_vocab_items_matched"])
    num_ddi_pairs_raw = int(len(ddinter_dataset.pair_records))
    num_ddi_pairs_matched = int(matrix_meta["num_projected_vocab_pairs"])
    match_rate = (
        float(num_vocab_drugs_matched) / float(num_vocab_drugs)
        if num_vocab_drugs > 0
        else 0.0
    )

    if num_vocab_drugs_matched == 0:
        warning = "No medication vocabulary items were matched to DDInter entities."
        LOGGER.warning(warning)
        warnings.append(warning)
    if num_ddi_pairs_matched == 0:
        warning = "Projected DDInter pair count is 0; the saved DDI matrix will be all zeros."
        LOGGER.warning(warning)
        warnings.append(warning)
    if matrix_meta["unknown_severity_levels"]:
        warning = (
            "Observed unrecognized DDInter severity levels: "
            f"{matrix_meta['unknown_severity_levels'][:10]}"
        )
        LOGGER.warning(warning)
        warnings.append(warning)

    notes = [
        "Matrix rows and columns follow med_vocab_main.json, including PAD and UNK as zero rows.",
        "Binary adjacency marks any DDInter interaction projected onto the benchmark medication vocabulary.",
        "severity_matrix stores the maximum DDInter severity weight observed per vocab pair.",
    ]

    report = {
        "representation": representation,
        "label_level": "ingredient_rxcui",
        "label_token_format": "MEDRX:<ingredient_rxcui>",
        "num_vocab_tokens_total": int(vocab_size),
        "num_vocab_drugs": int(num_vocab_drugs),
        "num_vocab_drugs_matched": int(num_vocab_drugs_matched),
        "match_rate": float(match_rate),
        "num_ddi_pairs_raw": int(num_ddi_pairs_raw),
        "num_ddi_pairs_matched": int(num_ddi_pairs_matched),
        "matrix_shape": [int(vocab_size), int(vocab_size)],
        "matrix_nonzero": int(matrix_meta["matrix_nonzero"]),
        "matrix_nonzero_upper": int(matrix_meta["num_projected_vocab_pairs"]),
        "severity_nonzero": int(matrix_meta["severity_nonzero"]),
        "severity_counts": dict(matrix_meta["severity_counts"]),
        "severity_level_to_weight": {
            key: int(value) for key, value in sorted(LEVEL_SEVERITY_PRIORITY.items())
        },
        "matched_pairs": int(num_ddi_pairs_matched),
        "num_vocab_tokens_mapped_to_ddinter": int(num_vocab_drugs_matched),
        "num_interacting_pairs_real_labels": int(num_ddi_pairs_matched),
        "unmatched_vocab_examples": list(vocab_match_meta["unmatched_examples"]),
        "notes": notes,
        "warnings": warnings,
        "source_paths": {
            **med_vocab_paths,
            "rxnorm_root": str(rxnorm_root),
            "drugbank_vocab_path": str(drugbank_vocab_path),
            "ddinter_root": str(ddinter_root),
            "ddinter_glob": ddinter_glob,
        },
        "rxnorm_meta": dict(rxnorm_index.meta),
        "drugbank_meta": dict(drugbank_index.meta),
        "ddinter_meta": dict(ddinter_dataset.meta),
        "candidate_generation": candidate_meta,
        "vocab_match": vocab_match_meta,
        "pair_projection": pair_projection_meta,
        "ddinter_projection": {
            **{
                key: value
                for key, value in vocab_match_meta.items()
                if key
                in {
                    "num_vocab_items",
                    "num_vocab_items_matched",
                    "num_vocab_items_unmatched",
                    "matched_ddinter_id_count",
                    "direct_name_match_count",
                    "drugbank_bridge_match_count",
                    "unmatched_name_count",
                }
            },
            **{
                key: value
                for key, value in pair_projection_meta.items()
                if key
                in {
                    "num_vocab_items_with_ddinter_ids",
                    "num_ddinter_ids_in_vocab_map",
                    "num_ddinter_pair_records",
                    "num_ddinter_pair_records_with_vocab_match",
                    "num_ddinter_pair_records_without_vocab_match",
                    "num_pair_records_collapsed_to_same_item",
                    "num_projected_vocab_pairs",
                }
            },
        },
        "pair_metadata_examples": list(matrix_meta["pair_metadata_examples"]),
    }

    LOGGER.info("Saving DDI tensor payload to %s", matrix_path)
    save_pt(
        matrix_path,
        {
            "matrix": matrix,
            "severity_matrix": severity_matrix,
            "severity_level_to_weight": {
                key: int(value) for key, value in sorted(LEVEL_SEVERITY_PRIORITY.items())
            },
            "representation": representation,
            "vocab_name": str(med_vocab.get("name", CANONICAL_DDI_REPRESENTATION)),
            "vocab_size": int(vocab_size),
            "pad_idx": int(med_vocab["pad_idx"]),
            "unk_idx": int(med_vocab["unk_idx"]),
            "num_vocab_drugs": int(num_vocab_drugs),
            "num_vocab_drugs_matched": int(num_vocab_drugs_matched),
            "num_ddi_pairs_raw": int(num_ddi_pairs_raw),
            "num_ddi_pairs_matched": int(num_ddi_pairs_matched),
            "matched_pairs": int(num_ddi_pairs_matched),
            "pair_metadata_examples": list(matrix_meta["pair_metadata_examples"]),
        },
    )
    write_json(report_path, report)

    LOGGER.info(
        "DDI matrix built successfully (matched_vocab=%s/%s, matched_pairs=%s)",
        num_vocab_drugs_matched,
        num_vocab_drugs,
        num_ddi_pairs_matched,
    )
    return matrix_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a DDInter-backed DDI matrix aligned to the benchmark med_vocab_main."
    )
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    args = parser.parse_args()
    build_ddi_matrix(args.config)


if __name__ == "__main__":
    main()
