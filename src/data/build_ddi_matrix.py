from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import warnings
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Mapping

from src.data.build_vocab import load_vocab_bundle
from src.data.load_mimic import open_csv
from src.features.medication_history import canonicalize_medication_text
from src.utils.io import (
    ensure_dir,
    load_yaml_config,
    parse_float,
    parse_int,
    read_json,
    resolve_path,
    save_pt,
    write_json,
)


PAIR_COLUMNS_A = ("drug_a", "drug1", "drug_1", "med_a", "left_drug", "source_drug")
PAIR_COLUMNS_B = ("drug_b", "drug2", "drug_2", "med_b", "right_drug", "target_drug")
CANONICAL_TOKEN_COLUMNS_A = ("drug_1_token", "drug_a_token", "left_token", "source_drug_token")
CANONICAL_TOKEN_COLUMNS_B = ("drug_2_token", "drug_b_token", "right_token", "target_drug_token")
_FALLBACK_SOURCE = "fallback_zero"
_MANUAL_SMOKE_SOURCE_FORMAT = "manual_smoke_csv"
_TWOSIDES_SOURCE_FORMAT = "twosides_csv"
_CANONICAL_PAIR_SOURCE_FORMAT = "canonical_pair_csv"
_SUPPORTED_SOURCE_FORMATS = {
    _MANUAL_SMOKE_SOURCE_FORMAT,
    _TWOSIDES_SOURCE_FORMAT,
    _CANONICAL_PAIR_SOURCE_FORMAT,
}
_TWOSIDES_RXNORM_A_COLUMNS = ("drug_1_rxnorm_id", "drug_1_rxnorn_id")
_TWOSIDES_RXNORM_B_COLUMNS = ("drug_2_rxnorm_id", "drug_2_rxnorn_id")
_PROGRESS_LOG_EVERY_ROWS = 1_000_000
_UNMATCHED_COUNTER_PRUNE_EVERY_ROWS = 1_000_000
_UNMATCHED_COUNTER_PRUNE_LIMIT = 1_000
_UNMATCHED_COUNTER_REPORT_LIMIT = 200
_CANONICAL_PAIR_FIELDS = (
    "source_format",
    "source_path",
    "drug_1_rxnorm_id",
    "drug_1_concept_name",
    "drug_1_token",
    "drug_1_vocab_idx",
    "drug_2_rxnorm_id",
    "drug_2_concept_name",
    "drug_2_token",
    "drug_2_vocab_idx",
    "condition_meddra_id",
    "condition_concept_name",
    "A",
    "PRR",
    "PRR_error",
    "mean_reporting_frequency",
    "passes_statistical_filter",
    "both_drugs_matched_vocab",
)


def _ddi_output_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    processed_root = resolve_path(config["_project_root"], config["paths"]["processed_root"])
    ddi_dir = ensure_dir(processed_root / "ddi")
    ddi_cfg = dict(config.get("ddi", {}))
    canonical_pairs_path = ddi_cfg.get("canonical_pairs_path", "data/processed/ddi/drug_ddi_pairs.csv.gz")
    canonical_pairs_path = resolve_path(config["_project_root"], canonical_pairs_path).resolve()
    return (
        ddi_dir / "drug_ddi.pt",
        ddi_dir / "drug_ddi_report.json",
        canonical_pairs_path,
    )


def _part_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.part")


def _prepare_part_path(path: Path) -> Path:
    ensure_dir(path.parent)
    part_path = _part_path(path)
    if part_path.exists():
        part_path.unlink()
    return part_path


def _atomic_replace(temp_path: Path, final_path: Path) -> Path:
    ensure_dir(final_path.parent)
    if not temp_path.exists():
        if final_path.exists():
            return final_path
        raise FileNotFoundError(f"Atomic replace source is missing: {temp_path}")
    os.replace(temp_path, final_path)
    return final_path


def _current_rss_mb() -> float | None:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("VmRSS:"):
                continue
            parts = line.split()
            if len(parts) < 2:
                return None
            return float(parts[1]) / 1024.0
    except OSError:
        return None
    return None


def _find_pair_columns(fieldnames: list[str]) -> tuple[str, str] | None:
    lowered = {name.lower(): name for name in fieldnames}
    col_a = next((lowered[name] for name in PAIR_COLUMNS_A if name in lowered), None)
    col_b = next((lowered[name] for name in PAIR_COLUMNS_B if name in lowered), None)
    if col_a and col_b:
        return col_a, col_b
    if len(fieldnames) >= 2:
        return fieldnames[0], fieldnames[1]
    return None


def _find_canonical_token_columns(fieldnames: list[str]) -> tuple[str, str] | None:
    lowered = {name.lower(): name for name in fieldnames}
    col_a = next((lowered[name] for name in CANONICAL_TOKEN_COLUMNS_A if name in lowered), None)
    col_b = next((lowered[name] for name in CANONICAL_TOKEN_COLUMNS_B if name in lowered), None)
    if col_a and col_b:
        return col_a, col_b
    return None


def _source_metadata_sidecar_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}.metadata.json")


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _normalize_source_metadata(
    payload: Mapping[str, Any] | None,
    *,
    default_kind: str,
    default_display_name: str,
    default_purpose: str,
    default_pair_schema: str,
    default_research_grade: bool,
) -> dict[str, Any]:
    raw = dict(payload or {})
    return {
        "kind": str(raw.get("kind") or default_kind),
        "purpose": str(raw.get("purpose") or default_purpose),
        "research_grade": _coerce_bool(raw.get("research_grade"), default=default_research_grade),
        "pair_schema": str(raw.get("pair_schema") or default_pair_schema),
        "display_name": str(raw.get("display_name") or default_display_name),
    }


def _default_source_metadata(source_path: Path | None, *, source_format: str) -> dict[str, Any]:
    display_name = "" if source_path is None else source_path.name
    if source_format == _MANUAL_SMOKE_SOURCE_FORMAT:
        return _normalize_source_metadata(
            None,
            default_kind="manual_smoke",
            default_display_name=display_name or "Manual Smoke DDI",
            default_purpose="local wiring only",
            default_pair_schema="canonicalized_drug_token_pairs",
            default_research_grade=False,
        )
    if source_format == _TWOSIDES_SOURCE_FORMAT:
        return _normalize_source_metadata(
            None,
            default_kind="twosides_real_condition_aggregated",
            default_display_name=display_name or "TWOSIDES.csv",
            default_purpose="TwoSIDES condition-aggregated real DDI source",
            default_pair_schema="twosides_condition_rows",
            default_research_grade=False,
        )
    if source_format == _CANONICAL_PAIR_SOURCE_FORMAT:
        return _normalize_source_metadata(
            None,
            default_kind="canonical_pair_csv",
            default_display_name=display_name or "Canonical Pair CSV",
            default_purpose="canonical normalized pair source",
            default_pair_schema="canonical_pair_rows",
            default_research_grade=False,
        )
    return _normalize_source_metadata(
        None,
        default_kind="unclassified_external",
        default_display_name=display_name or "External DDI Source",
        default_purpose="source metadata missing; artifact is runnable but not research-grade by default",
        default_pair_schema="canonicalized_drug_token_pairs",
        default_research_grade=False,
    )


def _read_source_metadata(source_path: Path | None, *, source_format: str) -> dict[str, Any]:
    if source_path is None:
        return _default_source_metadata(None, source_format=source_format)
    sidecar_path = _source_metadata_sidecar_path(source_path)
    if sidecar_path.exists():
        payload = read_json(sidecar_path)
        if isinstance(payload, Mapping):
            defaults = _default_source_metadata(source_path, source_format=source_format)
            return _normalize_source_metadata(
                payload,
                default_kind=str(defaults["kind"]),
                default_display_name=str(defaults["display_name"]),
                default_purpose=str(defaults["purpose"]),
                default_pair_schema=str(defaults["pair_schema"]),
                default_research_grade=bool(defaults["research_grade"]),
            )
    return _default_source_metadata(source_path, source_format=source_format)


def _resolve_optional_path(project_root: Path, raw_value: Any) -> Path | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _resolve_ddi_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    project_root = Path(config["_project_root"]).resolve()
    ddi_cfg = dict(config.get("ddi", {}))
    legacy_source = config.get("ddi_source_path") or config.get("paths", {}).get("ddi_source_path")

    source_path = _resolve_optional_path(project_root, ddi_cfg.get("source_path"))
    if source_path is None:
        source_path = _resolve_optional_path(project_root, legacy_source)

    raw_format = str(ddi_cfg.get("source_format") or "").strip().lower()
    source_format = raw_format or _MANUAL_SMOKE_SOURCE_FORMAT
    if source_format not in _SUPPORTED_SOURCE_FORMATS:
        raise ValueError(f"Unsupported DDI source_format `{source_format}`. Expected one of {sorted(_SUPPORTED_SOURCE_FORMATS)}")

    fallback_source_path = _resolve_optional_path(project_root, ddi_cfg.get("fallback_source_path"))
    fallback_source_format = str(ddi_cfg.get("fallback_source_format") or _MANUAL_SMOKE_SOURCE_FORMAT).strip().lower()
    if fallback_source_format not in _SUPPORTED_SOURCE_FORMATS:
        raise ValueError(
            f"Unsupported fallback DDI source_format `{fallback_source_format}`. Expected one of {sorted(_SUPPORTED_SOURCE_FORMATS)}"
        )

    return {
        "source_path": source_path,
        "source_format": source_format,
        "fallback_source_path": fallback_source_path,
        "fallback_source_format": fallback_source_format,
        "min_support_a": int(ddi_cfg.get("min_support_a", 5)),
        "min_prr_ci_lower_bound": float(ddi_cfg.get("min_prr_ci_lower_bound", 1.0)),
        "canonical_pairs_configured": "canonical_pairs_path" in ddi_cfg,
    }


def _empty_effective_report(*, drug_count: int) -> dict[str, Any]:
    source_metadata = _normalize_source_metadata(
        None,
        default_kind="fallback_zero",
        default_display_name="Fallback Zero DDI",
        default_purpose="no DDI source configured; inactive artifact for explicit fallback",
        default_pair_schema="none",
        default_research_grade=False,
    )
    return {
        "active": False,
        "reason": "missing_ddi_source_path",
        "source": _FALLBACK_SOURCE,
        "requested_source": "",
        "requested_source_format": "",
        "effective_source": _FALLBACK_SOURCE,
        "effective_source_format": "fallback_zero",
        "source_format": "fallback_zero",
        "ddi_type": "fallback_zero",
        "ddi_source": _FALLBACK_SOURCE,
        "ddi_research_grade": False,
        "matched_pairs": 0,
        "nonzero_pairs": 0,
        "rows_scanned": 0,
        "rows_mapped": 0,
        "rows_retained": 0,
        "header_rows_skipped": 0,
        "invalid_numeric_rows": 0,
        "self_pair_rows_skipped": 0,
        "vocab_size": int(drug_count),
        "fallback_reason": "missing_ddi_source_path",
        "source_metadata": source_metadata,
        "unmatched_drugs_topk": [],
        "unmatched_drugs": [],
        "canonical_rows_written": 0,
    }


def _init_parse_stats(
    *,
    source_path: Path,
    source_format: str,
    source_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_path": str(source_path.resolve()),
        "source_format": source_format,
        "source_metadata": dict(source_metadata),
        "rows_scanned": 0,
        "rows_mapped": 0,
        "rows_retained": 0,
        "header_rows_skipped": 0,
        "invalid_numeric_rows": 0,
        "self_pair_rows_skipped": 0,
        "candidate_pairs": set(),
        "retained_pairs": set(),
        "unmatched_counter": Counter(),
        "canonical_rows_written": 0,
        "progress_output_path": "",
        "reason": "unknown",
    }


def _prune_unmatched_counter(unmatched_counter: Counter[str], *, keep_limit: int = _UNMATCHED_COUNTER_PRUNE_LIMIT) -> None:
    if len(unmatched_counter) <= int(keep_limit):
        return
    retained_items = dict(unmatched_counter.most_common(int(keep_limit)))
    unmatched_counter.clear()
    unmatched_counter.update(retained_items)


def _maybe_prune_unmatched_counter(stats: dict[str, Any]) -> None:
    if int(stats["rows_scanned"]) <= 0:
        return
    if int(stats["rows_scanned"]) % int(_UNMATCHED_COUNTER_PRUNE_EVERY_ROWS) != 0:
        return
    _prune_unmatched_counter(stats["unmatched_counter"])


def _log_parse_progress(stats: Mapping[str, Any]) -> None:
    rss_mb = _current_rss_mb()
    rss_text = "" if rss_mb is None else f" rss_mb={rss_mb:.1f}"
    print(
        "DDI parse progress: "
        f"source_format={stats['source_format']} "
        f"rows_scanned={int(stats['rows_scanned'])} "
        f"rows_mapped={int(stats['rows_mapped'])} "
        f"rows_retained={int(stats['rows_retained'])} "
        f"header_rows_skipped={int(stats['header_rows_skipped'])} "
        f"invalid_numeric_rows={int(stats['invalid_numeric_rows'])} "
        f"self_pair_rows_skipped={int(stats['self_pair_rows_skipped'])} "
        f"candidate_pairs={len(stats['candidate_pairs'])} "
        f"retained_pairs={len(stats['retained_pairs'])} "
        f"canonical_rows_written={int(stats['canonical_rows_written'])} "
        f"output_temp={stats.get('progress_output_path', '')}"
        f"{rss_text}",
        flush=True,
    )


def _maybe_log_parse_progress(stats: Mapping[str, Any]) -> None:
    if int(stats["rows_scanned"]) <= 0:
        return
    if int(stats["rows_scanned"]) % int(_PROGRESS_LOG_EVERY_ROWS) != 0:
        return
    _log_parse_progress(stats)


def _record_unmatched(unmatched_counter: Counter[str], *, token_a: str | None, token_b: str | None, matched_a: bool, matched_b: bool) -> None:
    if token_a and not matched_a:
        unmatched_counter[token_a] += 1
    if token_b and not matched_b:
        unmatched_counter[token_b] += 1


def _pair_key(idx_a: int, idx_b: int, *, stride: int) -> int:
    left = int(idx_a)
    right = int(idx_b)
    if left > right:
        left, right = right, left
    return (left * int(stride)) + right


def _decode_pair_key(pair_key: int, *, stride: int) -> tuple[int, int]:
    return divmod(int(pair_key), int(stride))


def _base_canonical_row(*, source_format: str, source_path: Path) -> dict[str, Any]:
    return {
        "source_format": source_format,
        "source_path": str(source_path.resolve()),
        "drug_1_rxnorm_id": "",
        "drug_1_concept_name": "",
        "drug_1_token": "",
        "drug_1_vocab_idx": "",
        "drug_2_rxnorm_id": "",
        "drug_2_concept_name": "",
        "drug_2_token": "",
        "drug_2_vocab_idx": "",
        "condition_meddra_id": "",
        "condition_concept_name": "",
        "A": "",
        "PRR": "",
        "PRR_error": "",
        "mean_reporting_frequency": "",
        "passes_statistical_filter": "",
        "both_drugs_matched_vocab": "",
    }


@contextmanager
def _canonical_pair_writer(path: Path):
    part_path = _prepare_part_path(path)
    with gzip.open(part_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CANONICAL_PAIR_FIELDS)
        writer.writeheader()

        def emit(row: Mapping[str, Any]) -> None:
            writer.writerow({field: row.get(field, "") for field in _CANONICAL_PAIR_FIELDS})

        try:
            yield emit
        except Exception:
            raise
    _atomic_replace(part_path, path)


def _emit_canonical_row(
    stats: dict[str, Any],
    canonical_row: Mapping[str, Any],
    *,
    canonical_row_writer: Any | None = None,
) -> None:
    if canonical_row_writer is not None:
        canonical_row_writer(canonical_row)
        stats["canonical_rows_written"] += 1


def _parse_manual_smoke_source(
    *,
    source_path: Path,
    drug_vocab: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    canonical_row_writer: Any | None = None,
    pair_stride: int,
    progress_output_path: str = "",
) -> dict[str, Any]:
    stats = _init_parse_stats(source_path=source_path, source_format=_MANUAL_SMOKE_SOURCE_FORMAT, source_metadata=source_metadata)
    stats["progress_output_path"] = progress_output_path
    with open_csv(source_path) as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            stats["reason"] = "missing_header"
            return stats
        pair_columns = _find_pair_columns(list(reader.fieldnames))
        if pair_columns is None:
            raise ValueError(f"Unable to determine DDI pair columns from {source_path}")
        col_a, col_b = pair_columns
        token_to_idx = drug_vocab["token_to_idx"]
        for row in reader:
            stats["rows_scanned"] += 1
            token_a = canonicalize_medication_text(row.get(col_a, ""))
            token_b = canonicalize_medication_text(row.get(col_b, ""))
            if token_a and token_b and token_a == token_b:
                stats["self_pair_rows_skipped"] += 1
                _maybe_prune_unmatched_counter(stats)
                _maybe_log_parse_progress(stats)
                continue
            if not token_a or not token_b:
                _maybe_prune_unmatched_counter(stats)
                _maybe_log_parse_progress(stats)
                continue
            idx_a = token_to_idx.get(token_a)
            idx_b = token_to_idx.get(token_b)
            matched_a = idx_a is not None
            matched_b = idx_b is not None
            if matched_a and matched_b:
                pair = _pair_key(int(idx_a), int(idx_b), stride=pair_stride)
                stats["candidate_pairs"].add(pair)
                stats["retained_pairs"].add(pair)
                stats["rows_mapped"] += 1
                stats["rows_retained"] += 1
            else:
                _record_unmatched(
                    stats["unmatched_counter"],
                    token_a=token_a,
                    token_b=token_b,
                    matched_a=matched_a,
                    matched_b=matched_b,
                )
            canonical_row = _base_canonical_row(source_format=_MANUAL_SMOKE_SOURCE_FORMAT, source_path=source_path)
            canonical_row.update(
                {
                    "drug_1_concept_name": row.get(col_a, ""),
                    "drug_1_token": token_a or "",
                    "drug_1_vocab_idx": "" if idx_a is None else int(idx_a),
                    "drug_2_concept_name": row.get(col_b, ""),
                    "drug_2_token": token_b or "",
                    "drug_2_vocab_idx": "" if idx_b is None else int(idx_b),
                    "passes_statistical_filter": True if matched_a and matched_b else False,
                    "both_drugs_matched_vocab": bool(matched_a and matched_b),
                }
            )
            _emit_canonical_row(stats, canonical_row, canonical_row_writer=canonical_row_writer)
            _maybe_prune_unmatched_counter(stats)
            _maybe_log_parse_progress(stats)

    if stats["retained_pairs"]:
        stats["reason"] = "available"
    elif stats["candidate_pairs"]:
        stats["reason"] = "no_retained_pairs"
    else:
        stats["reason"] = "no_matched_pairs"
    return stats


def _twosides_ci_lower_bound(*, prr: float | None, prr_error: float | None) -> float | None:
    if prr is None or prr_error is None:
        return None
    if not math.isfinite(prr) or not math.isfinite(prr_error):
        return None
    return float(prr - (1.96 * prr_error))


def _safe_parse_int(raw_value: Any) -> int | None:
    try:
        return parse_int(raw_value)
    except (TypeError, ValueError):
        return None


def _safe_parse_float(raw_value: Any) -> float | None:
    try:
        return parse_float(raw_value)
    except (TypeError, ValueError):
        return None


def _looks_like_twosides_header_row(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("drug_1_concept_name", "")).strip().lower() == "drug_1_concept_name"
        or str(row.get("drug_2_concept_name", "")).strip().lower() == "drug_2_concept_name"
        or (
            str(row.get("A", "")).strip().lower() == "a"
            and str(row.get("PRR", "")).strip().lower() == "prr"
        )
    )


def _parse_twosides_source(
    *,
    source_path: Path,
    drug_vocab: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    min_support_a: int,
    min_prr_ci_lower_bound: float,
    canonical_row_writer: Any | None = None,
    pair_stride: int,
    progress_output_path: str = "",
) -> dict[str, Any]:
    stats = _init_parse_stats(source_path=source_path, source_format=_TWOSIDES_SOURCE_FORMAT, source_metadata=source_metadata)
    stats["progress_output_path"] = progress_output_path
    token_to_idx = drug_vocab["token_to_idx"]

    with open_csv(source_path) as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            stats["reason"] = "missing_header"
            return stats
        required_columns = {
            "drug_1_concept_name",
            "drug_2_concept_name",
            "condition_meddra_id",
            "condition_concept_name",
            "A",
            "PRR",
            "PRR_error",
            "mean_reporting_frequency",
        }
        missing = sorted(column for column in required_columns if column not in reader.fieldnames)
        if missing:
            raise ValueError(f"TwoSIDES source is missing required columns {missing} in {source_path}")

        for row in reader:
            stats["rows_scanned"] += 1
            if _looks_like_twosides_header_row(row):
                stats["header_rows_skipped"] += 1
                _maybe_prune_unmatched_counter(stats)
                _maybe_log_parse_progress(stats)
                continue
            token_a = canonicalize_medication_text(row.get("drug_1_concept_name", ""))
            token_b = canonicalize_medication_text(row.get("drug_2_concept_name", ""))
            if token_a and token_b and token_a == token_b:
                stats["self_pair_rows_skipped"] += 1
                _maybe_prune_unmatched_counter(stats)
                _maybe_log_parse_progress(stats)
                continue
            if not token_a or not token_b:
                _maybe_prune_unmatched_counter(stats)
                _maybe_log_parse_progress(stats)
                continue

            idx_a = token_to_idx.get(token_a)
            idx_b = token_to_idx.get(token_b)
            matched_a = idx_a is not None
            matched_b = idx_b is not None
            pair_matched = matched_a and matched_b
            raw_support_a = row.get("A")
            raw_prr = row.get("PRR")
            raw_prr_error = row.get("PRR_error")
            support_a = _safe_parse_int(raw_support_a)
            prr = _safe_parse_float(raw_prr)
            prr_error = _safe_parse_float(raw_prr_error)
            invalid_numeric_row = False
            if str(raw_support_a or "").strip() and support_a is None:
                invalid_numeric_row = True
            if str(raw_prr or "").strip() and prr is None:
                invalid_numeric_row = True
            if str(raw_prr_error or "").strip() and prr_error is None:
                invalid_numeric_row = True
            if invalid_numeric_row:
                stats["invalid_numeric_rows"] += 1
            ci_lower_bound = _twosides_ci_lower_bound(prr=prr, prr_error=prr_error)
            passes_filter = (
                pair_matched
                and support_a is not None
                and int(support_a) >= int(min_support_a)
                and ci_lower_bound is not None
                and float(ci_lower_bound) > float(min_prr_ci_lower_bound)
            )

            if pair_matched:
                pair = _pair_key(int(idx_a), int(idx_b), stride=pair_stride)
                stats["candidate_pairs"].add(pair)
                stats["rows_mapped"] += 1
                if passes_filter:
                    stats["retained_pairs"].add(pair)
                    stats["rows_retained"] += 1
            else:
                _record_unmatched(
                    stats["unmatched_counter"],
                    token_a=token_a,
                    token_b=token_b,
                    matched_a=matched_a,
                    matched_b=matched_b,
                )

            canonical_row = _base_canonical_row(source_format=_TWOSIDES_SOURCE_FORMAT, source_path=source_path)
            canonical_row.update(
                {
                    "drug_1_rxnorm_id": next((row.get(column, "") for column in _TWOSIDES_RXNORM_A_COLUMNS if row.get(column, "")), ""),
                    "drug_1_concept_name": row.get("drug_1_concept_name", ""),
                    "drug_1_token": token_a or "",
                    "drug_1_vocab_idx": "" if idx_a is None else int(idx_a),
                    "drug_2_rxnorm_id": next((row.get(column, "") for column in _TWOSIDES_RXNORM_B_COLUMNS if row.get(column, "")), ""),
                    "drug_2_concept_name": row.get("drug_2_concept_name", ""),
                    "drug_2_token": token_b or "",
                    "drug_2_vocab_idx": "" if idx_b is None else int(idx_b),
                    "condition_meddra_id": row.get("condition_meddra_id", ""),
                    "condition_concept_name": row.get("condition_concept_name", ""),
                    "A": "" if support_a is None else int(support_a),
                    "PRR": "" if prr is None else float(prr),
                    "PRR_error": "" if prr_error is None else float(prr_error),
                    "mean_reporting_frequency": row.get("mean_reporting_frequency", ""),
                    "passes_statistical_filter": bool(passes_filter),
                    "both_drugs_matched_vocab": bool(pair_matched),
                }
            )
            _emit_canonical_row(stats, canonical_row, canonical_row_writer=canonical_row_writer)
            _maybe_prune_unmatched_counter(stats)
            _maybe_log_parse_progress(stats)

    if stats["retained_pairs"]:
        stats["reason"] = "available"
    elif stats["candidate_pairs"]:
        stats["reason"] = "no_retained_pairs"
    else:
        stats["reason"] = "no_matched_pairs"
    return stats


def _canonical_row_passes_filter(row: Mapping[str, str]) -> bool:
    raw_flag = row.get("passes_statistical_filter")
    if raw_flag is not None and str(raw_flag).strip() != "":
        return _coerce_bool(raw_flag, default=False)
    return False


def _parse_canonical_pair_source(
    *,
    source_path: Path,
    drug_vocab: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    canonical_row_writer: Any | None = None,
    pair_stride: int,
    progress_output_path: str = "",
) -> dict[str, Any]:
    stats = _init_parse_stats(source_path=source_path, source_format=_CANONICAL_PAIR_SOURCE_FORMAT, source_metadata=source_metadata)
    stats["progress_output_path"] = progress_output_path
    token_to_idx = drug_vocab["token_to_idx"]

    with open_csv(source_path) as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            stats["reason"] = "missing_header"
            return stats
        canonical_token_columns = _find_canonical_token_columns(list(reader.fieldnames))
        pair_columns = _find_pair_columns(list(reader.fieldnames))
        if canonical_token_columns is None and pair_columns is None:
            raise ValueError(f"Canonical pair source {source_path} must contain token columns or generic pair columns.")

        for row in reader:
            stats["rows_scanned"] += 1
            if canonical_token_columns is not None:
                token_a = str(row.get(canonical_token_columns[0], "")).strip() or None
                token_b = str(row.get(canonical_token_columns[1], "")).strip() or None
            else:
                col_a, col_b = pair_columns
                token_a = canonicalize_medication_text(row.get(col_a, ""))
                token_b = canonicalize_medication_text(row.get(col_b, ""))
            if token_a and token_b and token_a == token_b:
                stats["self_pair_rows_skipped"] += 1
                _maybe_prune_unmatched_counter(stats)
                _maybe_log_parse_progress(stats)
                continue
            if not token_a or not token_b:
                _maybe_prune_unmatched_counter(stats)
                _maybe_log_parse_progress(stats)
                continue

            idx_a = token_to_idx.get(token_a)
            idx_b = token_to_idx.get(token_b)
            matched_a = idx_a is not None
            matched_b = idx_b is not None
            pair_matched = matched_a and matched_b
            passes_filter = pair_matched and _canonical_row_passes_filter(row)
            if pair_matched:
                pair = _pair_key(int(idx_a), int(idx_b), stride=pair_stride)
                stats["candidate_pairs"].add(pair)
                stats["rows_mapped"] += 1
                if passes_filter:
                    stats["retained_pairs"].add(pair)
                    stats["rows_retained"] += 1
            else:
                _record_unmatched(
                    stats["unmatched_counter"],
                    token_a=token_a,
                    token_b=token_b,
                    matched_a=matched_a,
                    matched_b=matched_b,
                )

            canonical_row = _base_canonical_row(source_format=_CANONICAL_PAIR_SOURCE_FORMAT, source_path=source_path)
            canonical_row.update({field: row.get(field, canonical_row[field]) for field in _CANONICAL_PAIR_FIELDS if field in row})
            canonical_row["drug_1_token"] = token_a or ""
            canonical_row["drug_2_token"] = token_b or ""
            canonical_row["drug_1_vocab_idx"] = "" if idx_a is None else int(idx_a)
            canonical_row["drug_2_vocab_idx"] = "" if idx_b is None else int(idx_b)
            canonical_row["passes_statistical_filter"] = bool(passes_filter)
            canonical_row["both_drugs_matched_vocab"] = bool(pair_matched)
            _emit_canonical_row(stats, canonical_row, canonical_row_writer=canonical_row_writer)
            _maybe_prune_unmatched_counter(stats)
            _maybe_log_parse_progress(stats)

    if stats["retained_pairs"]:
        stats["reason"] = "available"
    elif stats["candidate_pairs"]:
        stats["reason"] = "no_retained_pairs"
    else:
        stats["reason"] = "no_matched_pairs"
    return stats


def _build_matrix_from_pairs(*, retained_pairs: set[int], drug_count: int) -> list[list[int]]:
    matrix = [[0 for _ in range(drug_count)] for _ in range(drug_count)]
    for pair_key in retained_pairs:
        idx_a, idx_b = _decode_pair_key(int(pair_key), stride=drug_count)
        matrix[idx_a][idx_b] = 1
        matrix[idx_b][idx_a] = 1
    return matrix


def _build_source_result(
    *,
    source_path: Path,
    source_format: str,
    drug_vocab: Mapping[str, Any],
    min_support_a: int,
    min_prr_ci_lower_bound: float,
    canonical_pairs_path: Path | None = None,
) -> dict[str, Any]:
    if not source_path.exists():
        raise FileNotFoundError(f"Configured DDI source path does not exist: {source_path}")
    source_metadata = _read_source_metadata(source_path, source_format=source_format)
    progress_output_path = "" if canonical_pairs_path is None else str(_part_path(canonical_pairs_path))
    writer_context = _canonical_pair_writer(canonical_pairs_path) if canonical_pairs_path is not None else nullcontext(None)
    with writer_context as canonical_row_writer:
        if source_format == _MANUAL_SMOKE_SOURCE_FORMAT:
            stats = _parse_manual_smoke_source(
                source_path=source_path,
                drug_vocab=drug_vocab,
                source_metadata=source_metadata,
                canonical_row_writer=canonical_row_writer,
                pair_stride=len(drug_vocab["idx_to_token"]),
                progress_output_path=progress_output_path,
            )
        elif source_format == _TWOSIDES_SOURCE_FORMAT:
            stats = _parse_twosides_source(
                source_path=source_path,
                drug_vocab=drug_vocab,
                source_metadata=source_metadata,
                min_support_a=min_support_a,
                min_prr_ci_lower_bound=min_prr_ci_lower_bound,
                canonical_row_writer=canonical_row_writer,
                pair_stride=len(drug_vocab["idx_to_token"]),
                progress_output_path=progress_output_path,
            )
        elif source_format == _CANONICAL_PAIR_SOURCE_FORMAT:
            stats = _parse_canonical_pair_source(
                source_path=source_path,
                drug_vocab=drug_vocab,
                source_metadata=source_metadata,
                canonical_row_writer=canonical_row_writer,
                pair_stride=len(drug_vocab["idx_to_token"]),
                progress_output_path=progress_output_path,
            )
        else:
            raise ValueError(f"Unsupported source_format `{source_format}`.")

    retained_pairs = set(stats["retained_pairs"])
    candidate_pairs = set(stats["candidate_pairs"])
    source_metadata = dict(stats["source_metadata"])
    is_research_grade = bool(source_format == _TWOSIDES_SOURCE_FORMAT and retained_pairs)
    source_metadata["research_grade"] = is_research_grade
    matrix = _build_matrix_from_pairs(retained_pairs=retained_pairs, drug_count=len(drug_vocab["idx_to_token"]))
    ddi_type = str(source_metadata.get("kind") or "unknown")
    return {
        "matrix": matrix,
        "active": bool(retained_pairs),
        "reason": str(stats["reason"]),
        "source": str(source_path.resolve()),
        "source_path": str(source_path.resolve()),
        "source_format": source_format,
        "ddi_type": ddi_type,
        "ddi_source": str(source_path.resolve()),
        "ddi_research_grade": is_research_grade,
        "matched_pairs": int(len(candidate_pairs)),
        "nonzero_pairs": int(len(retained_pairs)),
        "rows_scanned": int(stats["rows_scanned"]),
        "rows_mapped": int(stats["rows_mapped"]),
        "rows_retained": int(stats["rows_retained"]),
        "header_rows_skipped": int(stats["header_rows_skipped"]),
        "invalid_numeric_rows": int(stats["invalid_numeric_rows"]),
        "self_pair_rows_skipped": int(stats["self_pair_rows_skipped"]),
        "source_metadata": source_metadata,
        "unmatched_drugs_topk": [token for token, _ in stats["unmatched_counter"].most_common(_UNMATCHED_COUNTER_REPORT_LIMIT)],
        "unmatched_drugs": [token for token, _ in stats["unmatched_counter"].most_common(_UNMATCHED_COUNTER_REPORT_LIMIT)],
        "canonical_rows_written": int(stats["canonical_rows_written"]),
    }


def _write_pt_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    part_path = _prepare_part_path(path)
    save_pt(part_path, payload)
    return _atomic_replace(part_path, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    part_path = _prepare_part_path(path)
    write_json(part_path, payload)
    return _atomic_replace(part_path, path)


def build_ddi_matrix(config_path: str | Path) -> Path:
    config = load_yaml_config(config_path)
    vocab_bundle = load_vocab_bundle(config)
    drug_vocab = vocab_bundle["drug"]
    drug_count = len(drug_vocab["idx_to_token"])
    matrix_path, report_path, canonical_pairs_path = _ddi_output_paths(config)
    settings = _resolve_ddi_settings(config)
    canonical_pairs_temp_path = _part_path(canonical_pairs_path) if settings["canonical_pairs_configured"] else None

    print(
        "Starting DDI build: "
        f"requested_source={settings['source_path']} "
        f"source_format={settings['source_format']} "
        f"fallback_source={settings['fallback_source_path']} "
        f"fallback_source_format={settings['fallback_source_format']} "
        f"canonical_pairs_temp={canonical_pairs_temp_path}",
        flush=True,
    )

    report = _empty_effective_report(drug_count=drug_count)
    report["vocab_size"] = int(drug_count)

    primary_result: dict[str, Any] | None = None
    source_path = settings["source_path"]
    if source_path is not None:
        primary_result = _build_source_result(
            source_path=source_path,
            source_format=str(settings["source_format"]),
            drug_vocab=drug_vocab,
            min_support_a=int(settings["min_support_a"]),
            min_prr_ci_lower_bound=float(settings["min_prr_ci_lower_bound"]),
            canonical_pairs_path=canonical_pairs_path if settings["canonical_pairs_configured"] else None,
        )
    elif settings["canonical_pairs_configured"]:
        warnings.warn(
            "DDI canonical_pairs_path was configured but ddi.source_path is empty; no canonical pair file was written.",
            stacklevel=2,
        )

    effective_result = primary_result
    fallback_reason = ""
    fallback_source_path = settings["fallback_source_path"]
    if primary_result is None:
        fallback_reason = "missing_ddi_source_path"
    elif not bool(primary_result["active"]):
        fallback_reason = str(primary_result["reason"])

    should_use_fallback = bool(fallback_source_path is not None and fallback_reason)
    if should_use_fallback:
        if fallback_source_path.exists():
            fallback_result = _build_source_result(
                source_path=fallback_source_path,
                source_format=str(settings["fallback_source_format"]),
                drug_vocab=drug_vocab,
                min_support_a=int(settings["min_support_a"]),
                min_prr_ci_lower_bound=float(settings["min_prr_ci_lower_bound"]),
                canonical_pairs_path=None,
            )
            if bool(fallback_result["active"]):
                effective_result = fallback_result
            else:
                effective_result = fallback_result
                if primary_result is None:
                    fallback_reason = f"{fallback_reason}; fallback_result={fallback_result['reason']}"
                else:
                    fallback_reason = (
                        f"{fallback_reason}; fallback_result={fallback_result['reason']}"
                    )
        else:
            if primary_result is None:
                fallback_reason = (
                    f"{fallback_reason}; fallback_source_missing={fallback_source_path.resolve()}"
                )
            else:
                fallback_reason = (
                    f"{fallback_reason}; fallback_source_missing={fallback_source_path.resolve()}"
                )

    if effective_result is None:
        matrix = [[0 for _ in range(drug_count)] for _ in range(drug_count)]
        effective_result = {
            "matrix": matrix,
            "active": False,
            "reason": "missing_ddi_source_path",
            "source": _FALLBACK_SOURCE,
            "source_path": _FALLBACK_SOURCE,
            "source_format": "fallback_zero",
            "ddi_type": "fallback_zero",
            "ddi_source": _FALLBACK_SOURCE,
            "ddi_research_grade": False,
            "matched_pairs": 0,
            "nonzero_pairs": 0,
            "rows_scanned": 0,
            "rows_mapped": 0,
            "rows_retained": 0,
            "header_rows_skipped": 0,
            "invalid_numeric_rows": 0,
            "self_pair_rows_skipped": 0,
            "source_metadata": report["source_metadata"],
            "unmatched_drugs_topk": [],
            "unmatched_drugs": [],
            "canonical_rows_written": 0,
        }

    requested_source = "" if source_path is None else str(source_path.resolve())
    requested_source_format = "" if source_path is None else str(settings["source_format"])
    effective_source = str(effective_result["source"])
    effective_source_format = str(effective_result["source_format"])
    if effective_result["active"]:
        report.update(
            {
                "active": True,
                "reason": "available",
                "source": effective_source,
                "requested_source": requested_source,
                "requested_source_format": requested_source_format,
                "effective_source": effective_source,
                "effective_source_format": effective_source_format,
                "source_format": effective_source_format,
                "ddi_type": str(effective_result["ddi_type"]),
                "ddi_source": effective_source,
                "ddi_research_grade": bool(effective_result["ddi_research_grade"]),
                "matched_pairs": int(effective_result["matched_pairs"]),
                "nonzero_pairs": int(effective_result["nonzero_pairs"]),
                "rows_scanned": int(effective_result["rows_scanned"]),
                "rows_mapped": int(effective_result["rows_mapped"]),
                "rows_retained": int(effective_result["rows_retained"]),
                "header_rows_skipped": int(effective_result.get("header_rows_skipped", 0)),
                "invalid_numeric_rows": int(effective_result.get("invalid_numeric_rows", 0)),
                "self_pair_rows_skipped": int(effective_result.get("self_pair_rows_skipped", 0)),
                "source_metadata": dict(effective_result["source_metadata"]),
                "fallback_reason": "" if effective_source == requested_source or not fallback_reason else fallback_reason,
                "unmatched_drugs_topk": list(effective_result["unmatched_drugs_topk"]),
                "unmatched_drugs": list(effective_result["unmatched_drugs"]),
                "canonical_rows_written": int(effective_result.get("canonical_rows_written", 0)),
            }
        )
    else:
        reason = str(fallback_reason or effective_result["reason"] or "no_matched_pairs")
        report.update(
            {
                "active": False,
                "reason": reason,
                "source": effective_source,
                "requested_source": requested_source,
                "requested_source_format": requested_source_format,
                "effective_source": effective_source,
                "effective_source_format": effective_source_format,
                "source_format": effective_source_format,
                "ddi_type": str(effective_result["ddi_type"]),
                "ddi_source": effective_source,
                "ddi_research_grade": False,
                "matched_pairs": int(effective_result["matched_pairs"]),
                "nonzero_pairs": int(effective_result["nonzero_pairs"]),
                "rows_scanned": int(effective_result["rows_scanned"]),
                "rows_mapped": int(effective_result["rows_mapped"]),
                "rows_retained": int(effective_result["rows_retained"]),
                "header_rows_skipped": int(effective_result.get("header_rows_skipped", 0)),
                "invalid_numeric_rows": int(effective_result.get("invalid_numeric_rows", 0)),
                "self_pair_rows_skipped": int(effective_result.get("self_pair_rows_skipped", 0)),
                "source_metadata": dict(effective_result["source_metadata"]),
                "fallback_reason": reason if fallback_reason else "",
                "unmatched_drugs_topk": list(effective_result["unmatched_drugs_topk"]),
                "unmatched_drugs": list(effective_result["unmatched_drugs"]),
                "canonical_rows_written": int(effective_result.get("canonical_rows_written", 0)),
            }
        )
        warnings.warn(
            "DDI source did not yield an active artifact; writing an inactive fallback_zero DDI artifact.",
            stacklevel=2,
        )

    effective_matrix = effective_result["matrix"]
    _write_pt_atomic(
        matrix_path,
        {
            "matrix": effective_matrix,
            "active": bool(report["active"]),
            "reason": str(report["reason"]),
            "source": report["source"],
            "requested_source": str(report["requested_source"]),
            "requested_source_format": str(report["requested_source_format"]),
            "effective_source": str(report["effective_source"]),
            "effective_source_format": str(report["effective_source_format"]),
            "source_format": str(report["source_format"]),
            "ddi_type": str(report["ddi_type"]),
            "ddi_source": str(report["ddi_source"]),
            "ddi_research_grade": bool(report["ddi_research_grade"]),
            "matched_pairs": int(report["matched_pairs"]),
            "nonzero_pairs": int(report["nonzero_pairs"]),
            "rows_scanned": int(report["rows_scanned"]),
            "rows_mapped": int(report["rows_mapped"]),
            "rows_retained": int(report["rows_retained"]),
            "header_rows_skipped": int(report.get("header_rows_skipped", 0)),
            "invalid_numeric_rows": int(report.get("invalid_numeric_rows", 0)),
            "self_pair_rows_skipped": int(report.get("self_pair_rows_skipped", 0)),
            "vocab_size": int(drug_count),
            "pad_idx": int(drug_vocab["pad_idx"]),
            "unk_idx": int(drug_vocab["unk_idx"]),
            "fallback_reason": str(report["fallback_reason"]),
            "source_metadata": dict(report["source_metadata"]),
            "unmatched_drugs_topk": list(report["unmatched_drugs_topk"]),
            "unmatched_drugs": list(report["unmatched_drugs"]),
            "canonical_rows_written": int(report.get("canonical_rows_written", 0)),
        },
    )
    _write_json_atomic(report_path, report)
    print(
        "Completed DDI build: "
        f"active={bool(report['active'])} "
        f"ddi_type={report['ddi_type']} "
        f"ddi_research_grade={bool(report['ddi_research_grade'])} "
        f"matched_pairs={int(report['matched_pairs'])} "
        f"nonzero_pairs={int(report['nonzero_pairs'])} "
        f"effective_source={report['effective_source']} "
        f"fallback_reason={report['fallback_reason']}",
        flush=True,
    )
    return matrix_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DDI matrix aligned with the drug vocabulary.")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    args = parser.parse_args()
    build_ddi_matrix(args.config)


if __name__ == "__main__":
    main()
