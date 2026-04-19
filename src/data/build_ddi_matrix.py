from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Any, Mapping

from src.data.build_vocab import load_vocab_bundle
from src.data.load_mimic import open_csv
from src.features.medication_history import canonicalize_medication_text
from src.utils.io import ensure_dir, load_yaml_config, read_json, resolve_path, save_pt, write_json


PAIR_COLUMNS_A = ("drug_a", "drug1", "drug_1", "med_a", "left_drug", "source_drug")
PAIR_COLUMNS_B = ("drug_b", "drug2", "drug_2", "med_b", "right_drug", "target_drug")

_FALLBACK_SOURCE = "fallback_zero"


def _ddi_output_paths(config: dict) -> tuple[Path, Path]:
    processed_root = resolve_path(config["_project_root"], config["paths"]["processed_root"])
    ddi_dir = ensure_dir(processed_root / "ddi")
    return ddi_dir / "drug_ddi.pt", ddi_dir / "drug_ddi_report.json"


def _find_pair_columns(fieldnames: list[str]) -> tuple[str, str] | None:
    lowered = {name.lower(): name for name in fieldnames}
    col_a = next((lowered[name] for name in PAIR_COLUMNS_A if name in lowered), None)
    col_b = next((lowered[name] for name in PAIR_COLUMNS_B if name in lowered), None)
    if col_a and col_b:
        return col_a, col_b
    if len(fieldnames) >= 2:
        return fieldnames[0], fieldnames[1]
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


def _read_source_metadata(source_path: Path) -> dict[str, Any]:
    sidecar_path = _source_metadata_sidecar_path(source_path)
    if sidecar_path.exists():
        payload = read_json(sidecar_path)
        if isinstance(payload, Mapping):
            return _normalize_source_metadata(
                payload,
                default_kind="unclassified_external",
                default_display_name=source_path.name,
                default_purpose="metadata sidecar present, but purpose was not specified",
                default_pair_schema="canonicalized_drug_token_pairs",
                default_research_grade=False,
            )
    return _normalize_source_metadata(
        None,
        default_kind="unclassified_external",
        default_display_name=source_path.name,
        default_purpose="source metadata missing; artifact is runnable but not research-grade by default",
        default_pair_schema="canonicalized_drug_token_pairs",
        default_research_grade=False,
    )


def build_ddi_matrix(config_path: str | Path) -> Path:
    config = load_yaml_config(config_path)
    vocab_bundle = load_vocab_bundle(config)
    drug_vocab = vocab_bundle["drug"]
    drug_count = len(drug_vocab["idx_to_token"])
    matrix = [[0 for _ in range(drug_count)] for _ in range(drug_count)]
    report: dict[str, object] = {
        "active": False,
        "reason": "missing_ddi_source_path",
        "source": _FALLBACK_SOURCE,
        "matched_pairs": 0,
        "nonzero_pairs": 0,
        "vocab_size": int(drug_count),
        "source_metadata": _normalize_source_metadata(
            None,
            default_kind="fallback_zero",
            default_display_name="Fallback Zero DDI",
            default_purpose="no DDI source configured; inactive artifact for explicit fallback",
            default_pair_schema="none",
            default_research_grade=False,
        ),
        "unmatched_drugs": [],
    }

    ddi_source_path = str(config.get("ddi_source_path") or config.get("paths", {}).get("ddi_source_path", "")).strip()
    if not ddi_source_path:
        warnings.warn(
            "DDI source path is empty; writing an inactive fallback_zero DDI artifact.",
            stacklevel=2,
        )
    else:
        source_path = Path(ddi_source_path)
        if not source_path.is_absolute():
            source_path = Path(config["_project_root"]) / source_path
        if not source_path.exists():
            raise FileNotFoundError(f"Configured DDI source path does not exist: {source_path}")
        source_metadata = _read_source_metadata(source_path)

        unmatched: set[str] = set()
        matched_pairs = 0
        with open_csv(source_path) as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                pair_columns = _find_pair_columns(reader.fieldnames)
                if pair_columns is None:
                    raise ValueError(f"Unable to determine DDI pair columns from {source_path}")
                col_a, col_b = pair_columns
                for row in reader:
                    token_a = canonicalize_medication_text(row.get(col_a, ""))
                    token_b = canonicalize_medication_text(row.get(col_b, ""))
                    if not token_a or not token_b:
                        continue
                    idx_a = drug_vocab["token_to_idx"].get(token_a)
                    idx_b = drug_vocab["token_to_idx"].get(token_b)
                    if idx_a is None or idx_b is None:
                        if idx_a is None:
                            unmatched.add(token_a)
                        if idx_b is None:
                            unmatched.add(token_b)
                        continue
                    matrix[idx_a][idx_b] = 1
                    matrix[idx_b][idx_a] = 1
                    matched_pairs += 1

        nonzero_pairs = sum(matrix[i][j] for i in range(drug_count) for j in range(i + 1, drug_count))
        if nonzero_pairs > 0 and matched_pairs > 0:
            report["active"] = True
            report["reason"] = "available"
            report["source"] = str(source_path.resolve())
        else:
            warnings.warn(
                "DDI source was provided but produced no matched drug pairs; writing an inactive DDI artifact.",
                stacklevel=2,
            )
            report["active"] = False
            report["reason"] = "no_matched_pairs"
            report["source"] = str(source_path.resolve())

        report["matched_pairs"] = int(matched_pairs)
        report["nonzero_pairs"] = int(nonzero_pairs)
        report["source_metadata"] = source_metadata
        report["unmatched_drugs"] = sorted(unmatched)[:200]

    matrix_path, report_path = _ddi_output_paths(config)
    save_pt(
        matrix_path,
        {
            "matrix": matrix,
            "active": bool(report["active"]),
            "reason": str(report["reason"]),
            "source": report["source"],
            "matched_pairs": int(report["matched_pairs"]),
            "nonzero_pairs": int(report["nonzero_pairs"]),
            "vocab_size": int(drug_count),
            "pad_idx": int(drug_vocab["pad_idx"]),
            "unk_idx": int(drug_vocab["unk_idx"]),
            "source_metadata": dict(report["source_metadata"]),
        },
    )
    write_json(report_path, report)
    return matrix_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DDI matrix aligned with the drug vocabulary.")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    args = parser.parse_args()
    build_ddi_matrix(args.config)


if __name__ == "__main__":
    main()
