from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.data.build_vocab import load_vocab_bundle
from src.data.load_mimic import open_csv
from src.features.medication_history import canonicalize_medication_text
from src.utils.io import ensure_dir, load_yaml_config, resolve_path, save_pt, write_json


PAIR_COLUMNS_A = ("drug_a", "drug1", "drug_1", "med_a", "left_drug", "source_drug")
PAIR_COLUMNS_B = ("drug_b", "drug2", "drug_2", "med_b", "right_drug", "target_drug")


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


def build_ddi_matrix(config_path: str | Path) -> Path:
    config = load_yaml_config(config_path)
    vocab_bundle = load_vocab_bundle(config)
    drug_vocab = vocab_bundle["drug"]
    drug_count = len(drug_vocab["idx_to_token"])
    matrix = [[0 for _ in range(drug_count)] for _ in range(drug_count)]
    report = {
        "source": "fallback_zero",
        "matched_pairs": 0,
        "unmatched_drugs": [],
    }

    ddi_source_path = str(config.get("ddi_source_path") or config.get("paths", {}).get("ddi_source_path", "")).strip()
    if ddi_source_path:
        source_path = Path(ddi_source_path)
        if not source_path.is_absolute():
            source_path = Path(config["_project_root"]) / source_path
        if source_path.exists():
            unmatched: set[str] = set()
            matched_pairs = 0
            with open_csv(source_path) as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames:
                    pair_columns = _find_pair_columns(reader.fieldnames)
                    if pair_columns:
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
                        report["source"] = str(source_path)
                        report["matched_pairs"] = matched_pairs
                        report["unmatched_drugs"] = sorted(unmatched)[:200]

    matrix_path, report_path = _ddi_output_paths(config)
    save_pt(
        matrix_path,
        {
            "matrix": matrix,
            "source": report["source"],
            "vocab_size": drug_count,
            "pad_idx": drug_vocab["pad_idx"],
            "unk_idx": drug_vocab["unk_idx"],
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
