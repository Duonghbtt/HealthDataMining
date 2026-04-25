from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from src.data.build_vocab import load_vocab_bundle
from src.data.drugbank import (
    DRUGBANK_DDI_TYPE,
    DRUGBANK_SOURCE_FORMAT,
    drugbank_source_metadata,
    iter_drugbank_records,
    resolve_drugbank_paths,
    resolve_record_vocab_match,
)
from src.utils.io import ensure_dir, load_yaml_config, save_pt, write_json


_EXAMPLE_LIMIT = 25


def _part_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.part")


def _replace_path(temp_path: Path, final_path: Path) -> Path:
    ensure_dir(final_path.parent)
    os.replace(temp_path, final_path)
    return final_path


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    temp_path = _part_path(path)
    write_json(temp_path, payload)
    return _replace_path(temp_path, path)


def _write_pt_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    temp_path = _part_path(path)
    save_pt(temp_path, payload)
    return _replace_path(temp_path, path)


def _write_empty_jsonl_gz(path: Path) -> Path:
    temp_path = _part_path(path)
    ensure_dir(temp_path.parent)
    with gzip.open(temp_path, "wt", encoding="utf-8"):
        pass
    return _replace_path(temp_path, path)


def _append_example(target: list[dict[str, Any]], example: dict[str, Any]) -> None:
    if len(target) >= _EXAMPLE_LIMIT:
        return
    target.append(example)


def _fraction(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _candidate_vocab_size(token_to_idx: Mapping[str, int]) -> int:
    return sum(1 for token in token_to_idx if str(token) not in {"PAD", "UNK"})


def _zero_matrix(size: int) -> list[list[int]]:
    return [[0 for _ in range(int(size))] for _ in range(int(size))]


def _build_matrix(size: int, pair_keys: set[tuple[int, int]]) -> list[list[int]]:
    matrix = _zero_matrix(size)
    for idx_a, idx_b in pair_keys:
        matrix[idx_a][idx_b] = 1
        matrix[idx_b][idx_a] = 1
    return matrix


def _inactive_report(
    *,
    source_path: Path,
    ddi_pairs_path: Path,
    ddi_matrix_path: Path,
    ddi_report_path: Path,
    drug_vocab: Mapping[str, Any],
) -> dict[str, Any]:
    drug_count = len(drug_vocab["idx_to_token"])
    return {
        "active": False,
        "reason": "missing_source_path",
        "source": str(source_path),
        "requested_source": str(source_path),
        "requested_source_format": DRUGBANK_SOURCE_FORMAT,
        "effective_source": str(source_path),
        "effective_source_format": DRUGBANK_SOURCE_FORMAT,
        "source_format": DRUGBANK_SOURCE_FORMAT,
        "source_type": "drugbank",
        "ddi_type": DRUGBANK_DDI_TYPE,
        "ddi_source": str(source_path),
        "ddi_research_grade": False,
        "ddi_pairs_path": str(ddi_pairs_path),
        "ddi_matrix_path": str(ddi_matrix_path),
        "ddi_report_path": str(ddi_report_path),
        "vocab_size": int(drug_count),
        "drugbank_drugs_parsed": 0,
        "mapped_drugbank_records": 0,
        "mapped_vocab_drugs": 0,
        "raw_interaction_edges": 0,
        "mapped_interaction_edges": 0,
        "mapped_pairs": 0,
        "matched_pairs": 0,
        "nonzero_pairs": 0,
        "dropped_interaction_edges": 0,
        "self_interaction_edges_skipped": 0,
        "match_source_counts": {},
        "collision_counts": {
            "ambiguous_record_matches": 0,
            "vocab_tokens_with_multiple_drugbank_records": 0,
            "extra_mapped_records_beyond_first": 0,
        },
        "coverage": {
            "mapped_record_fraction": 0.0,
            "mapped_vocab_fraction": 0.0,
            "interaction_edge_mapping_fraction": 0.0,
        },
        "unmatched_examples": [],
        "ambiguous_examples": [],
        "source_metadata": drugbank_source_metadata(source_path),
    }


def build_drugbank_ddi_matrix(config_path: str | Path) -> Path:
    config = load_yaml_config(config_path)
    paths = resolve_drugbank_paths(config)
    source_path = paths["source_path"]
    ddi_pairs_path = paths["ddi_pairs_path"]
    ddi_matrix_path = paths["ddi_matrix_path"]
    ddi_report_path = paths["ddi_report_path"]

    drug_vocab = load_vocab_bundle(config)["drug"]
    token_to_idx = dict(drug_vocab["token_to_idx"])
    drug_count = len(drug_vocab["idx_to_token"])

    if not source_path.exists():
        _write_empty_jsonl_gz(ddi_pairs_path)
        inactive_report = _inactive_report(
            source_path=source_path,
            ddi_pairs_path=ddi_pairs_path,
            ddi_matrix_path=ddi_matrix_path,
            ddi_report_path=ddi_report_path,
            drug_vocab=drug_vocab,
        )
        _write_pt_atomic(
            ddi_matrix_path,
            {
                "matrix": _zero_matrix(drug_count),
                "active": False,
                "reason": inactive_report["reason"],
                "source": inactive_report["source"],
                "requested_source": inactive_report["requested_source"],
                "requested_source_format": inactive_report["requested_source_format"],
                "effective_source": inactive_report["effective_source"],
                "effective_source_format": inactive_report["effective_source_format"],
                "source_format": inactive_report["source_format"],
                "ddi_type": inactive_report["ddi_type"],
                "ddi_source": inactive_report["ddi_source"],
                "ddi_research_grade": inactive_report["ddi_research_grade"],
                "matched_pairs": inactive_report["matched_pairs"],
                "nonzero_pairs": inactive_report["nonzero_pairs"],
                "vocab_size": int(drug_count),
                "pad_idx": int(drug_vocab["pad_idx"]),
                "unk_idx": int(drug_vocab["unk_idx"]),
                "source_metadata": inactive_report["source_metadata"],
            },
        )
        _write_json_atomic(ddi_report_path, inactive_report)
        print(
            "DrugBank DDI build skipped: "
            f"source_path={source_path} reason={inactive_report['reason']}",
            flush=True,
        )
        return ddi_matrix_path

    match_source_counts: Counter[str] = Counter()
    token_matches: dict[str, list[str]] = {}
    unmatched_examples: list[dict[str, Any]] = []
    ambiguous_examples: list[dict[str, Any]] = []
    drugbank_id_to_vocab: dict[str, dict[str, Any]] = {}
    drug_count_parsed = 0
    ambiguous_record_count = 0

    for record in iter_drugbank_records(source_path):
        drug_count_parsed += 1
        primary_drugbank_id = str(record["primary_drugbank_id"])
        if not primary_drugbank_id:
            _append_example(
                unmatched_examples,
                {
                    "primary_drugbank_id": "",
                    "name": record["name"],
                    "name_token": record["name_token"],
                    "synonym_tokens": list(record["synonym_tokens"]),
                    "product_tokens": list(record["product_tokens"]),
                },
            )
            continue

        match = resolve_record_vocab_match(record, token_to_idx)
        if match["status"] == "matched":
            vocab_token = str(match["vocab_token"])
            match_source = str(match["match_source"])
            mapped_payload = {
                "vocab_token": vocab_token,
                "vocab_idx": int(match["vocab_idx"]),
                "match_source": match_source,
                "name": record["name"],
            }
            drugbank_id_to_vocab[primary_drugbank_id] = mapped_payload
            match_source_counts[match_source] += 1
            token_matches.setdefault(vocab_token, []).append(primary_drugbank_id)
        elif match["status"] == "ambiguous":
            ambiguous_record_count += 1
            _append_example(
                ambiguous_examples,
                {
                    "primary_drugbank_id": primary_drugbank_id,
                    "name": record["name"],
                    "match_source": match["match_source"],
                    "candidate_vocab_tokens": list(match["candidate_vocab_tokens"]),
                },
            )
        else:
            _append_example(
                unmatched_examples,
                {
                    "primary_drugbank_id": primary_drugbank_id,
                    "name": record["name"],
                    "name_token": record["name_token"],
                    "synonym_tokens": list(record["synonym_tokens"]),
                    "product_tokens": list(record["product_tokens"]),
                },
            )

    collision_token_count = 0
    extra_mapped_records = 0
    for mapped_ids in token_matches.values():
        if len(mapped_ids) > 1:
            collision_token_count += 1
            extra_mapped_records += len(mapped_ids) - 1

    raw_interaction_edges = 0
    mapped_interaction_edges = 0
    self_interaction_edges_skipped = 0
    pair_keys: set[tuple[int, int]] = set()
    temp_pairs_path = _part_path(ddi_pairs_path)
    ensure_dir(temp_pairs_path.parent)

    with gzip.open(temp_pairs_path, "wt", encoding="utf-8") as handle:
        for record in iter_drugbank_records(source_path):
            source_drugbank_id = str(record["primary_drugbank_id"])
            source_mapping = drugbank_id_to_vocab.get(source_drugbank_id)
            for interaction in record["interactions"]:
                raw_interaction_edges += 1
                target_drugbank_id = str(interaction["drugbank_id"])
                target_mapping = drugbank_id_to_vocab.get(target_drugbank_id)

                kept = False
                if source_mapping is not None and target_mapping is not None:
                    idx_a = int(source_mapping["vocab_idx"])
                    idx_b = int(target_mapping["vocab_idx"])
                    if idx_a != idx_b:
                        pair_keys.add((min(idx_a, idx_b), max(idx_a, idx_b)))
                        mapped_interaction_edges += 1
                        kept = True
                    else:
                        self_interaction_edges_skipped += 1

                serialized_pair = {
                    "source_drugbank_id": source_drugbank_id,
                    "source_name": record["name"],
                    "source_vocab_token": "" if source_mapping is None else source_mapping["vocab_token"],
                    "source_match_source": "" if source_mapping is None else source_mapping["match_source"],
                    "target_drugbank_id": target_drugbank_id,
                    "target_name": interaction["name"],
                    "target_vocab_token": "" if target_mapping is None else target_mapping["vocab_token"],
                    "target_match_source": "" if target_mapping is None else target_mapping["match_source"],
                    "description": interaction["description"],
                    "both_drugs_mapped": bool(source_mapping is not None and target_mapping is not None),
                    "kept": bool(kept),
                }
                handle.write(json.dumps(serialized_pair, ensure_ascii=True, sort_keys=True))
                handle.write("\n")

    _replace_path(temp_pairs_path, ddi_pairs_path)

    deduped_pair_count = int(len(pair_keys))
    candidate_vocab_size = _candidate_vocab_size(token_to_idx)
    matrix = _build_matrix(drug_count, pair_keys)
    report = {
        "active": bool(deduped_pair_count > 0),
        "reason": "available" if deduped_pair_count > 0 else "no_mapped_pairs",
        "source": str(source_path.resolve()),
        "requested_source": str(source_path.resolve()),
        "requested_source_format": DRUGBANK_SOURCE_FORMAT,
        "effective_source": str(source_path.resolve()),
        "effective_source_format": DRUGBANK_SOURCE_FORMAT,
        "source_format": DRUGBANK_SOURCE_FORMAT,
        "source_type": "drugbank",
        "ddi_type": DRUGBANK_DDI_TYPE,
        "ddi_source": str(source_path.resolve()),
        "ddi_research_grade": False,
        "ddi_pairs_path": str(ddi_pairs_path.resolve()),
        "ddi_matrix_path": str(ddi_matrix_path.resolve()),
        "ddi_report_path": str(ddi_report_path.resolve()),
        "vocab_size": int(drug_count),
        "drugbank_drugs_parsed": int(drug_count_parsed),
        "mapped_drugbank_records": int(len(drugbank_id_to_vocab)),
        "mapped_vocab_drugs": int(len(token_matches)),
        "raw_interaction_edges": int(raw_interaction_edges),
        "mapped_interaction_edges": int(mapped_interaction_edges),
        "mapped_pairs": deduped_pair_count,
        "matched_pairs": deduped_pair_count,
        "nonzero_pairs": deduped_pair_count,
        "dropped_interaction_edges": int(raw_interaction_edges - mapped_interaction_edges),
        "self_interaction_edges_skipped": int(self_interaction_edges_skipped),
        "match_source_counts": dict(sorted(match_source_counts.items())),
        "collision_counts": {
            "ambiguous_record_matches": int(ambiguous_record_count),
            "vocab_tokens_with_multiple_drugbank_records": int(collision_token_count),
            "extra_mapped_records_beyond_first": int(extra_mapped_records),
        },
        "coverage": {
            "mapped_record_fraction": _fraction(len(drugbank_id_to_vocab), drug_count_parsed),
            "mapped_vocab_fraction": _fraction(len(token_matches), max(candidate_vocab_size, 1)),
            "interaction_edge_mapping_fraction": _fraction(mapped_interaction_edges, raw_interaction_edges),
        },
        "unmatched_examples": unmatched_examples,
        "ambiguous_examples": ambiguous_examples,
        "source_metadata": drugbank_source_metadata(source_path),
    }
    _write_pt_atomic(
        ddi_matrix_path,
        {
            "matrix": matrix,
            "active": bool(report["active"]),
            "reason": report["reason"],
            "source": report["source"],
            "requested_source": report["requested_source"],
            "requested_source_format": report["requested_source_format"],
            "effective_source": report["effective_source"],
            "effective_source_format": report["effective_source_format"],
            "source_format": report["source_format"],
            "ddi_type": report["ddi_type"],
            "ddi_source": report["ddi_source"],
            "ddi_research_grade": report["ddi_research_grade"],
            "matched_pairs": report["matched_pairs"],
            "nonzero_pairs": report["nonzero_pairs"],
            "vocab_size": int(drug_count),
            "pad_idx": int(drug_vocab["pad_idx"]),
            "unk_idx": int(drug_vocab["unk_idx"]),
            "raw_interaction_edges": report["raw_interaction_edges"],
            "mapped_interaction_edges": report["mapped_interaction_edges"],
            "mapped_pairs": report["mapped_pairs"],
            "dropped_interaction_edges": report["dropped_interaction_edges"],
            "self_interaction_edges_skipped": report["self_interaction_edges_skipped"],
            "match_source_counts": dict(report["match_source_counts"]),
            "collision_counts": dict(report["collision_counts"]),
            "source_metadata": dict(report["source_metadata"]),
        },
    )
    _write_json_atomic(ddi_report_path, report)
    print(
        "Completed DrugBank DDI build: "
        f"drugbank_drugs_parsed={report['drugbank_drugs_parsed']} "
        f"mapped_vocab_drugs={report['mapped_vocab_drugs']} "
        f"matched_pairs={report['matched_pairs']} "
        f"active={bool(report['active'])}",
        flush=True,
    )
    return ddi_matrix_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an optional DrugBank-derived DDI matrix.")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    args = parser.parse_args()
    build_drugbank_ddi_matrix(args.config)


if __name__ == "__main__":
    main()
