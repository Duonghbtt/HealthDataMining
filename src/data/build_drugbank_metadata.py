from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from src.data.drugbank import (
    DRUGBANK_SOURCE_FORMAT,
    drugbank_source_metadata,
    iter_drugbank_records,
    resolve_drugbank_paths,
    resolve_record_vocab_match,
)
from src.utils.io import ensure_dir, load_yaml_config, read_json, resolve_path, write_json


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


def _coerce_fraction(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _candidate_vocab_size(token_to_idx: Mapping[str, int]) -> int:
    return sum(1 for token in token_to_idx if str(token) not in {"PAD", "UNK"})


def _drug_vocab_path(config: Mapping[str, Any]) -> Path:
    interim_root = resolve_path(config["_project_root"], config["paths"]["interim_root"])
    return Path(interim_root) / "vocab" / "drug_vocab.json"


def _load_optional_drug_vocab(config: Mapping[str, Any]) -> dict[str, Any] | None:
    vocab_path = _drug_vocab_path(config)
    if not vocab_path.exists():
        return None
    payload = read_json(vocab_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Drug vocab payload must be a JSON object: {vocab_path}")
    return payload


def _inactive_summary(
    *,
    source_path: Path,
    records_path: Path,
    summary_path: Path,
    vocab_metadata_path: Path,
    drug_vocab: Mapping[str, Any] | None,
) -> dict[str, Any]:
    token_to_idx = {} if not isinstance(drug_vocab, Mapping) else dict(drug_vocab.get("token_to_idx", {}))
    return {
        "active": False,
        "reason": "missing_source_path",
        "source": str(source_path),
        "source_path": str(source_path),
        "source_exists": False,
        "source_format": DRUGBANK_SOURCE_FORMAT,
        "source_type": "drugbank",
        "records_path": str(records_path),
        "summary_path": str(summary_path),
        "vocab_metadata_path": str(vocab_metadata_path),
        "drugbank_drugs_parsed": 0,
        "raw_interaction_edges": 0,
        "matched_drugbank_records": 0,
        "matched_vocab_drugs": 0,
        "match_source_counts": {},
        "collision_counts": {
            "ambiguous_record_matches": 0,
            "vocab_tokens_with_multiple_drugbank_records": 0,
            "extra_mapped_records_beyond_first": 0,
        },
        "coverage": {
            "mapped_record_fraction": 0.0,
            "mapped_vocab_fraction": 0.0,
        },
        "unmatched_examples": [],
        "ambiguous_examples": [],
        "vocab_size": int(len(token_to_idx)),
        "vocab_metadata_written": isinstance(drug_vocab, Mapping),
        "research_grade": False,
        "auxiliary_only": True,
        "source_metadata": drugbank_source_metadata(source_path),
    }


def build_drugbank_metadata(config_path: str | Path) -> Path:
    config = load_yaml_config(config_path)
    paths = resolve_drugbank_paths(config)
    source_path = paths["source_path"]
    records_path = paths["records_path"]
    summary_path = paths["summary_path"]
    vocab_metadata_path = paths["vocab_metadata_path"]

    drug_vocab = _load_optional_drug_vocab(config)
    token_to_idx = {} if not isinstance(drug_vocab, Mapping) else dict(drug_vocab.get("token_to_idx", {}))

    if not source_path.exists():
        _write_empty_jsonl_gz(records_path)
        if isinstance(drug_vocab, Mapping):
            _write_json_atomic(vocab_metadata_path, {})
        summary = _inactive_summary(
            source_path=source_path,
            records_path=records_path,
            summary_path=summary_path,
            vocab_metadata_path=vocab_metadata_path,
            drug_vocab=drug_vocab,
        )
        _write_json_atomic(summary_path, summary)
        print(
            "DrugBank metadata build skipped: "
            f"source_path={source_path} reason={summary['reason']}",
            flush=True,
        )
        return records_path

    temp_records_path = _part_path(records_path)
    ensure_dir(temp_records_path.parent)

    match_source_counts: Counter[str] = Counter()
    token_matches: dict[str, list[dict[str, Any]]] = {}
    unmatched_examples: list[dict[str, Any]] = []
    ambiguous_examples: list[dict[str, Any]] = []
    drug_count = 0
    raw_interaction_edges = 0
    ambiguous_record_count = 0

    with gzip.open(temp_records_path, "wt", encoding="utf-8") as handle:
        for record in iter_drugbank_records(source_path):
            drug_count += 1
            raw_interaction_edges += int(record["interaction_count"])
            serialized_record = {
                "primary_drugbank_id": record["primary_drugbank_id"],
                "drugbank_ids": list(record["drugbank_ids"]),
                "alias_drugbank_ids": list(record["alias_drugbank_ids"]),
                "name": record["name"],
                "name_token": record["name_token"],
                "synonyms": list(record["synonyms"]),
                "synonym_tokens": list(record["synonym_tokens"]),
                "product_names": list(record["product_names"]),
                "product_tokens": list(record["product_tokens"]),
                "interaction_count": int(record["interaction_count"]),
            }
            handle.write(json.dumps(serialized_record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")

            if not token_to_idx:
                continue

            match = resolve_record_vocab_match(record, token_to_idx)
            if match["status"] == "matched":
                match_source = str(match["match_source"])
                vocab_token = str(match["vocab_token"])
                match_source_counts[match_source] += 1
                token_matches.setdefault(vocab_token, []).append(
                    {
                        "primary_drugbank_id": record["primary_drugbank_id"],
                        "name": record["name"],
                        "match_source": match_source,
                    }
                )
            elif match["status"] == "ambiguous":
                ambiguous_record_count += 1
                _append_example(
                    ambiguous_examples,
                    {
                        "primary_drugbank_id": record["primary_drugbank_id"],
                        "name": record["name"],
                        "match_source": match["match_source"],
                        "candidate_vocab_tokens": list(match["candidate_vocab_tokens"]),
                    },
                )
            else:
                _append_example(
                    unmatched_examples,
                    {
                        "primary_drugbank_id": record["primary_drugbank_id"],
                        "name": record["name"],
                        "name_token": record["name_token"],
                        "synonym_tokens": list(record["synonym_tokens"]),
                        "product_tokens": list(record["product_tokens"]),
                    },
                )

    _replace_path(temp_records_path, records_path)

    vocab_metadata: dict[str, Any] = {}
    collision_token_count = 0
    extra_mapped_records = 0
    for vocab_token, matches in sorted(token_matches.items()):
        if len(matches) > 1:
            collision_token_count += 1
            extra_mapped_records += len(matches) - 1
        vocab_metadata[vocab_token] = {
            "vocab_index": int(token_to_idx[vocab_token]),
            "matched_drugbank_ids": [match["primary_drugbank_id"] for match in matches],
            "matched_names": [match["name"] for match in matches],
            "match_sources": [match["match_source"] for match in matches],
            "record_count": int(len(matches)),
            "collision": bool(len(matches) > 1),
        }

    if isinstance(drug_vocab, Mapping):
        _write_json_atomic(vocab_metadata_path, vocab_metadata)

    matched_record_count = int(sum(match_source_counts.values()))
    matched_vocab_drug_count = int(len(vocab_metadata))
    candidate_vocab_size = _candidate_vocab_size(token_to_idx)
    summary = {
        "active": True,
        "reason": "available",
        "source": str(source_path.resolve()),
        "source_path": str(source_path.resolve()),
        "source_exists": True,
        "source_format": DRUGBANK_SOURCE_FORMAT,
        "source_type": "drugbank",
        "records_path": str(records_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "vocab_metadata_path": str(vocab_metadata_path.resolve()),
        "drugbank_drugs_parsed": int(drug_count),
        "raw_interaction_edges": int(raw_interaction_edges),
        "matched_drugbank_records": matched_record_count,
        "matched_vocab_drugs": matched_vocab_drug_count,
        "match_source_counts": dict(sorted(match_source_counts.items())),
        "collision_counts": {
            "ambiguous_record_matches": int(ambiguous_record_count),
            "vocab_tokens_with_multiple_drugbank_records": int(collision_token_count),
            "extra_mapped_records_beyond_first": int(extra_mapped_records),
        },
        "coverage": {
            "mapped_record_fraction": _coerce_fraction(matched_record_count, drug_count),
            "mapped_vocab_fraction": _coerce_fraction(matched_vocab_drug_count, max(candidate_vocab_size, 1)),
        },
        "unmatched_examples": unmatched_examples,
        "ambiguous_examples": ambiguous_examples,
        "vocab_size": int(len(token_to_idx)),
        "vocab_metadata_written": isinstance(drug_vocab, Mapping),
        "research_grade": False,
        "auxiliary_only": True,
        "source_metadata": drugbank_source_metadata(source_path),
    }
    _write_json_atomic(summary_path, summary)
    print(
        "Completed DrugBank metadata build: "
        f"drugbank_drugs_parsed={summary['drugbank_drugs_parsed']} "
        f"matched_drugbank_records={summary['matched_drugbank_records']} "
        f"matched_vocab_drugs={summary['matched_vocab_drugs']}",
        flush=True,
    )
    return records_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build optional DrugBank metadata artifacts.")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    args = parser.parse_args()
    build_drugbank_metadata(args.config)


if __name__ == "__main__":
    main()
