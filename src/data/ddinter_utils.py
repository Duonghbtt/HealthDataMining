from __future__ import annotations

import csv
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.drugbank_vocab_utils import DrugBankVocabularyIndex, normalize_drugbank_text


LOGGER = logging.getLogger(__name__)

DEFAULT_DDINTER_GLOB = "ddinter_downloads_code_*.csv"

LEFT_ID_COLUMN_ALIASES = ("DDInterID_A", "DDInterID1", "left_id", "drug_a_id", "source_id")
LEFT_NAME_COLUMN_ALIASES = ("Drug_A", "Drug1", "left_drug", "drug_a", "source_drug")
RIGHT_ID_COLUMN_ALIASES = ("DDInterID_B", "DDInterID2", "right_id", "drug_b_id", "target_id")
RIGHT_NAME_COLUMN_ALIASES = ("Drug_B", "Drug2", "right_drug", "drug_b", "target_drug")
LEVEL_COLUMN_ALIASES = ("Level", "Severity", "level", "severity")

LEVEL_SEVERITY_PRIORITY = {
    "major": 3,
    "moderate": 2,
    "minor": 1,
    "unknown": 0,
}

ItemKey = str | int


@dataclass(frozen=True)
class DDInterRawRow:
    """Single parsed DDInter CSV row before cleaning or deduplication."""

    source_file: str
    row_number: int
    left_ddinter_id: str
    left_name: str
    right_ddinter_id: str
    right_name: str
    level: str | None


@dataclass(frozen=True)
class DDInterEntity:
    """Normalized DDInter drug entity aggregated across raw rows."""

    ddinter_id: str
    canonical_name: str
    normalized_name: str
    observed_names: tuple[str, ...]
    normalized_aliases: tuple[str, ...]


@dataclass(frozen=True)
class DDInterPairRecord:
    """Cleaned unique DDInter interaction pair with aggregated metadata."""

    left_ddinter_id: str
    right_ddinter_id: str
    left_name: str
    right_name: str
    left_normalized_name: str
    right_normalized_name: str
    level_counts: tuple[tuple[str, int], ...]
    dominant_level: str | None
    max_severity_level: str | None
    row_count: int
    source_files: tuple[str, ...]


@dataclass(frozen=True)
class DDInterDataset:
    """DDInter entities and pairwise interactions ready for downstream projection."""

    entities: dict[str, DDInterEntity]
    normalized_name_to_ids: dict[str, tuple[str, ...]]
    ambiguous_normalized_names: dict[str, tuple[str, ...]]
    pair_records: tuple[DDInterPairRecord, ...]
    pair_key_to_record: dict[tuple[str, str], DDInterPairRecord]
    meta: dict[str, Any]


@dataclass(frozen=True)
class DDInterProjectedPair:
    """DDInter pair projected onto benchmark medication vocabulary items."""

    left_item_id: ItemKey
    right_item_id: ItemKey
    ddinter_pairs: tuple[tuple[str, str], ...]
    level_counts: tuple[tuple[str, int], ...]
    dominant_level: str | None
    max_severity_level: str | None
    row_count: int


def normalize_ddinter_text(value: str | None) -> str | None:
    """Normalize DDInter entity names for stable cross-dataset matching."""

    return normalize_drugbank_text(value, strip_dosage_form=True)


def resolve_ddinter_files(
    path_or_dir: str | Path,
    *,
    glob_pattern: str = DEFAULT_DDINTER_GLOB,
) -> tuple[Path, ...]:
    """Resolve DDInter CSV shards from either a directory or a direct CSV path."""

    candidate = Path(path_or_dir)
    if candidate.is_file():
        return (candidate,)
    if not candidate.exists():
        raise FileNotFoundError(f"DDInter path does not exist: {candidate}")
    if not candidate.is_dir():
        raise FileNotFoundError(f"DDInter path is neither a file nor a directory: {candidate}")

    matches = tuple(sorted(path for path in candidate.glob(glob_pattern) if path.is_file()))
    if not matches:
        raise FileNotFoundError(
            f"Could not find DDInter CSV files under {candidate} matching {glob_pattern!r}."
        )
    return matches


def load_ddinter_raw_rows(
    path_or_dir: str | Path,
    *,
    glob_pattern: str = DEFAULT_DDINTER_GLOB,
) -> tuple[list[DDInterRawRow], dict[str, Any]]:
    """Parse DDInter CSV shards into raw rows without deduplication."""

    files = resolve_ddinter_files(path_or_dir, glob_pattern=glob_pattern)
    raw_rows: list[DDInterRawRow] = []
    rows_scanned = 0
    rows_with_missing_left_id = 0
    rows_with_missing_right_id = 0
    rows_with_missing_left_name = 0
    rows_with_missing_right_name = 0
    rows_with_missing_level = 0

    for path in files:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [_strip_bom(name) for name in reader.fieldnames or ()]
            if not fieldnames:
                raise ValueError(f"DDInter file has no header row: {path}")
            reader.fieldnames = fieldnames

            left_id_column = _select_column_name(
                path,
                fieldnames,
                LEFT_ID_COLUMN_ALIASES,
                role="left DDInter ID",
            )
            left_name_column = _select_column_name(
                path,
                fieldnames,
                LEFT_NAME_COLUMN_ALIASES,
                role="left drug name",
            )
            right_id_column = _select_column_name(
                path,
                fieldnames,
                RIGHT_ID_COLUMN_ALIASES,
                role="right DDInter ID",
            )
            right_name_column = _select_column_name(
                path,
                fieldnames,
                RIGHT_NAME_COLUMN_ALIASES,
                role="right drug name",
            )
            level_column = _select_column_name(path, fieldnames, LEVEL_COLUMN_ALIASES, role="severity/level")

            for row_number, row in enumerate(reader, start=2):
                rows_scanned += 1
                left_ddinter_id = str(row.get(left_id_column, "")).strip()
                left_name = str(row.get(left_name_column, "")).strip()
                right_ddinter_id = str(row.get(right_id_column, "")).strip()
                right_name = str(row.get(right_name_column, "")).strip()
                level = str(row.get(level_column, "")).strip() or None

                if not left_ddinter_id:
                    rows_with_missing_left_id += 1
                if not right_ddinter_id:
                    rows_with_missing_right_id += 1
                if not left_name:
                    rows_with_missing_left_name += 1
                if not right_name:
                    rows_with_missing_right_name += 1
                if not level:
                    rows_with_missing_level += 1

                raw_rows.append(
                    DDInterRawRow(
                        source_file=str(path),
                        row_number=row_number,
                        left_ddinter_id=left_ddinter_id,
                        left_name=left_name,
                        right_ddinter_id=right_ddinter_id,
                        right_name=right_name,
                        level=level,
                    )
                )

    meta = {
        "source_path": str(Path(path_or_dir)),
        "glob_pattern": glob_pattern,
        "files": [str(path) for path in files],
        "file_count": int(len(files)),
        "rows_scanned": int(rows_scanned),
        "rows_loaded": int(len(raw_rows)),
        "rows_with_missing_left_id": int(rows_with_missing_left_id),
        "rows_with_missing_right_id": int(rows_with_missing_right_id),
        "rows_with_missing_left_name": int(rows_with_missing_left_name),
        "rows_with_missing_right_name": int(rows_with_missing_right_name),
        "rows_with_missing_level": int(rows_with_missing_level),
    }
    LOGGER.info("Loaded DDInter raw rows from %s file(s): rows=%s", len(files), len(raw_rows))
    return raw_rows, meta


def build_ddinter_entities(
    raw_rows: Iterable[DDInterRawRow],
) -> tuple[dict[str, DDInterEntity], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]], dict[str, Any]]:
    """Build normalized DDInter entity indexes from raw rows."""

    observed_name_counters: dict[str, Counter[str]] = defaultdict(Counter)
    observed_normalized_counters: dict[str, Counter[str]] = defaultdict(Counter)
    entity_mentions = 0
    mentions_missing_id_or_name = 0
    mentions_with_empty_normalized_name = 0

    for row in raw_rows:
        for ddinter_id, raw_name in (
            (row.left_ddinter_id, row.left_name),
            (row.right_ddinter_id, row.right_name),
        ):
            ddinter_id = str(ddinter_id).strip()
            raw_name = str(raw_name).strip()
            if not ddinter_id or not raw_name:
                mentions_missing_id_or_name += 1
                continue

            normalized_name = normalize_ddinter_text(raw_name)
            if not normalized_name:
                mentions_with_empty_normalized_name += 1
                continue

            observed_name_counters[ddinter_id][raw_name] += 1
            observed_normalized_counters[ddinter_id][normalized_name] += 1
            entity_mentions += 1

    entities: dict[str, DDInterEntity] = {}
    normalized_name_to_ids: dict[str, set[str]] = defaultdict(set)
    for ddinter_id in sorted(observed_name_counters):
        name_counter = observed_name_counters[ddinter_id]
        normalized_counter = observed_normalized_counters.get(ddinter_id, Counter())
        if not name_counter or not normalized_counter:
            continue

        canonical_name = _pick_most_representative_text(name_counter)
        normalized_name = normalize_ddinter_text(canonical_name) or _pick_most_representative_text(normalized_counter)
        normalized_aliases = tuple(sorted(normalized_counter))
        observed_names = tuple(sorted(name_counter))
        entities[ddinter_id] = DDInterEntity(
            ddinter_id=ddinter_id,
            canonical_name=canonical_name,
            normalized_name=normalized_name,
            observed_names=observed_names,
            normalized_aliases=normalized_aliases,
        )
        for alias in normalized_aliases:
            normalized_name_to_ids[alias].add(ddinter_id)

    normalized_name_index = {
        alias: tuple(sorted(ids))
        for alias, ids in sorted(normalized_name_to_ids.items())
        if alias and ids
    }
    ambiguous_normalized_names = {
        alias: ids
        for alias, ids in normalized_name_index.items()
        if len(ids) > 1
    }
    meta = {
        "entity_mentions": int(entity_mentions),
        "mentions_missing_id_or_name": int(mentions_missing_id_or_name),
        "mentions_with_empty_normalized_name": int(mentions_with_empty_normalized_name),
        "entity_count": int(len(entities)),
        "normalized_name_index_size": int(len(normalized_name_index)),
        "ambiguous_normalized_name_count": int(len(ambiguous_normalized_names)),
    }
    LOGGER.info(
        "Built DDInter entity index (entities=%s, normalized_names=%s)",
        len(entities),
        len(normalized_name_index),
    )
    return entities, normalized_name_index, ambiguous_normalized_names, meta


def build_pairwise_interactions(
    raw_rows: Iterable[DDInterRawRow],
    *,
    entities: Mapping[str, DDInterEntity] | None = None,
    include_unknown: bool = True,
) -> tuple[tuple[DDInterPairRecord, ...], dict[str, Any]]:
    """Clean and deduplicate DDInter rows into a unique pairwise interaction table."""

    pair_aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    rows_missing_required_fields = 0
    rows_with_empty_normalized_name = 0
    rows_self_pairs = 0
    rows_unknown_filtered = 0
    candidate_pair_rows = 0
    kept_pair_rows = 0
    level_counter_candidate_rows: Counter[str] = Counter()
    level_counter_kept_rows: Counter[str] = Counter()

    for row in raw_rows:
        left_ddinter_id = str(row.left_ddinter_id).strip()
        right_ddinter_id = str(row.right_ddinter_id).strip()
        left_name = str(row.left_name).strip()
        right_name = str(row.right_name).strip()

        if not left_ddinter_id or not right_ddinter_id or not left_name or not right_name:
            rows_missing_required_fields += 1
            continue

        left_normalized_name = normalize_ddinter_text(left_name)
        right_normalized_name = normalize_ddinter_text(right_name)
        if not left_normalized_name or not right_normalized_name:
            rows_with_empty_normalized_name += 1
            continue

        if left_ddinter_id == right_ddinter_id:
            rows_self_pairs += 1
            continue

        level = str(row.level or "Unknown").strip() or "Unknown"
        candidate_pair_rows += 1
        level_counter_candidate_rows[level] += 1

        if not include_unknown and level.lower() == "unknown":
            rows_unknown_filtered += 1
            continue

        kept_pair_rows += 1
        level_counter_kept_rows[level] += 1
        left_id, right_id = _ordered_ddinter_pair(left_ddinter_id, right_ddinter_id)
        if (left_id, right_id) != (left_ddinter_id, right_ddinter_id):
            left_name, right_name = right_name, left_name
            left_normalized_name, right_normalized_name = right_normalized_name, left_normalized_name

        aggregate = pair_aggregates.setdefault(
            (left_id, right_id),
            {
                "left_names": Counter(),
                "right_names": Counter(),
                "left_normalized_names": Counter(),
                "right_normalized_names": Counter(),
                "level_counter": Counter(),
                "source_files": set(),
                "row_count": 0,
            },
        )
        aggregate["left_names"][left_name] += 1
        aggregate["right_names"][right_name] += 1
        aggregate["left_normalized_names"][left_normalized_name] += 1
        aggregate["right_normalized_names"][right_normalized_name] += 1
        aggregate["level_counter"][level] += 1
        aggregate["source_files"].add(str(row.source_file))
        aggregate["row_count"] += 1

    entity_lookup = dict(entities or {})
    pair_records: list[DDInterPairRecord] = []
    for left_ddinter_id, right_ddinter_id in sorted(pair_aggregates):
        aggregate = pair_aggregates[(left_ddinter_id, right_ddinter_id)]
        left_entity = entity_lookup.get(left_ddinter_id)
        right_entity = entity_lookup.get(right_ddinter_id)
        left_name = left_entity.canonical_name if left_entity else _pick_most_representative_text(aggregate["left_names"])
        right_name = right_entity.canonical_name if right_entity else _pick_most_representative_text(aggregate["right_names"])
        left_normalized_name = (
            left_entity.normalized_name
            if left_entity
            else _pick_most_representative_text(aggregate["left_normalized_names"])
        )
        right_normalized_name = (
            right_entity.normalized_name
            if right_entity
            else _pick_most_representative_text(aggregate["right_normalized_names"])
        )
        level_counter = aggregate["level_counter"]
        pair_records.append(
            DDInterPairRecord(
                left_ddinter_id=left_ddinter_id,
                right_ddinter_id=right_ddinter_id,
                left_name=left_name,
                right_name=right_name,
                left_normalized_name=left_normalized_name,
                right_normalized_name=right_normalized_name,
                level_counts=_sorted_counter_items(level_counter),
                dominant_level=_pick_dominant_level(level_counter),
                max_severity_level=_pick_max_severity_level(level_counter),
                row_count=int(aggregate["row_count"]),
                source_files=tuple(sorted(aggregate["source_files"])),
            )
        )

    meta = {
        "include_unknown": bool(include_unknown),
        "rows_missing_required_fields": int(rows_missing_required_fields),
        "rows_with_empty_normalized_name": int(rows_with_empty_normalized_name),
        "rows_self_pairs": int(rows_self_pairs),
        "rows_unknown_filtered": int(rows_unknown_filtered),
        "pair_rows_before_cleaning": int(candidate_pair_rows),
        "pair_rows_after_cleaning": int(kept_pair_rows),
        "unique_pair_count": int(len(pair_records)),
        "level_counts_candidate_rows": {key: int(value) for key, value in sorted(level_counter_candidate_rows.items())},
        "level_counts_kept_rows": {key: int(value) for key, value in sorted(level_counter_kept_rows.items())},
    }
    LOGGER.info(
        "Built DDInter pair table (candidate_rows=%s, unique_pairs=%s, include_unknown=%s)",
        candidate_pair_rows,
        len(pair_records),
        include_unknown,
    )
    return tuple(pair_records), meta


def build_ddinter_dataset(
    raw_rows: Iterable[DDInterRawRow],
    *,
    include_unknown: bool = True,
) -> DDInterDataset:
    """Build the full DDInter dataset indexes from raw parsed rows."""

    raw_row_list = list(raw_rows)
    entities, normalized_name_to_ids, ambiguous_normalized_names, entity_meta = build_ddinter_entities(raw_row_list)
    pair_records, pair_meta = build_pairwise_interactions(
        raw_row_list,
        entities=entities,
        include_unknown=include_unknown,
    )
    pair_key_to_record = {
        (record.left_ddinter_id, record.right_ddinter_id): record
        for record in pair_records
    }
    return DDInterDataset(
        entities=entities,
        normalized_name_to_ids=normalized_name_to_ids,
        ambiguous_normalized_names=ambiguous_normalized_names,
        pair_records=pair_records,
        pair_key_to_record=pair_key_to_record,
        meta={
            "include_unknown": bool(include_unknown),
            "entity_index": entity_meta,
            "pair_index": pair_meta,
        },
    )


def load_ddinter_dataset(
    path_or_dir: str | Path,
    *,
    glob_pattern: str = DEFAULT_DDINTER_GLOB,
    include_unknown: bool = True,
) -> DDInterDataset:
    """Convenience entrypoint that loads, cleans, and indexes DDInter."""

    raw_rows, raw_meta = load_ddinter_raw_rows(path_or_dir, glob_pattern=glob_pattern)
    dataset = build_ddinter_dataset(raw_rows, include_unknown=include_unknown)
    meta = {
        **dataset.meta,
        "raw_load": raw_meta,
    }
    LOGGER.info(
        "Loaded DDInter dataset (entities=%s, unique_pairs=%s)",
        len(dataset.entities),
        len(dataset.pair_records),
    )
    return DDInterDataset(
        entities=dataset.entities,
        normalized_name_to_ids=dataset.normalized_name_to_ids,
        ambiguous_normalized_names=dataset.ambiguous_normalized_names,
        pair_records=dataset.pair_records,
        pair_key_to_record=dataset.pair_key_to_record,
        meta=meta,
    )


def match_names_to_ddinter_ids(
    names: Iterable[str],
    *,
    ddinter_dataset: DDInterDataset,
    drugbank_index: DrugBankVocabularyIndex | None = None,
) -> tuple[set[str], dict[str, Any]]:
    """Match candidate drug names to DDInter IDs using direct and DrugBank-bridged lookup."""

    name_list = [str(name) for name in names]
    ddinter_ids: set[str] = set()
    direct_name_matches: dict[str, list[str]] = {}
    drugbank_bridge_matches: dict[str, list[str]] = {}
    unmatched_names: list[str] = []
    normalized_inputs: list[str] = []

    for raw_name in name_list:
        normalized_name = normalize_ddinter_text(raw_name)
        if not normalized_name:
            continue
        normalized_inputs.append(normalized_name)

        direct_ids = sorted(ddinter_dataset.normalized_name_to_ids.get(normalized_name, ()))
        if direct_ids:
            ddinter_ids.update(direct_ids)
            direct_name_matches[raw_name] = direct_ids
            continue

        bridged_ids: set[str] = set()
        for drugbank_id in _resolve_drugbank_ids_for_normalized_name(
            normalized_name,
            drugbank_index=drugbank_index,
        ):
            for alias in drugbank_index.drugbank_id_to_all_normalized_names.get(drugbank_id, ()):
                bridged_ids.update(ddinter_dataset.normalized_name_to_ids.get(alias, ()))

        if bridged_ids:
            resolved_ids = sorted(bridged_ids)
            ddinter_ids.update(resolved_ids)
            drugbank_bridge_matches[raw_name] = resolved_ids
            continue

        unmatched_names.append(raw_name)

    report = {
        "input_name_count": int(len(name_list)),
        "normalized_input_count": int(len(normalized_inputs)),
        "matched_ddinter_id_count": int(len(ddinter_ids)),
        "direct_name_matches": direct_name_matches,
        "drugbank_bridge_matches": drugbank_bridge_matches,
        "unmatched_names": unmatched_names,
    }
    return ddinter_ids, report


def map_vocab_items_to_ddinter_ids(
    vocab_item_to_candidate_names: Mapping[ItemKey, Iterable[str]],
    *,
    ddinter_dataset: DDInterDataset,
    drugbank_index: DrugBankVocabularyIndex | None = None,
    unmatched_example_limit: int = 20,
) -> tuple[dict[ItemKey, tuple[str, ...]], dict[str, Any]]:
    """Map benchmark medication vocabulary items to DDInter IDs."""

    vocab_item_to_ddinter_ids: dict[ItemKey, tuple[str, ...]] = {}
    unmatched_examples: list[dict[str, Any]] = []
    sample_matches: list[dict[str, Any]] = []
    matched_ddinter_ids: set[str] = set()

    for item_id, candidate_names in vocab_item_to_candidate_names.items():
        candidate_name_list = [
            str(name).strip()
            for name in candidate_names
            if str(name).strip()
        ]
        resolved_ids, match_report = match_names_to_ddinter_ids(
            candidate_name_list,
            ddinter_dataset=ddinter_dataset,
            drugbank_index=drugbank_index,
        )
        if resolved_ids:
            sorted_ids = tuple(sorted(resolved_ids))
            vocab_item_to_ddinter_ids[item_id] = sorted_ids
            matched_ddinter_ids.update(sorted_ids)
            if len(sample_matches) < 20:
                sample_matches.append(
                    {
                        "item_id": item_id,
                        "candidate_names": candidate_name_list[:10],
                        "ddinter_ids": list(sorted_ids),
                    }
                )
            continue

        if len(unmatched_examples) < unmatched_example_limit:
            unmatched_examples.append(
                {
                    "item_id": item_id,
                    "candidate_names": candidate_name_list[:10],
                    "unmatched_names": list(match_report["unmatched_names"])[:10],
                }
            )

    meta = {
        "num_vocab_items": int(len(vocab_item_to_candidate_names)),
        "num_vocab_items_matched": int(len(vocab_item_to_ddinter_ids)),
        "num_vocab_items_unmatched": int(len(vocab_item_to_candidate_names) - len(vocab_item_to_ddinter_ids)),
        "matched_ddinter_id_count": int(len(matched_ddinter_ids)),
        "unmatched_examples": unmatched_examples,
        "sample_matches": sample_matches,
    }
    LOGGER.info(
        "Mapped DDInter IDs for %s/%s vocab items",
        len(vocab_item_to_ddinter_ids),
        len(vocab_item_to_candidate_names),
    )
    return vocab_item_to_ddinter_ids, meta


def project_ddinter_pairs_to_vocab(
    vocab_item_to_ddinter_ids: Mapping[ItemKey, Iterable[str]],
    *,
    ddinter_dataset: DDInterDataset,
    unmatched_example_limit: int = 20,
) -> tuple[tuple[DDInterProjectedPair, ...], dict[str, Any]]:
    """Project DDInter pairwise interactions onto benchmark medication vocabulary."""

    ddinter_id_to_vocab_items: dict[str, set[ItemKey]] = defaultdict(set)
    for item_id, ddinter_ids in vocab_item_to_ddinter_ids.items():
        for ddinter_id in ddinter_ids:
            ddinter_id_to_vocab_items[str(ddinter_id)].add(item_id)

    projected_pairs: dict[tuple[ItemKey, ItemKey], dict[str, Any]] = {}
    matched_pair_records = 0
    unmatched_pair_records = 0
    collapsed_same_item_records = 0
    unmatched_examples: list[dict[str, Any]] = []

    for pair_record in ddinter_dataset.pair_records:
        left_items = ddinter_id_to_vocab_items.get(pair_record.left_ddinter_id, set())
        right_items = ddinter_id_to_vocab_items.get(pair_record.right_ddinter_id, set())
        if not left_items or not right_items:
            unmatched_pair_records += 1
            if len(unmatched_examples) < unmatched_example_limit:
                unmatched_examples.append(
                    {
                        "left_ddinter_id": pair_record.left_ddinter_id,
                        "left_name": pair_record.left_name,
                        "right_ddinter_id": pair_record.right_ddinter_id,
                        "right_name": pair_record.right_name,
                        "max_severity_level": pair_record.max_severity_level,
                    }
                )
            continue

        emitted = False
        for left_item in left_items:
            for right_item in right_items:
                if left_item == right_item:
                    collapsed_same_item_records += 1
                    continue
                pair_key = _ordered_item_pair(left_item, right_item)
                aggregate = projected_pairs.setdefault(
                    pair_key,
                    {
                        "ddinter_pairs": set(),
                        "level_counter": Counter(),
                        "row_count": 0,
                    },
                )
                aggregate["ddinter_pairs"].add((pair_record.left_ddinter_id, pair_record.right_ddinter_id))
                for level, count in pair_record.level_counts:
                    aggregate["level_counter"][level] += int(count)
                aggregate["row_count"] += int(pair_record.row_count)
                emitted = True

        if emitted:
            matched_pair_records += 1

    finalized_pairs: list[DDInterProjectedPair] = []
    for left_item_id, right_item_id in sorted(projected_pairs, key=lambda pair: (_item_sort_key(pair[0]), _item_sort_key(pair[1]))):
        aggregate = projected_pairs[(left_item_id, right_item_id)]
        level_counter = aggregate["level_counter"]
        finalized_pairs.append(
            DDInterProjectedPair(
                left_item_id=left_item_id,
                right_item_id=right_item_id,
                ddinter_pairs=tuple(sorted(aggregate["ddinter_pairs"])),
                level_counts=_sorted_counter_items(level_counter),
                dominant_level=_pick_dominant_level(level_counter),
                max_severity_level=_pick_max_severity_level(level_counter),
                row_count=int(aggregate["row_count"]),
            )
        )

    meta = {
        "num_vocab_items_with_ddinter_ids": int(len(vocab_item_to_ddinter_ids)),
        "num_ddinter_ids_in_vocab_map": int(len(ddinter_id_to_vocab_items)),
        "num_ddinter_pair_records": int(len(ddinter_dataset.pair_records)),
        "num_ddinter_pair_records_with_vocab_match": int(matched_pair_records),
        "num_ddinter_pair_records_without_vocab_match": int(unmatched_pair_records),
        "num_pair_records_collapsed_to_same_item": int(collapsed_same_item_records),
        "num_projected_vocab_pairs": int(len(finalized_pairs)),
        "unmatched_examples": unmatched_examples,
    }
    LOGGER.info(
        "Projected DDInter pairs to vocab (projected_pairs=%s, matched_pair_records=%s)",
        len(finalized_pairs),
        matched_pair_records,
    )
    return tuple(finalized_pairs), meta


def _strip_bom(value: str | None) -> str:
    return str(value or "").lstrip("\ufeff")


def _normalize_header_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _strip_bom(value).strip().lower()).strip("_")


def _select_column_name(path: Path, fieldnames: Iterable[str], aliases: Iterable[str], *, role: str) -> str:
    normalized_to_original = {
        _normalize_header_name(fieldname): fieldname
        for fieldname in fieldnames
        if str(fieldname).strip()
    }
    for alias in aliases:
        normalized_alias = _normalize_header_name(alias)
        if normalized_alias in normalized_to_original:
            return normalized_to_original[normalized_alias]
    raise KeyError(
        f"DDInter file {path} is missing a {role} column. Available columns: {list(fieldnames)}"
    )


def _ordered_ddinter_pair(left_id: str, right_id: str) -> tuple[str, str]:
    return (left_id, right_id) if str(left_id) <= str(right_id) else (right_id, left_id)


def _item_sort_key(value: ItemKey) -> tuple[int, str]:
    if isinstance(value, int):
        return (0, f"{value:012d}")
    return (1, str(value))


def _ordered_item_pair(left_item: ItemKey, right_item: ItemKey) -> tuple[ItemKey, ItemKey]:
    return (
        (left_item, right_item)
        if _item_sort_key(left_item) <= _item_sort_key(right_item)
        else (right_item, left_item)
    )


def _pick_most_representative_text(counter: Counter[str]) -> str:
    if not counter:
        raise ValueError("Cannot choose representative text from an empty counter.")
    return sorted(
        counter.items(),
        key=lambda item: (-int(item[1]), len(str(item[0])), str(item[0]).lower()),
    )[0][0]


def _sorted_counter_items(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(
        (str(level), int(count))
        for level, count in sorted(
            counter.items(),
            key=lambda item: (
                -int(item[1]),
                -LEVEL_SEVERITY_PRIORITY.get(str(item[0]).lower(), -1),
                str(item[0]).lower(),
            ),
        )
    )


def _pick_dominant_level(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return sorted(
        counter.items(),
        key=lambda item: (
            -int(item[1]),
            -LEVEL_SEVERITY_PRIORITY.get(str(item[0]).lower(), -1),
            str(item[0]).lower(),
        ),
    )[0][0]


def _pick_max_severity_level(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return sorted(
        counter.items(),
        key=lambda item: (
            -LEVEL_SEVERITY_PRIORITY.get(str(item[0]).lower(), -1),
            -int(item[1]),
            str(item[0]).lower(),
        ),
    )[0][0]


def _resolve_drugbank_ids_for_normalized_name(
    normalized_name: str,
    *,
    drugbank_index: DrugBankVocabularyIndex | None,
) -> set[str]:
    if drugbank_index is None:
        return set()

    resolved_ids: set[str] = set()
    direct_name_id = drugbank_index.normalized_name_to_drugbank_id.get(normalized_name)
    if direct_name_id:
        resolved_ids.add(str(direct_name_id))
    synonym_id = drugbank_index.synonym_to_drugbank_id.get(normalized_name)
    if synonym_id:
        resolved_ids.add(str(synonym_id))
    resolved_ids.update(drugbank_index.ambiguous_normalized_names.get(normalized_name, ()))
    resolved_ids.update(drugbank_index.ambiguous_synonyms.get(normalized_name, ()))
    return {drugbank_id for drugbank_id in resolved_ids if str(drugbank_id).strip()}
