from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LOGGER = logging.getLogger(__name__)

DEFAULT_DRUGBANK_FILENAMES = (
    "drugbank vocabulary.csv",
    "DrugBankVocabulary.csv",
    "drugbank_vocabulary.csv",
)
TRAILING_DOSAGE_FORM_TOKENS = {
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "injection",
    "injectable",
    "solution",
    "suspension",
    "syrup",
    "cream",
    "ointment",
    "gel",
    "lotion",
    "patch",
    "spray",
    "drop",
    "drops",
    "powder",
    "aerosol",
    "emulsion",
    "suppository",
    "enema",
    "elixir",
    "concentrate",
    "kit",
}
TRAILING_ROUTE_TOKENS = {
    "oral",
    "topical",
    "ophthalmic",
    "otic",
    "nasal",
    "dermal",
    "intravenous",
    "intramuscular",
    "subcutaneous",
    "injectable",
}


@dataclass(frozen=True)
class DrugBankVocabularyEntry:
    """Single normalized row from DrugBank Vocabulary."""

    drugbank_id: str
    canonical_name: str
    normalized_canonical_name: str
    synonyms: tuple[str, ...]
    normalized_synonyms: tuple[str, ...]


@dataclass(frozen=True)
class DrugBankVocabularyIndex:
    """Reusable DrugBank Vocabulary indexes for name normalization."""

    normalized_name_to_drugbank_id: dict[str, str]
    synonym_to_drugbank_id: dict[str, str]
    drugbank_id_to_canonical_name: dict[str, str]
    drugbank_id_to_all_normalized_names: dict[str, tuple[str, ...]]
    ambiguous_normalized_names: dict[str, tuple[str, ...]]
    ambiguous_synonyms: dict[str, tuple[str, ...]]
    meta: dict[str, Any]


def normalize_drugbank_text(
    value: str | None,
    *,
    strip_dosage_form: bool = True,
) -> str | None:
    """
    Normalize DrugBank names for loose matching across external vocabularies.

    The normalization is intentionally simple:
    - lowercase
    - replace light punctuation with spaces
    - collapse repeated spaces
    - optionally strip trailing dosage-form / route tokens
    """

    text = str(value or "").strip().lower()
    if not text:
        return None

    text = re.sub(r"[\(\)\[\]\{\},;:/\\\-\+]+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None

    if not strip_dosage_form:
        return text

    tokens = text.split()
    trailing_tokens = TRAILING_DOSAGE_FORM_TOKENS.union(TRAILING_ROUTE_TOKENS)
    while len(tokens) > 1 and tokens[-1] in trailing_tokens:
        tokens.pop()
    while len(tokens) > 1 and tokens[-1] in TRAILING_ROUTE_TOKENS:
        tokens.pop()

    normalized = " ".join(tokens).strip()
    return normalized or None


def resolve_drugbank_vocabulary_path(path_or_dir: str | Path) -> Path:
    """Resolve a DrugBank Vocabulary CSV from either a direct file path or a directory."""

    candidate = Path(path_or_dir)
    if candidate.is_file():
        return candidate
    if not candidate.exists():
        raise FileNotFoundError(f"DrugBank path does not exist: {candidate}")
    if not candidate.is_dir():
        raise FileNotFoundError(f"DrugBank path is neither a file nor a directory: {candidate}")

    for filename in DEFAULT_DRUGBANK_FILENAMES:
        resolved = candidate / filename
        if resolved.is_file():
            return resolved

    glob_matches = sorted(path for path in candidate.glob("*drugbank*voc*.csv") if path.is_file())
    if glob_matches:
        return glob_matches[0]

    raise FileNotFoundError(
        f"Could not find DrugBank Vocabulary CSV under {candidate}. "
        f"Checked: {list(DEFAULT_DRUGBANK_FILENAMES)} and '*drugbank*voc*.csv'."
    )


def load_drugbank_vocabulary(
    path_or_dir: str | Path,
) -> tuple[list[DrugBankVocabularyEntry], dict[str, Any]]:
    """
    Load and normalize DrugBank Vocabulary rows from CSV.

    The expected columns are:
    - DrugBank ID
    - Common name
    - Synonyms
    """

    csv_path = resolve_drugbank_vocabulary_path(path_or_dir)
    rows_scanned = 0
    rows_with_missing_id = 0
    rows_with_missing_canonical_name = 0
    rows_with_empty_normalized_name = 0
    total_synonym_assignments = 0
    entries: list[DrugBankVocabularyEntry] = []

    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required = {"DrugBank ID", "Common name", "Synonyms"}
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise KeyError(
                f"DrugBank Vocabulary is missing required columns {missing}. "
                f"Available columns: {fieldnames}"
            )

        for row in reader:
            rows_scanned += 1
            drugbank_id = str(row.get("DrugBank ID", "")).strip()
            canonical_name = str(row.get("Common name", "")).strip()
            if not drugbank_id:
                rows_with_missing_id += 1
                continue
            if not canonical_name:
                rows_with_missing_canonical_name += 1
                continue

            normalized_canonical_name = normalize_drugbank_text(canonical_name)
            if not normalized_canonical_name:
                rows_with_empty_normalized_name += 1
                continue

            synonyms = tuple(sorted(_iter_synonyms(row.get("Synonyms"))))
            normalized_synonyms = tuple(
                sorted(
                    {
                        normalized
                        for synonym in synonyms
                        if (normalized := normalize_drugbank_text(synonym))
                    }
                )
            )
            total_synonym_assignments += len(normalized_synonyms)
            entries.append(
                DrugBankVocabularyEntry(
                    drugbank_id=drugbank_id,
                    canonical_name=canonical_name,
                    normalized_canonical_name=normalized_canonical_name,
                    synonyms=synonyms,
                    normalized_synonyms=normalized_synonyms,
                )
            )

    meta = {
        "source_path": str(csv_path),
        "rows_scanned": int(rows_scanned),
        "rows_with_missing_id": int(rows_with_missing_id),
        "rows_with_missing_canonical_name": int(rows_with_missing_canonical_name),
        "rows_with_empty_normalized_name": int(rows_with_empty_normalized_name),
        "entries_loaded": int(len(entries)),
        "synonym_assignments_loaded": int(total_synonym_assignments),
    }
    LOGGER.info(
        "Loaded DrugBank Vocabulary from %s (entries=%s, normalized_synonyms=%s)",
        csv_path,
        len(entries),
        total_synonym_assignments,
    )
    return entries, meta


def build_drugbank_name_index(
    entries: Iterable[DrugBankVocabularyEntry],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]], dict[str, Any]]:
    """Build canonical normalized_name -> DrugBank ID mapping."""

    normalized_name_to_ids: dict[str, set[str]] = defaultdict(set)
    canonical_name_count = 0
    for entry in entries:
        normalized_name_to_ids[entry.normalized_canonical_name].add(entry.drugbank_id)
        canonical_name_count += 1

    normalized_name_to_drugbank_id, ambiguous_names = _finalize_single_id_index(normalized_name_to_ids)
    meta = {
        "canonical_name_assignments": int(canonical_name_count),
        "canonical_name_index_size": int(len(normalized_name_to_drugbank_id)),
        "ambiguous_canonical_name_count": int(len(ambiguous_names)),
    }
    return normalized_name_to_drugbank_id, ambiguous_names, meta


def build_drugbank_synonym_index(
    entries: Iterable[DrugBankVocabularyEntry],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]], dict[str, Any]]:
    """Build normalized synonym -> DrugBank ID mapping."""

    synonym_to_ids: dict[str, set[str]] = defaultdict(set)
    synonym_assignment_count = 0
    for entry in entries:
        for normalized_synonym in entry.normalized_synonyms:
            synonym_to_ids[normalized_synonym].add(entry.drugbank_id)
            synonym_assignment_count += 1

    synonym_to_drugbank_id, ambiguous_synonyms = _finalize_single_id_index(synonym_to_ids)
    meta = {
        "synonym_assignments": int(synonym_assignment_count),
        "synonym_index_size": int(len(synonym_to_drugbank_id)),
        "ambiguous_synonym_count": int(len(ambiguous_synonyms)),
    }
    return synonym_to_drugbank_id, ambiguous_synonyms, meta


def load_drugbank_vocabulary_index(path_or_dir: str | Path) -> DrugBankVocabularyIndex:
    """
    Load DrugBank Vocabulary and build reusable canonical/synonym indexes.

    This helper is the intended entrypoint for downstream modules such as
    `build_ddi_matrix.py`.
    """

    entries, load_meta = load_drugbank_vocabulary(path_or_dir)
    normalized_name_to_drugbank_id, ambiguous_normalized_names, name_meta = build_drugbank_name_index(entries)
    synonym_to_drugbank_id, ambiguous_synonyms, synonym_meta = build_drugbank_synonym_index(entries)
    drugbank_id_to_canonical_name = {
        entry.drugbank_id: entry.canonical_name
        for entry in entries
    }
    drugbank_id_to_all_normalized_names = {
        entry.drugbank_id: tuple(
            sorted({entry.normalized_canonical_name, *entry.normalized_synonyms})
        )
        for entry in entries
    }

    LOGGER.info(
        "Built DrugBank indexes (canonical_names=%s, synonyms=%s)",
        len(normalized_name_to_drugbank_id),
        len(synonym_to_drugbank_id),
    )
    return DrugBankVocabularyIndex(
        normalized_name_to_drugbank_id=normalized_name_to_drugbank_id,
        synonym_to_drugbank_id=synonym_to_drugbank_id,
        drugbank_id_to_canonical_name=drugbank_id_to_canonical_name,
        drugbank_id_to_all_normalized_names=drugbank_id_to_all_normalized_names,
        ambiguous_normalized_names=ambiguous_normalized_names,
        ambiguous_synonyms=ambiguous_synonyms,
        meta={
            **load_meta,
            **name_meta,
            **synonym_meta,
            "drugbank_id_count": int(len(drugbank_id_to_canonical_name)),
        },
    )


def _iter_synonyms(value: str | None) -> Iterable[str]:
    text = str(value or "").strip()
    if not text:
        return ()
    raw_parts = re.split(r"[|;\n]+", text)
    return tuple(
        sorted(
            {
                part.strip()
                for part in raw_parts
                if part and part.strip()
            }
        )
    )


def _finalize_single_id_index(
    mapping: dict[str, set[str]],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    single_id_index: dict[str, str] = {}
    ambiguous_index: dict[str, tuple[str, ...]] = {}
    for key, ids in mapping.items():
        if not key:
            continue
        sorted_ids = tuple(sorted(str(value) for value in ids if str(value).strip()))
        if not sorted_ids:
            continue
        single_id_index[key] = sorted_ids[0]
        if len(sorted_ids) > 1:
            ambiguous_index[key] = sorted_ids
    return single_id_index, ambiguous_index
