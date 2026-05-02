from __future__ import annotations

import json
import heapq
import logging
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence


LOGGER = logging.getLogger(__name__)

RXNCONSO_RXCUI_INDEX = 0
RXNCONSO_LAT_INDEX = 1
RXNCONSO_RXAUI_INDEX = 7
RXNCONSO_SAB_INDEX = 11
RXNCONSO_TTY_INDEX = 12
RXNCONSO_STR_INDEX = 14
RXNCONSO_SUPPRESS_INDEX = 16

RXNREL_RXCUI1_INDEX = 0
RXNREL_AUI1_INDEX = 1
RXNREL_RXCUI2_INDEX = 4
RXNREL_AUI2_INDEX = 5
RXNREL_RELA_INDEX = 7
RXNREL_SUPPRESS_INDEX = 14

RXNSAT_RXCUI_INDEX = 0
RXNSAT_CODE_INDEX = 5
RXNSAT_ATN_INDEX = 8
RXNSAT_SAB_INDEX = 9
RXNSAT_ATV_INDEX = 10
RXNSAT_SUPPRESS_INDEX = 11

DEFAULT_NAME_SABS = ("RXNORM", "MTHSPL")
DEFAULT_INGREDIENT_TTYS = ("IN", "MIN", "PIN")
RXNORM_PROGRESS_LOG_EVERY = 2_000_000
DEFAULT_MAX_COMPONENT_INGREDIENT_COUNT_FOR_AUTO_ASSIGNMENT = 8
DEFAULT_INGREDIENT_BRIDGE_RELATIONS = (
    "has_ingredient",
    "ingredient_of",
    "consists_of",
    "constitutes",
)
NDC_SAB_PRIORITY = {"RXNORM": 0, "MTHSPL": 1, "NDDF": 2, "VANDF": 3}
TTY_PRIORITY = {
    "IN": 0,
    "PIN": 1,
    "MIN": 2,
    "SCD": 3,
    "SBD": 4,
    "SCDF": 5,
    "SBDF": 6,
    "SCDC": 7,
    "SBDC": 8,
    "GPCK": 9,
    "BPCK": 10,
    "BN": 11,
    "SY": 12,
    "DF": 13,
}


@dataclass(frozen=True)
class RxNormTables:
    """Resolved RxNorm RRF file paths under a release directory."""

    root: Path
    conso_path: Path
    rel_path: Path
    sat_path: Path


@dataclass(frozen=True)
class RxNormNdcIndex:
    """NDC -> RxCUI mapping plus build metadata."""

    ndc_to_rxcui: dict[str, str]
    meta: dict[str, Any]


@dataclass(frozen=True)
class RxNormNameRecord:
    """Canonical and synonym names known for a single RxCUI."""

    preferred_name: str
    normalized_preferred_name: str | None
    synonyms: tuple[str, ...]
    normalized_synonyms: tuple[str, ...]
    term_types: tuple[str, ...]
    source_abbreviations: tuple[str, ...]


@dataclass(frozen=True)
class RxNormNameIndex:
    """Forward and reverse indexes for RxCUI name normalization."""

    rxcui_to_names: dict[str, RxNormNameRecord]
    normalized_name_to_rxcuis: dict[str, tuple[str, ...]]
    meta: dict[str, Any]


@dataclass(frozen=True)
class RxNormIngredientIndex:
    """Ingredient-level normalization derived from RxNorm relations."""

    rxcui_to_ingredient_rxcuis: dict[str, tuple[str, ...]]
    rxcui_to_ingredient_names: dict[str, tuple[str, ...]]
    ingredient_rxcui_to_name: dict[str, str]
    ingredient_safe_rxcuis: tuple[str, ...]
    meta: dict[str, Any]


@dataclass(frozen=True)
class RxNormMinimalIndex:
    """Minimal RxNorm knowledge bundle for prescription normalization and DDI joins."""

    tables: RxNormTables
    ndc_index: RxNormNdcIndex
    name_index: RxNormNameIndex
    ingredient_index: RxNormIngredientIndex
    meta: dict[str, Any]


@dataclass(frozen=True)
class _NameCandidate:
    rxcui: str
    sab: str
    tty: str
    name: str


def normalize_rxcui(value: str | None) -> str | None:
    """Normalize a raw RxCUI to digits without leading zeros."""

    digits = re.sub(r"\D", "", str(value or ""))
    if not digits:
        return None
    return digits.lstrip("0") or "0"


def normalize_ndc(value: str | None) -> str | None:
    """Normalize a raw NDC to digits only."""

    digits = re.sub(r"\D", "", str(value or ""))
    if not digits or set(digits) == {"0"}:
        return None
    return digits


def normalize_drug_name(value: str | None) -> str | None:
    """Normalize a drug name for deterministic matching across sources."""

    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").upper()).strip("_")
    return cleaned or None


def is_suspicious_ingredient_name(value: str | None) -> bool:
    """
    Heuristic guardrail for ingredient labels.

    Ingredient-level labels should not look like clinical drug products,
    branded packs, or route/dose-form specific strings.
    """

    text = str(value or "").strip()
    if not text:
        return True
    upper = text.upper()
    if "[" in upper or "]" in upper or "{" in upper or "}" in upper:
        return True
    if re.search(r"\b\d+(\.\d+)?\s*(MG|MCG|G|ML|MEQ|UNT|UNITS|%)\b", upper):
        return True
    if re.search(
        r"\b(TABLET|CAPSULE|INJECTION|INJECTABLE|SOLUTION|SUSPENSION|SYRUP|PACK|KIT|VIAL|PATCH|LOTION|CREAM|"
        r"OINTMENT|SPRAY|GEL|PILL|PRODUCT|DOSE|ORAL|TOPICAL|OPHTHALMIC|OTIC|NASAL|RECTAL|VAGINAL|"
        r"SUBCUTANEOUS|INTRAVENOUS|IRRIGATION|INHALATION|INHALER|AEROSOL|LIQUID|POWDER|NEBULIZER|NEB)\b",
        upper,
    ):
        return True
    return False


def load_rxnorm_tables(rxnorm_root: str | Path) -> RxNormTables:
    """
    Resolve the core RxNorm RRF files under an extracted release directory.

    Supported layouts:
    - <root>/rrf/RXN*.RRF
    - <root>/prescribe/rrf/RXN*.RRF
    """

    root = Path(rxnorm_root)
    conso_candidates = (
        root / "rrf" / "RXNCONSO.RRF",
        root / "prescribe" / "rrf" / "RXNCONSO.RRF",
    )
    rel_candidates = (
        root / "rrf" / "RXNREL.RRF",
        root / "prescribe" / "rrf" / "RXNREL.RRF",
    )
    sat_candidates = (
        root / "rrf" / "RXNSAT.RRF",
        root / "prescribe" / "rrf" / "RXNSAT.RRF",
    )

    conso_path = next((path for path in conso_candidates if path.is_file()), None)
    rel_path = next((path for path in rel_candidates if path.is_file()), None)
    sat_path = next((path for path in sat_candidates if path.is_file()), None)
    if conso_path is None or rel_path is None or sat_path is None:
        raise FileNotFoundError(
            "RxNorm release is incomplete. Expected RXNCONSO.RRF, RXNREL.RRF, and RXNSAT.RRF "
            f"under {root}."
        )

    tables = RxNormTables(
        root=root,
        conso_path=conso_path,
        rel_path=rel_path,
        sat_path=sat_path,
    )
    LOGGER.info(
        "Resolved RxNorm tables under %s (conso=%s, rel=%s, sat=%s)",
        tables.root,
        tables.conso_path,
        tables.rel_path,
        tables.sat_path,
    )
    return tables


def build_ndc_to_rxcui_map(
    tables_or_root: RxNormTables | str | Path,
) -> RxNormNdcIndex:
    """
    Build a deterministic NDC -> RxCUI map directly from RXNSAT.RRF.

    Duplicate NDCs are resolved by SAB priority, non-suppressed preference,
    then stable lexical RxCUI order. Malformed rows and invalid RxCUIs are
    skipped and counted in the returned metadata.
    """

    tables = _coerce_tables(tables_or_root)
    ndc_candidates: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    malformed_rows = 0
    invalid_rxcui_rows = 0
    missing_ndc_rows = 0
    ndc_rows = 0

    with tables.sat_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            parts = _split_rrf_row(raw_line)
            if len(parts) <= RXNSAT_SUPPRESS_INDEX:
                malformed_rows += 1
                _warn_truncated_row(tables.sat_path, line_number, len(parts), RXNSAT_SUPPRESS_INDEX + 1)
                continue

            atn = str(parts[RXNSAT_ATN_INDEX]).strip().upper()
            if atn != "NDC":
                continue

            ndc_rows += 1
            rxcui = normalize_rxcui(parts[RXNSAT_RXCUI_INDEX])
            if not rxcui:
                invalid_rxcui_rows += 1
                continue

            ndc = normalize_ndc(parts[RXNSAT_ATV_INDEX] or parts[RXNSAT_CODE_INDEX])
            if not ndc:
                missing_ndc_rows += 1
                continue

            sab = str(parts[RXNSAT_SAB_INDEX]).strip().upper()
            suppress = str(parts[RXNSAT_SUPPRESS_INDEX]).strip().upper()
            ndc_candidates[ndc].append((sab, suppress, rxcui))

    ndc_to_rxcui: dict[str, str] = {}
    ambiguous_examples: list[dict[str, Any]] = []
    ambiguous_ndc_count = 0
    for ndc, candidates in ndc_candidates.items():
        ranked = sorted(
            candidates,
            key=lambda item: (
                _sab_priority(item[0], NDC_SAB_PRIORITY),
                _suppression_rank(item[1]),
                item[2],
            ),
        )
        selected_rxcui = ranked[0][2]
        ndc_to_rxcui[ndc] = selected_rxcui

        unique_rxcuis = sorted({candidate[2] for candidate in candidates})
        if len(unique_rxcuis) > 1:
            ambiguous_ndc_count += 1
            if len(ambiguous_examples) < 25:
                ambiguous_examples.append(
                    {
                        "ndc": ndc,
                        "selected_rxcui": selected_rxcui,
                        "candidate_rxcuis": unique_rxcuis,
                    }
                )

    meta = {
        "source_path": str(tables.sat_path),
        "rows_scanned": int(ndc_rows),
        "rows_malformed": int(malformed_rows),
        "rows_with_invalid_rxcui": int(invalid_rxcui_rows),
        "rows_with_missing_ndc": int(missing_ndc_rows),
        "num_ndc_entries": int(len(ndc_to_rxcui)),
        "ambiguous_ndc_count": int(ambiguous_ndc_count),
        "ambiguous_ndc_examples": ambiguous_examples,
        "ndc_value_fields_preference": ["ATV", "CODE"],
    }
    LOGGER.info(
        "Built NDC -> RxCUI map with %s entries from %s (ambiguous_ndc=%s, malformed_rows=%s)",
        len(ndc_to_rxcui),
        tables.sat_path,
        ambiguous_ndc_count,
        malformed_rows,
    )
    return RxNormNdcIndex(ndc_to_rxcui=ndc_to_rxcui, meta=meta)


def build_rxcui_name_index(
    tables_or_root: RxNormTables | str | Path,
    *,
    allowed_name_sabs: Sequence[str] = DEFAULT_NAME_SABS,
) -> RxNormNameIndex:
    """
    Build RxCUI -> names/synonyms and normalized_name -> RxCUIs from RXNCONSO.RRF.

    The index is intended for robust medication normalization and joining to
    external vocabularies such as DrugBank Vocabulary and DDInter.
    """

    tables = _coerce_tables(tables_or_root)
    allowed_sabs = {str(value).strip().upper() for value in allowed_name_sabs}
    preferred_name_rank: dict[str, tuple[int, int, int, str]] = {}
    preferred_name_by_rxcui: dict[str, str] = {}
    synonyms_by_rxcui: dict[str, set[str]] = defaultdict(set)
    normalized_synonyms_by_rxcui: dict[str, set[str]] = defaultdict(set)
    term_types_by_rxcui: dict[str, set[str]] = defaultdict(set)
    source_abbreviations_by_rxcui: dict[str, set[str]] = defaultdict(set)
    name_candidates: dict[str, dict[str, _NameCandidate]] = defaultdict(dict)

    rows_scanned = 0
    malformed_rows = 0
    invalid_rxcui_rows = 0
    non_english_rows = 0
    suppressed_rows = 0
    empty_name_rows = 0
    filtered_sab_rows = 0

    with tables.conso_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            parts = _split_rrf_row(raw_line)
            if len(parts) <= RXNCONSO_SUPPRESS_INDEX:
                malformed_rows += 1
                _warn_truncated_row(tables.conso_path, line_number, len(parts), RXNCONSO_SUPPRESS_INDEX + 1)
                continue

            rows_scanned += 1
            rxcui = normalize_rxcui(parts[RXNCONSO_RXCUI_INDEX])
            if not rxcui:
                invalid_rxcui_rows += 1
                continue

            language = str(parts[RXNCONSO_LAT_INDEX]).strip().upper()
            if language != "ENG":
                non_english_rows += 1
                continue

            suppress = str(parts[RXNCONSO_SUPPRESS_INDEX]).strip().upper()
            if suppress == "Y":
                suppressed_rows += 1
                continue

            sab = str(parts[RXNCONSO_SAB_INDEX]).strip().upper()
            tty = str(parts[RXNCONSO_TTY_INDEX]).strip().upper()
            name = str(parts[RXNCONSO_STR_INDEX]).strip()
            if not name:
                empty_name_rows += 1
                continue

            term_types_by_rxcui[rxcui].add(tty)
            source_abbreviations_by_rxcui[rxcui].add(sab)

            preferred_rank = _preferred_name_rank(sab=sab, tty=tty, name=name)
            current_rank = preferred_name_rank.get(rxcui)
            if current_rank is None or preferred_rank < current_rank:
                preferred_name_rank[rxcui] = preferred_rank
                preferred_name_by_rxcui[rxcui] = name

            if allowed_sabs and sab not in allowed_sabs:
                filtered_sab_rows += 1
                continue

            synonyms_by_rxcui[rxcui].add(name)
            normalized_name = normalize_drug_name(name)
            if not normalized_name:
                continue
            normalized_synonyms_by_rxcui[rxcui].add(normalized_name)

            candidate = _NameCandidate(rxcui=rxcui, sab=sab, tty=tty, name=name)
            current_candidate = name_candidates[normalized_name].get(rxcui)
            if current_candidate is None or _name_candidate_rank(candidate) < _name_candidate_rank(current_candidate):
                name_candidates[normalized_name][rxcui] = candidate

    normalized_name_to_rxcuis: dict[str, tuple[str, ...]] = {}
    for normalized_name, by_rxcui in name_candidates.items():
        normalized_name_to_rxcuis[normalized_name] = tuple(
            candidate.rxcui
            for candidate in sorted(by_rxcui.values(), key=_name_candidate_rank)
        )

    rxcui_to_names: dict[str, RxNormNameRecord] = {}
    for rxcui, preferred_name in preferred_name_by_rxcui.items():
        synonyms = tuple(sorted(synonyms_by_rxcui.get(rxcui, {preferred_name})))
        normalized_synonyms = tuple(
            sorted(
                {
                    normalized
                    for normalized in normalized_synonyms_by_rxcui.get(rxcui, set())
                    if normalized
                }
            )
        )
        rxcui_to_names[rxcui] = RxNormNameRecord(
            preferred_name=preferred_name,
            normalized_preferred_name=normalize_drug_name(preferred_name),
            synonyms=synonyms,
            normalized_synonyms=normalized_synonyms,
            term_types=tuple(sorted(term_types_by_rxcui.get(rxcui, set()))),
            source_abbreviations=tuple(sorted(source_abbreviations_by_rxcui.get(rxcui, set()))),
        )

    meta = {
        "source_path": str(tables.conso_path),
        "rows_scanned": int(rows_scanned),
        "rows_malformed": int(malformed_rows),
        "rows_with_invalid_rxcui": int(invalid_rxcui_rows),
        "rows_non_english": int(non_english_rows),
        "rows_suppressed": int(suppressed_rows),
        "rows_empty_name": int(empty_name_rows),
        "rows_filtered_by_sab": int(filtered_sab_rows),
        "allowed_name_sabs": sorted(allowed_sabs),
        "num_rxcui_entries": int(len(rxcui_to_names)),
        "num_normalized_name_entries": int(len(normalized_name_to_rxcuis)),
    }
    LOGGER.info(
        "Built RxCUI name index with %s RxCUIs and %s normalized names from %s",
        len(rxcui_to_names),
        len(normalized_name_to_rxcuis),
        tables.conso_path,
    )
    return RxNormNameIndex(
        rxcui_to_names=rxcui_to_names,
        normalized_name_to_rxcuis=normalized_name_to_rxcuis,
        meta=meta,
    )


def build_rxcui_ingredient_index(
    tables_or_root: RxNormTables | str | Path,
    *,
    name_index: RxNormNameIndex | None = None,
    ingredient_ttys: Sequence[str] = DEFAULT_INGREDIENT_TTYS,
    ingredient_bridge_relations: Sequence[str] = DEFAULT_INGREDIENT_BRIDGE_RELATIONS,
    max_component_ingredient_count_for_auto_assignment: int = DEFAULT_MAX_COMPONENT_INGREDIENT_COUNT_FOR_AUTO_ASSIGNMENT,
) -> RxNormIngredientIndex:
    """
    Build ingredient-level normalization for RxCUIs using RXNREL.RRF.

    If a RxCUI is already an ingredient term type and no explicit ingredient
    relation is present, the index falls back to mapping that RxCUI to itself.
    """

    tables = _coerce_tables(tables_or_root)
    ingredient_tty_set = {str(value).strip().upper() for value in ingredient_ttys}
    bridge_relations = frozenset(
        str(value).strip().lower()
        for value in ingredient_bridge_relations
        if str(value).strip()
    )
    aui_to_rxcui: dict[str, str] = {}
    rxcui_to_term_types: dict[str, set[str]] = defaultdict(set)
    rxcui_to_preferred_name: dict[str, str] = {}
    preferred_name_rank: dict[str, tuple[int, int, int, str]] = {}
    LOGGER.info("Ingredient index: preload RXNCONSO metadata from %s", tables.conso_path)

    conso_rows = 0
    malformed_conso_rows = 0
    invalid_conso_rxcui_rows = 0
    conso_preload_start = perf_counter()
    with tables.conso_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            parts = _split_rrf_row(raw_line)
            if len(parts) <= RXNCONSO_SUPPRESS_INDEX:
                malformed_conso_rows += 1
                _warn_truncated_row(tables.conso_path, line_number, len(parts), RXNCONSO_SUPPRESS_INDEX + 1)
                continue

            conso_rows += 1
            if conso_rows % RXNORM_PROGRESS_LOG_EVERY == 0:
                LOGGER.info("Ingredient index: RXNCONSO preload scanned %s rows", conso_rows)
            rxcui = normalize_rxcui(parts[RXNCONSO_RXCUI_INDEX])
            if not rxcui:
                invalid_conso_rxcui_rows += 1
                continue

            rxaui = str(parts[RXNCONSO_RXAUI_INDEX]).strip()
            if rxaui:
                aui_to_rxcui[rxaui] = rxcui

            tty = str(parts[RXNCONSO_TTY_INDEX]).strip().upper()
            if tty:
                rxcui_to_term_types[rxcui].add(tty)

            if str(parts[RXNCONSO_LAT_INDEX]).strip().upper() != "ENG":
                continue
            if str(parts[RXNCONSO_SUPPRESS_INDEX]).strip().upper() == "Y":
                continue

            sab = str(parts[RXNCONSO_SAB_INDEX]).strip().upper()
            name = str(parts[RXNCONSO_STR_INDEX]).strip()
            if not name:
                continue
            rank = _preferred_name_rank(sab=sab, tty=tty, name=name)
            current_rank = preferred_name_rank.get(rxcui)
            if current_rank is None or rank < current_rank:
                preferred_name_rank[rxcui] = rank
                rxcui_to_preferred_name[rxcui] = name
    _log_phase_timing(
        "ingredient index: RXNCONSO preload",
        conso_preload_start,
        rows_scanned=conso_rows,
        ingredient_safe_candidates=len(
            {
                rxcui
                for rxcui, term_types in rxcui_to_term_types.items()
                if term_types.intersection(ingredient_tty_set)
            }
        ),
    )

    ingredient_safe_rxcuis = {
        rxcui
        for rxcui, term_types in rxcui_to_term_types.items()
        if term_types.intersection(ingredient_tty_set)
    }
    relation_neighbors: dict[str, set[str]] = defaultdict(set)
    direct_rxcui_to_ingredient_rxcuis: dict[str, set[str]] = defaultdict(set)
    LOGGER.info(
        "Ingredient index: scan RXNREL relations from %s using bridge relations=%s",
        tables.rel_path,
        sorted(bridge_relations),
    )
    rel_rows = 0
    malformed_rel_rows = 0
    invalid_rel_rxcui_rows = 0
    kept_relation_rows = 0
    unsupported_relation_rows = 0
    relation_type_counter: Counter[str] = Counter()
    relation_scan_start = perf_counter()
    with tables.rel_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            parts = _split_rrf_row(raw_line)
            if len(parts) <= RXNREL_SUPPRESS_INDEX:
                malformed_rel_rows += 1
                _warn_truncated_row(tables.rel_path, line_number, len(parts), RXNREL_SUPPRESS_INDEX + 1)
                continue

            rel_rows += 1
            if rel_rows % RXNORM_PROGRESS_LOG_EVERY == 0:
                LOGGER.info(
                    "Ingredient index: RXNREL relation scan processed %s rows (kept=%s, invalid=%s)",
                    rel_rows,
                    kept_relation_rows,
                    invalid_rel_rxcui_rows,
                )
            if str(parts[RXNREL_SUPPRESS_INDEX]).strip().upper() == "Y":
                continue

            rela = str(parts[RXNREL_RELA_INDEX]).strip().lower()
            if rela not in bridge_relations:
                unsupported_relation_rows += 1
                continue

            left_rxcui = normalize_rxcui(parts[RXNREL_RXCUI1_INDEX]) or aui_to_rxcui.get(str(parts[RXNREL_AUI1_INDEX]).strip())
            right_rxcui = normalize_rxcui(parts[RXNREL_RXCUI2_INDEX]) or aui_to_rxcui.get(str(parts[RXNREL_AUI2_INDEX]).strip())
            if not left_rxcui or not right_rxcui or left_rxcui == right_rxcui:
                invalid_rel_rxcui_rows += 1
                continue

            kept_relation_rows += 1
            relation_type_counter[rela] += 1
            relation_neighbors[left_rxcui].add(right_rxcui)
            relation_neighbors[right_rxcui].add(left_rxcui)
            left_is_ingredient = left_rxcui in ingredient_safe_rxcuis
            right_is_ingredient = right_rxcui in ingredient_safe_rxcuis
            if left_is_ingredient and not right_is_ingredient:
                direct_rxcui_to_ingredient_rxcuis[right_rxcui].add(left_rxcui)
            elif right_is_ingredient and not left_is_ingredient:
                direct_rxcui_to_ingredient_rxcuis[left_rxcui].add(right_rxcui)
    _log_phase_timing(
        "ingredient index: RXNREL relation scan",
        relation_scan_start,
        rows_scanned=rel_rows,
        kept_relations=kept_relation_rows,
        relation_edges=sum(len(neighbors) for neighbors in relation_neighbors.values()) // 2,
        neighbor_nodes=len(relation_neighbors),
    )

    self_normalized_count = 0
    for rxcui in ingredient_safe_rxcuis:
        self_normalized_count += 1

    name_lookup = (
        {rxcui: record.preferred_name for rxcui, record in name_index.rxcui_to_names.items()}
        if name_index is not None
        else rxcui_to_preferred_name
    )
    valid_named_ingredient_rxcuis = {
        rxcui
        for rxcui in ingredient_safe_rxcuis
        if str(name_lookup.get(rxcui, "")).strip()
    }
    suspicious_ingredient_name_examples = [
        {
            "ingredient_rxcui": rxcui,
            "ingredient_name": str(name_lookup.get(rxcui, "")).strip(),
            "term_types": sorted(rxcui_to_term_types.get(rxcui, set())),
        }
        for rxcui in sorted(valid_named_ingredient_rxcuis)
        if is_suspicious_ingredient_name(name_lookup.get(rxcui, ""))
    ]
    valid_named_ingredient_rxcuis = {
        rxcui
        for rxcui in valid_named_ingredient_rxcuis
        if not is_suspicious_ingredient_name(name_lookup.get(rxcui, ""))
    }

    rxcui_to_ingredient_rxcuis: dict[str, set[str]] = defaultdict(set)
    LOGGER.info(
        "Ingredient index: collapse ingredient mappings by connected component across %s relation nodes",
        len(relation_neighbors),
    )
    propagation_start = perf_counter()
    visited_relation_nodes: set[str] = set()
    visited_relation_node_count = 0
    component_count = 0
    components_with_ingredients = 0
    assigned_rxcui_count = 0
    direct_assignment_fallback_count = 0
    max_component_size = 0
    max_component_ingredient_count = 0
    suspicious_component_count = 0
    component_summaries: list[dict[str, Any]] = []
    for start_rxcui in relation_neighbors:
        if start_rxcui in visited_relation_nodes:
            continue
        component_count += 1
        stack = [start_rxcui]
        component_nodes: list[str] = []
        component_ingredients: set[str] = set()
        while stack:
            current_rxcui = stack.pop()
            if current_rxcui in visited_relation_nodes:
                continue
            visited_relation_nodes.add(current_rxcui)
            visited_relation_node_count += 1
            if visited_relation_node_count % 100_000 == 0:
                LOGGER.info(
                    "Ingredient index: component collapse visited %s relation nodes across %s components",
                    visited_relation_node_count,
                    component_count,
                )
            component_nodes.append(current_rxcui)
            if current_rxcui in valid_named_ingredient_rxcuis:
                component_ingredients.add(current_rxcui)
            for neighbor in relation_neighbors.get(current_rxcui, ()):
                if neighbor not in visited_relation_nodes:
                    stack.append(neighbor)

        component_size = len(component_nodes)
        ingredient_count = len(component_ingredients)
        max_component_size = max(max_component_size, component_size)
        component_summary = {
            "seed_rxcui": start_rxcui,
            "seed_name": str(name_lookup.get(start_rxcui, "")).strip(),
            "component_size": int(component_size),
            "ingredient_count": int(ingredient_count),
            "sample_ingredient_names": [
                str(name_lookup.get(ingredient_rxcui, "")).strip()
                for ingredient_rxcui in sorted(component_ingredients)[:10]
            ],
            "sample_node_names": [
                str(name_lookup.get(node_rxcui, "")).strip()
                for node_rxcui in component_nodes[:10]
                if str(name_lookup.get(node_rxcui, "")).strip()
            ],
        }
        component_summaries.append(component_summary)
        if not component_ingredients:
            continue

        sorted_component_ingredients = tuple(sorted(component_ingredients))
        components_with_ingredients += 1
        max_component_ingredient_count = max(
            max_component_ingredient_count,
            len(sorted_component_ingredients),
        )
        if ingredient_count > max_component_ingredient_count_for_auto_assignment:
            suspicious_component_count += 1
            for current_rxcui in component_nodes:
                if current_rxcui in valid_named_ingredient_rxcuis:
                    rxcui_to_ingredient_rxcuis[current_rxcui].add(current_rxcui)
                    assigned_rxcui_count += 1
                    continue
                direct_ingredients = {
                    ingredient_rxcui
                    for ingredient_rxcui in direct_rxcui_to_ingredient_rxcuis.get(current_rxcui, set())
                    if ingredient_rxcui in valid_named_ingredient_rxcuis
                }
                if not direct_ingredients:
                    continue
                rxcui_to_ingredient_rxcuis[current_rxcui].update(direct_ingredients)
                assigned_rxcui_count += 1
                direct_assignment_fallback_count += 1
            continue

        assigned_rxcui_count += component_size
        for current_rxcui in component_nodes:
            rxcui_to_ingredient_rxcuis[current_rxcui].update(sorted_component_ingredients)

    for ingredient_rxcui in valid_named_ingredient_rxcuis:
        rxcui_to_ingredient_rxcuis[ingredient_rxcui].add(ingredient_rxcui)

    component_collapse_stats = {
        "mode": "connected_components",
        "component_count": int(component_count),
        "components_with_ingredients": int(components_with_ingredients),
        "visited_relation_nodes": int(visited_relation_node_count),
        "assigned_rxcui_count": int(assigned_rxcui_count),
        "direct_assignment_fallback_count": int(direct_assignment_fallback_count),
        "suspicious_component_count": int(suspicious_component_count),
        "max_component_size": int(max_component_size),
        "max_component_ingredient_count": int(max_component_ingredient_count),
        "max_component_ingredient_count_for_auto_assignment": int(
            max_component_ingredient_count_for_auto_assignment
        ),
    }
    top_largest_components = sorted(
        component_summaries,
        key=lambda item: (-int(item["component_size"]), -int(item["ingredient_count"]), str(item["seed_rxcui"])),
    )[:10]
    top_components_by_ingredient_count = sorted(
        component_summaries,
        key=lambda item: (-int(item["ingredient_count"]), -int(item["component_size"]), str(item["seed_rxcui"])),
    )[:10]
    suspicious_component_examples = [
        summary
        for summary in top_components_by_ingredient_count
        if int(summary["ingredient_count"]) > max_component_ingredient_count_for_auto_assignment
    ][:10]
    LOGGER.info(
        "Ingredient index: component collapse assigned ingredient sets to %s RxCUIs across %s components with ingredients (suspicious_components=%s)",
        assigned_rxcui_count,
        components_with_ingredients,
        suspicious_component_count,
    )
    _log_phase_timing(
        "ingredient index: propagate mappings",
        propagation_start,
        mode=component_collapse_stats["mode"],
        components=component_collapse_stats["component_count"],
        assigned_rxcuis=component_collapse_stats["assigned_rxcui_count"],
    )

    LOGGER.info("Ingredient index: finalize ingredient names and metadata")
    finalize_start = perf_counter()
    LOGGER.info("Ingredient index: finalize filtered RxCUI -> ingredient assignments")
    finalized_rxcui_to_ingredient_rxcuis: dict[str, tuple[str, ...]] = {}
    used_ingredient_rxcuis: set[str] = set()
    for rxcui, ingredient_rxcuis in rxcui_to_ingredient_rxcuis.items():
        filtered_ingredient_rxcuis = tuple(
            sorted(
                ingredient_rxcui
                for ingredient_rxcui in ingredient_rxcuis
                if ingredient_rxcui in valid_named_ingredient_rxcuis
            )
        )
        if not filtered_ingredient_rxcuis:
            continue
        finalized_rxcui_to_ingredient_rxcuis[rxcui] = filtered_ingredient_rxcuis
        used_ingredient_rxcuis.update(filtered_ingredient_rxcuis)

    LOGGER.info("Ingredient index: finalize ingredient name lookup for %s used ingredients", len(used_ingredient_rxcuis))
    ingredient_rxcui_to_name = {
        ingredient_rxcui: str(name_lookup.get(ingredient_rxcui, "")).strip()
        for ingredient_rxcui in sorted(used_ingredient_rxcuis)
    }
    LOGGER.info("Ingredient index: finalize RxCUI -> ingredient names for %s mapped RxCUIs", len(finalized_rxcui_to_ingredient_rxcuis))
    rxcui_to_ingredient_names = {
        rxcui: tuple(
            sorted(
                {
                    ingredient_rxcui_to_name[ingredient_rxcui]
                    for ingredient_rxcui in ingredient_rxcuis
                    if ingredient_rxcui in ingredient_rxcui_to_name
                }
            )
        )
        for rxcui, ingredient_rxcuis in finalized_rxcui_to_ingredient_rxcuis.items()
    }

    LOGGER.info("Ingredient index: finalize sample product -> ingredient mappings")
    sample_product_rxcuis = heapq.nsmallest(
        20,
        (
            rxcui
            for rxcui in finalized_rxcui_to_ingredient_rxcuis
            if rxcui not in ingredient_safe_rxcuis
        ),
    )
    sample_product_to_ingredient_mappings: list[dict[str, Any]] = []
    for rxcui in sample_product_rxcuis:
        ingredient_rxcuis = finalized_rxcui_to_ingredient_rxcuis[rxcui]
        sample_product_to_ingredient_mappings.append(
            {
                "source_rxcui": rxcui,
                "source_name": str(name_lookup.get(rxcui, "")).strip(),
                "source_term_types": sorted(rxcui_to_term_types.get(rxcui, set())),
                "ingredient_rxcuis": list(ingredient_rxcuis),
                "ingredient_names": [
                    ingredient_rxcui_to_name[ingredient_rxcui]
                    for ingredient_rxcui in ingredient_rxcuis
                    if ingredient_rxcui in ingredient_rxcui_to_name
                ],
            }
        )
    _log_phase_timing(
        "ingredient index: finalize metadata",
        finalize_start,
        named_ingredients=len(ingredient_rxcui_to_name),
        mapped_rxcuis=len(finalized_rxcui_to_ingredient_rxcuis),
    )
    meta = {
        "conso_path": str(tables.conso_path),
        "rel_path": str(tables.rel_path),
        "conso_rows_scanned": int(conso_rows),
        "conso_rows_malformed": int(malformed_conso_rows),
        "conso_rows_with_invalid_rxcui": int(invalid_conso_rxcui_rows),
        "rel_rows_scanned": int(rel_rows),
        "rel_rows_malformed": int(malformed_rel_rows),
        "rel_rows_with_invalid_rxcui": int(invalid_rel_rxcui_rows),
        "rel_rows_kept": int(kept_relation_rows),
        "rel_rows_unsupported_relation": int(unsupported_relation_rows),
        "ingredient_bridge_relations_used": sorted(bridge_relations),
        "neighbor_nodes_after_filter": int(len(relation_neighbors)),
        "relation_edge_count_after_filter": int(
            sum(len(neighbors) for neighbors in relation_neighbors.values()) // 2
        ),
        "relation_type_counter": {
            relation: int(count)
            for relation, count in sorted(relation_type_counter.items())
        },
        "self_normalized_ingredient_count": int(self_normalized_count),
        "propagation_pass_updates": [],
        "component_collapse_stats": component_collapse_stats,
        "top_largest_components": top_largest_components,
        "top_components_by_ingredient_count": top_components_by_ingredient_count,
        "suspicious_component_count": int(suspicious_component_count),
        "suspicious_component_examples": suspicious_component_examples,
        "num_product_rxcui_entries": int(len(finalized_rxcui_to_ingredient_rxcuis)),
        "num_named_ingredient_rxcuis": int(len(ingredient_rxcui_to_name)),
        "num_ingredient_safe_rxcuis": int(len(ingredient_safe_rxcuis)),
        "sample_product_to_ingredient_mappings": sample_product_to_ingredient_mappings,
        "sample_suspicious_ingredient_name_candidates": suspicious_ingredient_name_examples[:20],
    }
    LOGGER.info(
        "Built RxCUI ingredient index with %s RxCUIs mapped to %s named ingredient concepts from %s",
        len(finalized_rxcui_to_ingredient_rxcuis),
        len(ingredient_rxcui_to_name),
        tables.rel_path,
    )
    return RxNormIngredientIndex(
        rxcui_to_ingredient_rxcuis=finalized_rxcui_to_ingredient_rxcuis,
        rxcui_to_ingredient_names=rxcui_to_ingredient_names,
        ingredient_rxcui_to_name=ingredient_rxcui_to_name,
        ingredient_safe_rxcuis=tuple(sorted(ingredient_safe_rxcuis)),
        meta=meta,
    )


def build_minimal_rxnorm_index(
    tables_or_root: RxNormTables | str | Path,
    *,
    use_cache: bool = True,
    force_rebuild: bool = False,
    cache_dir: str | Path | None = None,
) -> RxNormMinimalIndex:
    """Convenience helper that builds or loads the minimal RxNorm indexes."""

    build_start = perf_counter()
    resolve_start = perf_counter()
    tables = _coerce_tables(tables_or_root)
    _log_phase_timing("resolve tables", resolve_start, root=tables.root)

    if use_cache and not force_rebuild:
        cached_index = load_cached_minimal_rxnorm_index(tables, cache_dir=cache_dir)
        if cached_index is not None:
            LOGGER.info(
                "RxNorm minimal index ready from cache in %.2fs",
                perf_counter() - build_start,
            )
            return cached_index
    elif force_rebuild:
        LOGGER.info("RxNorm cache bypassed because force_rebuild=True")

    if use_cache:
        LOGGER.info("RxNorm cache miss; rebuilding minimal index")
    else:
        LOGGER.info("RxNorm cache disabled; rebuilding minimal index")

    ndc_start = perf_counter()
    ndc_index = build_ndc_to_rxcui_map(tables)
    _log_phase_timing("RXNSAT ndc map", ndc_start, entries=len(ndc_index.ndc_to_rxcui))

    name_start = perf_counter()
    name_index = build_rxcui_name_index(tables)
    _log_phase_timing("RXNCONSO name index", name_start, rxcuis=len(name_index.rxcui_to_names))

    ingredient_start = perf_counter()
    ingredient_index = build_rxcui_ingredient_index(tables, name_index=name_index)
    _log_phase_timing(
        "RXNREL ingredient index",
        ingredient_start,
        mapped_rxcuis=len(ingredient_index.rxcui_to_ingredient_rxcuis),
        named_ingredients=len(ingredient_index.ingredient_rxcui_to_name),
    )

    minimal_index = RxNormMinimalIndex(
        tables=tables,
        ndc_index=ndc_index,
        name_index=name_index,
        ingredient_index=ingredient_index,
        meta={
            "rxnorm_root": str(tables.root),
            "conso_path": str(tables.conso_path),
            "rel_path": str(tables.rel_path),
            "sat_path": str(tables.sat_path),
            "ndc_entries": int(len(ndc_index.ndc_to_rxcui)),
            "named_rxcui_entries": int(len(name_index.rxcui_to_names)),
            "ingredient_entries": int(len(ingredient_index.rxcui_to_ingredient_rxcuis)),
        },
    )

    if use_cache:
        save_start = perf_counter()
        save_cached_minimal_rxnorm_index(minimal_index, cache_dir=cache_dir)
        _log_phase_timing("save cache", save_start)

    LOGGER.info("Built RxNorm minimal index in %.2fs", perf_counter() - build_start)
    return minimal_index


def compute_rxnorm_cache_fingerprint(
    tables_or_root: RxNormTables | str | Path,
) -> dict[str, Any]:
    """Build a stable fingerprint for cache invalidation."""

    tables = _coerce_tables(tables_or_root)
    sources = [tables.sat_path, tables.conso_path, tables.rel_path]
    return {
        "cache_version": 3,
        "rxnorm_root": str(tables.root.resolve(strict=False)),
        "sources": [
            {
                "path": str(path.resolve(strict=False)),
                "size": int(path.stat().st_size),
                "mtime": float(path.stat().st_mtime),
            }
            for path in sources
        ],
    }


def load_cached_minimal_rxnorm_index(
    tables_or_root: RxNormTables | str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> RxNormMinimalIndex | None:
    """Load a cached minimal RxNorm index when fingerprint matches."""

    tables = _coerce_tables(tables_or_root)
    cache_path, _ = _resolve_rxnorm_cache_paths(tables, cache_dir=cache_dir)
    if not cache_path.exists():
        LOGGER.info("RxNorm cache miss; no cache file at %s", cache_path)
        return None

    expected_fingerprint = compute_rxnorm_cache_fingerprint(tables)
    load_start = perf_counter()
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError) as exc:
        LOGGER.warning(
            "RxNorm cache at %s is unreadable or corrupt; rebuilding from source. Error: %s",
            cache_path,
            exc,
        )
        return None

    cached_fingerprint = payload.get("fingerprint")
    cached_index = payload.get("index")
    if cached_fingerprint != expected_fingerprint:
        LOGGER.info("RxNorm cache stale at %s; rebuilding minimal index", cache_path)
        return None
    if not isinstance(cached_index, RxNormMinimalIndex):
        LOGGER.warning(
            "RxNorm cache at %s does not contain a valid RxNormMinimalIndex; rebuilding from source.",
            cache_path,
        )
        return None

    LOGGER.info("RxNorm cache hit at %s", cache_path)
    LOGGER.info("Loaded cached RxNorm minimal index in %.2fs", perf_counter() - load_start)
    return cached_index


def save_cached_minimal_rxnorm_index(
    minimal_index: RxNormMinimalIndex,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Persist a minimal RxNorm index to disk with a fingerprint sidecar."""

    cache_path, metadata_path = _resolve_rxnorm_cache_paths(minimal_index.tables, cache_dir=cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = compute_rxnorm_cache_fingerprint(minimal_index.tables)
    payload = {
        "fingerprint": fingerprint,
        "index": minimal_index,
    }
    tmp_cache_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_cache_path.replace(cache_path)

    metadata_payload = {
        "cache_path": str(cache_path.resolve(strict=False)),
        "fingerprint": fingerprint,
    }
    tmp_metadata_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with tmp_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2, ensure_ascii=False)
    tmp_metadata_path.replace(metadata_path)

    LOGGER.info("Saved RxNorm cache to %s", cache_path)
    return cache_path


def _coerce_tables(tables_or_root: RxNormTables | str | Path) -> RxNormTables:
    if isinstance(tables_or_root, RxNormTables):
        return tables_or_root
    return load_rxnorm_tables(tables_or_root)


def _split_rrf_row(line: str) -> list[str]:
    return line.rstrip("\n").split("|")


def _resolve_rxnorm_cache_paths(
    tables: RxNormTables,
    *,
    cache_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else tables.root / "cache"
    cache_path = resolved_cache_dir / "rxnorm_minimal_index.pkl"
    metadata_path = resolved_cache_dir / "rxnorm_minimal_index.meta.json"
    return cache_path, metadata_path


def _log_phase_timing(phase_name: str, start_time: float, **stats: Any) -> None:
    elapsed = perf_counter() - start_time
    stat_text = ", ".join(f"{key}={value}" for key, value in stats.items())
    if stat_text:
        LOGGER.info("Phase `%s` finished in %.2fs (%s)", phase_name, elapsed, stat_text)
    else:
        LOGGER.info("Phase `%s` finished in %.2fs", phase_name, elapsed)


def _warn_truncated_row(path: Path, line_number: int, actual_columns: int, required_columns: int) -> None:
    LOGGER.debug(
        "Skipping truncated RRF row in %s at line %s (actual_columns=%s, required_columns=%s)",
        path,
        line_number,
        actual_columns,
        required_columns,
    )


def _sab_priority(value: str, mapping: Mapping[str, int], default: int = 99) -> int:
    return int(mapping.get(str(value or "").strip().upper(), default))


def _suppression_rank(value: str) -> int:
    normalized = str(value or "").strip().upper()
    return 0 if normalized in {"", "N"} else 1


def _tty_priority(value: str) -> int:
    return int(TTY_PRIORITY.get(str(value or "").strip().upper(), 99))


def _preferred_name_rank(*, sab: str, tty: str, name: str) -> tuple[int, int, int, str]:
    return (
        _sab_priority(sab, {"RXNORM": 0, "MTHSPL": 1}, default=9),
        _tty_priority(tty),
        len(str(name or "")),
        str(name or ""),
    )


def _name_candidate_rank(candidate: _NameCandidate) -> tuple[int, int, int, str]:
    return (
        _sab_priority(candidate.sab, {"RXNORM": 0, "MTHSPL": 1}, default=9),
        _tty_priority(candidate.tty),
        len(candidate.name),
        candidate.rxcui,
    )
