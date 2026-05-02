from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data.load_mimic import iter_csv_rows
from src.data.rxnorm_utils import (
    RxNormMinimalIndex,
    build_minimal_rxnorm_index,
    is_suspicious_ingredient_name,
    normalize_drug_name,
    normalize_ndc,
)
from src.features.medication_history import extract_medication_token
from src.utils.io import parse_int


LOGGER = logging.getLogger(__name__)

MAX_CANONICAL_INGREDIENTS_PER_MATCH = 8
MIN_PREFIX_MATCH_TERM_LENGTH = 6
LEXICAL_STOPWORDS = frozenset(
    {
        "MG",
        "MCG",
        "G",
        "ML",
        "MEQ",
        "UNT",
        "UNITS",
        "PERCENT",
        "ORAL",
        "TABLET",
        "CAPSULE",
        "SOLUTION",
        "SUSPENSION",
        "PATCH",
        "INJECTION",
        "IV",
        "PO",
        "PF",
        "XL",
        "XR",
        "ER",
        "DR",
        "EC",
        "HCL",
        "HBR",
        "NA",
        "USP",
        "W",
    }
)


@dataclass(frozen=True)
class MedicationNormalizationLookup:
    """Reusable RxNorm-backed lookup for canonical medication normalization."""

    rxnorm_index: RxNormMinimalIndex
    rxcui_to_preferred_name: dict[str, str]
    rxcui_to_term_types: dict[str, tuple[str, ...]]
    rxcui_to_ingredient_rxcuis: dict[str, tuple[str, ...]]
    ingredient_rxcui_to_name: dict[str, str]
    ingredient_safe_rxcuis: frozenset[str]


@dataclass(frozen=True)
class CanonicalMedicationMatch:
    """Normalization result for one raw medication token."""

    raw_token: str
    raw_token_body: str
    resolved_product_rxcui: str | None
    canonical_rxcuis: tuple[str, ...]
    canonical_tokens: tuple[str, ...]
    canonical_names: tuple[str, ...]
    match_source: str | None
    ambiguous_candidate_count: int


@dataclass(frozen=True)
class PrescriptionNormalizationEvidence:
    """One-pass prescription scan outputs reused during medication normalization."""

    token_counter: Counter[str]
    token_to_rxcui_counts: dict[str, Counter[str]]
    rows_scanned: int
    rows_with_target_token: int
    rows_with_non_empty_ndc: int
    rows_with_normalized_ndc: int
    rows_with_mapped_ndc: int
    sample_unmatched_ndc_examples: tuple[dict[str, str], ...]


def canonical_medication_token(rxcui: str) -> str:
    """Build the stable benchmark medication token for a canonical RxNorm concept."""

    return f"MEDRX:{str(rxcui).strip()}"


def medication_token_body(token: str) -> str:
    """Strip the medication token prefix when present."""

    return str(token).split(":", 1)[1] if ":" in str(token) else str(token)


def build_medication_normalization_lookup(
    rxnorm_root: str | Path,
    *,
    use_cache: bool = True,
    force_rebuild: bool = False,
    cache_dir: str | Path | None = None,
) -> MedicationNormalizationLookup:
    """Load the minimal RxNorm indexes needed for benchmark medication normalization."""

    rxnorm_index = build_minimal_rxnorm_index(
        rxnorm_root,
        use_cache=use_cache,
        force_rebuild=force_rebuild,
        cache_dir=cache_dir,
    )
    rxcui_to_preferred_name = {
        rxcui: record.preferred_name
        for rxcui, record in rxnorm_index.name_index.rxcui_to_names.items()
    }
    rxcui_to_term_types = {
        rxcui: tuple(record.term_types)
        for rxcui, record in rxnorm_index.name_index.rxcui_to_names.items()
    }
    return MedicationNormalizationLookup(
        rxnorm_index=rxnorm_index,
        rxcui_to_preferred_name=rxcui_to_preferred_name,
        rxcui_to_term_types=rxcui_to_term_types,
        rxcui_to_ingredient_rxcuis=dict(rxnorm_index.ingredient_index.rxcui_to_ingredient_rxcuis),
        ingredient_rxcui_to_name=dict(rxnorm_index.ingredient_index.ingredient_rxcui_to_name),
        ingredient_safe_rxcuis=frozenset(rxnorm_index.ingredient_index.ingredient_safe_rxcuis),
    )


def build_prescription_token_normalization_map(
    prescriptions_path: str | Path,
    *,
    lookup: MedicationNormalizationLookup,
    raw_tokens: Sequence[str],
    hadm_ids: set[int] | None = None,
    evidence: PrescriptionNormalizationEvidence | None = None,
) -> tuple[dict[str, CanonicalMedicationMatch], dict[str, Any]]:
    """
    Normalize raw prescription tokens to canonical medication tokens.

    Resolution order:
    1. Row-level NDC -> product RxCUI when available.
    2. Direct RxNorm name lookup on the raw token body.
    3. Ingredient-level fallback from the resolved product RxCUI.
    """

    target_tokens = [str(token) for token in raw_tokens if str(token).strip()]
    if not target_tokens:
        return {}, {
            "raw_row_count": 0,
            "unique_raw_token_count": 0,
            "matched_rxcui_unique_token_count": 0,
            "matched_canonical_token_unique_token_count": 0,
            "unmatched_top_examples": [],
        }

    target_token_set = set(target_tokens)
    if evidence is None:
        token_counter: Counter[str] = Counter()
        token_to_rxcui_counts: dict[str, Counter[str]] = defaultdict(Counter)
        rows_scanned = 0
        rows_with_target_token = 0
        rows_with_non_empty_ndc = 0
        rows_with_normalized_ndc = 0
        rows_with_mapped_ndc = 0
        unmatched_ndc_examples: list[dict[str, str]] = []

        for row in iter_csv_rows(
            prescriptions_path,
            fields=("hadm_id", "drug", "formulary_drug_cd", "ndc"),
        ):
            parsed_hadm_id = parse_int(row.get("hadm_id"))
            if hadm_ids is not None and parsed_hadm_id not in hadm_ids:
                continue

            rows_scanned += 1
            raw_token = extract_medication_token(row)
            if not raw_token or raw_token not in target_token_set:
                continue

            rows_with_target_token += 1
            token_counter[raw_token] += 1
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
                            "raw_token": raw_token,
                            "raw_ndc": raw_ndc,
                            "normalized_ndc": ndc,
                            "drug": str(row.get("drug", "")).strip(),
                            "formulary_drug_cd": str(row.get("formulary_drug_cd", "")).strip(),
                        }
                    )
                continue
            rows_with_mapped_ndc += 1
            token_to_rxcui_counts[raw_token][resolved_rxcui] += 1
    else:
        token_counter = Counter(
            {
                token: int(evidence.token_counter.get(token, 0))
                for token in target_tokens
                if int(evidence.token_counter.get(token, 0)) > 0
            }
        )
        token_to_rxcui_counts = {
            token: Counter(counter)
            for token in target_tokens
            if (counter := evidence.token_to_rxcui_counts.get(token))
        }
        rows_scanned = int(evidence.rows_scanned)
        rows_with_target_token = int(evidence.rows_with_target_token)
        rows_with_non_empty_ndc = int(evidence.rows_with_non_empty_ndc)
        rows_with_normalized_ndc = int(evidence.rows_with_normalized_ndc)
        rows_with_mapped_ndc = int(evidence.rows_with_mapped_ndc)
        unmatched_ndc_examples = list(evidence.sample_unmatched_ndc_examples)

    token_matches: dict[str, CanonicalMedicationMatch] = {}
    match_source_counter: Counter[str] = Counter()
    unmatched_examples: list[dict[str, Any]] = []
    matched_rxcui_unique_token_count = 0
    matched_canonical_unique_token_count = 0
    ambiguous_name_match_count = 0
    suspicious_name_match_examples: list[dict[str, Any]] = []
    suspicious_ndc_match_examples: list[dict[str, Any]] = []

    for raw_token in target_tokens:
        raw_token_body = medication_token_body(raw_token)
        resolved_product_rxcui: str | None = None
        ambiguous_candidate_count = 0
        match_source: str | None = None
        canonical_rxcuis: tuple[str, ...] = ()

        normalized_name = str(raw_token_body or "").strip().upper()
        candidate_rxcuis = lookup.rxnorm_index.name_index.normalized_name_to_rxcuis.get(normalized_name, ())
        if candidate_rxcuis:
            canonical_rxcuis, resolved_product_rxcui, suspicious_name_examples = _resolve_name_candidate_canonical_rxcuis(
                raw_token=raw_token,
                raw_token_body=raw_token_body,
                candidate_rxcuis=candidate_rxcuis,
                lookup=lookup,
            )
            for example in suspicious_name_examples:
                if len(suspicious_name_match_examples) >= 20:
                    break
                suspicious_name_match_examples.append(example)
            ambiguous_candidate_count = max(len(candidate_rxcuis) - 1, 0)
            if canonical_rxcuis:
                match_source = "rxnorm_name"

        rxcui_counts = token_to_rxcui_counts.get(raw_token)
        if not canonical_rxcuis and rxcui_counts:
            ranked = sorted(rxcui_counts.items(), key=lambda item: (-int(item[1]), item[0]))
            resolved_product_rxcui = str(ranked[0][0])
            ambiguous_candidate_count = max(len(ranked) - 1, 0)
            canonical_rxcuis, suspicious_reason = _resolve_candidate_canonical_rxcuis(
                raw_token=raw_token,
                raw_token_body=raw_token_body,
                resolved_product_rxcui=resolved_product_rxcui,
                source="prescriptions_ndc",
                lookup=lookup,
            )
            if suspicious_reason and len(suspicious_ndc_match_examples) < 20:
                suspicious_ndc_match_examples.append(
                    _build_suspicious_match_example(
                        raw_token=raw_token,
                        raw_token_body=raw_token_body,
                        resolved_product_rxcui=resolved_product_rxcui,
                        candidate_canonical_rxcuis=_resolve_canonical_rxcuis(
                            resolved_product_rxcui,
                            lookup=lookup,
                        ),
                        lookup=lookup,
                        reason=suspicious_reason,
                        source="prescriptions_ndc",
                    )
                )
            if canonical_rxcuis:
                match_source = "prescriptions_ndc"
        elif not canonical_rxcuis and candidate_rxcuis:
            resolved_product_rxcui = str(candidate_rxcuis[0])
            ambiguous_candidate_count = max(len(candidate_rxcuis) - 1, 0)

        if ambiguous_candidate_count > 0:
            ambiguous_name_match_count += 1

        canonical_names = tuple(
            _resolve_canonical_name(canonical_rxcui, lookup=lookup) or ""
            for canonical_rxcui in canonical_rxcuis
        )
        canonical_tokens = tuple(
            canonical_medication_token(canonical_rxcui)
            for canonical_rxcui in canonical_rxcuis
        )
        if resolved_product_rxcui:
            matched_rxcui_unique_token_count += 1
        if canonical_tokens:
            matched_canonical_unique_token_count += 1
        if match_source:
            match_source_counter[match_source] += 1
        elif len(unmatched_examples) < 30:
            unmatched_examples.append(
                {
                    "raw_token": raw_token,
                    "raw_token_body": raw_token_body,
                    "count": int(token_counter.get(raw_token, 0)),
                }
            )

        token_matches[raw_token] = CanonicalMedicationMatch(
            raw_token=raw_token,
            raw_token_body=raw_token_body,
            resolved_product_rxcui=resolved_product_rxcui,
            canonical_rxcuis=canonical_rxcuis,
            canonical_tokens=canonical_tokens,
            canonical_names=canonical_names,
            match_source=match_source,
            ambiguous_candidate_count=int(ambiguous_candidate_count),
        )

    unmatched_examples.sort(key=lambda item: (-int(item["count"]), str(item["raw_token"])))
    report = {
        "raw_row_count": int(sum(token_counter.values())),
        "rows_scanned": int(rows_scanned),
        "rows_with_target_token": int(rows_with_target_token),
        "rows_with_non_empty_ndc": int(rows_with_non_empty_ndc),
        "rows_with_normalized_ndc": int(rows_with_normalized_ndc),
        "rows_with_mapped_ndc": int(rows_with_mapped_ndc),
        "unique_raw_token_count": int(len(token_counter)),
        "matched_rxcui_unique_token_count": int(matched_rxcui_unique_token_count),
        "matched_canonical_token_unique_token_count": int(matched_canonical_unique_token_count),
        "ambiguous_name_match_count": int(ambiguous_name_match_count),
        "suspicious_name_match_examples": suspicious_name_match_examples,
        "suspicious_ndc_match_examples": suspicious_ndc_match_examples,
        "match_source_counter": {key: int(value) for key, value in sorted(match_source_counter.items())},
        "unmatched_top_examples": unmatched_examples[:20],
        "sample_unmatched_ndc_examples": unmatched_ndc_examples[:20],
        "rxnorm_meta": dict(lookup.rxnorm_index.meta),
        "rxnorm_ingredient_meta": dict(lookup.rxnorm_index.ingredient_index.meta),
    }
    LOGGER.info(
        "Built prescription medication normalization map for %s raw tokens "
        "(matched_rxcui=%s, matched_canonical=%s)",
        len(token_counter),
        matched_rxcui_unique_token_count,
        matched_canonical_unique_token_count,
    )
    return token_matches, report


def _resolve_canonical_rxcuis(
    resolved_product_rxcui: str | None,
    *,
    lookup: MedicationNormalizationLookup,
) -> tuple[str, ...]:
    if not resolved_product_rxcui:
        return ()

    if _is_valid_ingredient_rxcui(resolved_product_rxcui, lookup=lookup):
        return (resolved_product_rxcui,)

    ingredient_rxcuis = tuple(
        sorted(
            ingredient_rxcui
            for ingredient_rxcui in lookup.rxcui_to_ingredient_rxcuis.get(resolved_product_rxcui, ())
            if _is_valid_ingredient_rxcui(ingredient_rxcui, lookup=lookup)
        )
    )
    return ingredient_rxcuis


def _resolve_canonical_name(
    canonical_rxcui: str,
    *,
    lookup: MedicationNormalizationLookup,
) -> str | None:
    ingredient_name = str(lookup.ingredient_rxcui_to_name.get(canonical_rxcui, "")).strip()
    if ingredient_name:
        return ingredient_name
    preferred_name = str(lookup.rxcui_to_preferred_name.get(canonical_rxcui, "")).strip()
    if preferred_name and not is_suspicious_ingredient_name(preferred_name):
        return preferred_name
    return None


def _resolve_name_candidate_canonical_rxcuis(
    raw_token: str,
    raw_token_body: str,
    candidate_rxcuis: Sequence[str],
    *,
    lookup: MedicationNormalizationLookup,
) -> tuple[tuple[str, ...], str | None, list[dict[str, Any]]]:
    suspicious_examples: list[dict[str, Any]] = []
    for candidate_rxcui in candidate_rxcuis:
        canonical_rxcuis, suspicious_reason = _resolve_candidate_canonical_rxcuis(
            raw_token=raw_token,
            raw_token_body=raw_token_body,
            resolved_product_rxcui=str(candidate_rxcui),
            source="rxnorm_name",
            lookup=lookup,
        )
        if canonical_rxcuis:
            return canonical_rxcuis, str(candidate_rxcui), suspicious_examples
        if suspicious_reason and len(suspicious_examples) < 20:
            suspicious_examples.append(
                _build_suspicious_match_example(
                    raw_token=raw_token,
                    raw_token_body=raw_token_body,
                    resolved_product_rxcui=str(candidate_rxcui),
                    candidate_canonical_rxcuis=_resolve_canonical_rxcuis(
                        str(candidate_rxcui),
                        lookup=lookup,
                    ),
                    lookup=lookup,
                    reason=suspicious_reason,
                    source="rxnorm_name",
                )
            )
    if candidate_rxcuis:
        return (), str(candidate_rxcuis[0]), suspicious_examples
    return (), None, suspicious_examples


def _resolve_candidate_canonical_rxcuis(
    *,
    raw_token: str,
    raw_token_body: str,
    resolved_product_rxcui: str,
    source: str,
    lookup: MedicationNormalizationLookup,
) -> tuple[tuple[str, ...], str | None]:
    candidate_canonical_rxcuis = _resolve_canonical_rxcuis(resolved_product_rxcui, lookup=lookup)
    if not candidate_canonical_rxcuis:
        return (), None

    if len(candidate_canonical_rxcuis) > MAX_CANONICAL_INGREDIENTS_PER_MATCH:
        plausible_rxcuis = _filter_plausible_canonical_rxcuis(
            raw_token_body,
            candidate_canonical_rxcuis,
            lookup=lookup,
        )
        if plausible_rxcuis:
            return plausible_rxcuis, (
                f"{source}_large_ingredient_set_filtered:{len(candidate_canonical_rxcuis)}"
            )
        return (), f"{source}_large_ingredient_set_rejected:{len(candidate_canonical_rxcuis)}"

    if source == "rxnorm_name":
        plausible_rxcuis = _filter_plausible_canonical_rxcuis(
            raw_token_body,
            candidate_canonical_rxcuis,
            lookup=lookup,
        )
        if plausible_rxcuis:
            return plausible_rxcuis, None
        if _raw_matches_rxcui_name(raw_token_body, resolved_product_rxcui, lookup=lookup):
            return candidate_canonical_rxcuis, None
        return (), f"{source}_lexical_mismatch"

    if str(raw_token).startswith("CODE:"):
        return candidate_canonical_rxcuis, None

    plausible_rxcuis = _filter_plausible_canonical_rxcuis(
        raw_token_body,
        candidate_canonical_rxcuis,
        lookup=lookup,
    )
    if plausible_rxcuis:
        return plausible_rxcuis, None
    if _raw_matches_rxcui_name(raw_token_body, resolved_product_rxcui, lookup=lookup):
        return candidate_canonical_rxcuis, None
    return (), f"{source}_lexical_mismatch"


def _is_valid_ingredient_rxcui(
    rxcui: str,
    *,
    lookup: MedicationNormalizationLookup,
) -> bool:
    if str(rxcui) not in lookup.ingredient_safe_rxcuis:
        return False
    canonical_name = (
        str(lookup.ingredient_rxcui_to_name.get(rxcui, "")).strip()
        or str(lookup.rxcui_to_preferred_name.get(rxcui, "")).strip()
    )
    return bool(canonical_name) and not is_suspicious_ingredient_name(canonical_name)


def is_plausible_raw_to_ingredient_match(
    raw_token_body: str,
    canonical_rxcui: str,
    *,
    lookup: MedicationNormalizationLookup,
) -> bool:
    raw_name_key = normalize_drug_name(raw_token_body)
    if not raw_name_key:
        return False
    ingredient_record = lookup.rxnorm_index.name_index.rxcui_to_names.get(str(canonical_rxcui))
    candidate_normalized_names: set[str] = set()
    if ingredient_record is not None:
        if ingredient_record.normalized_preferred_name:
            candidate_normalized_names.add(str(ingredient_record.normalized_preferred_name))
        candidate_normalized_names.update(
            str(name)
            for name in ingredient_record.normalized_synonyms
            if str(name).strip()
        )
    if raw_name_key in candidate_normalized_names:
        return True

    raw_terms = _meaningful_name_terms(raw_name_key)
    if not raw_terms:
        return False
    for candidate_name in candidate_normalized_names:
        candidate_terms = _meaningful_name_terms(candidate_name)
        if not candidate_terms:
            continue
        overlap = raw_terms.intersection(candidate_terms)
        if overlap and (
            len(overlap) / len(raw_terms) >= 0.5
            or len(overlap) / len(candidate_terms) >= 0.5
        ):
            return True
        for raw_term in raw_terms:
            if len(raw_term) < MIN_PREFIX_MATCH_TERM_LENGTH:
                continue
            if any(
                len(candidate_term) >= MIN_PREFIX_MATCH_TERM_LENGTH
                and (candidate_term.startswith(raw_term) or raw_term.startswith(candidate_term))
                for candidate_term in candidate_terms
            ):
                return True
    return False


def _filter_plausible_canonical_rxcuis(
    raw_token_body: str,
    canonical_rxcuis: Sequence[str],
    *,
    lookup: MedicationNormalizationLookup,
) -> tuple[str, ...]:
    plausible_rxcuis = tuple(
        canonical_rxcui
        for canonical_rxcui in canonical_rxcuis
        if is_plausible_raw_to_ingredient_match(
            raw_token_body,
            canonical_rxcui,
            lookup=lookup,
        )
    )
    return tuple(sorted(dict.fromkeys(plausible_rxcuis)))


def _raw_matches_rxcui_name(
    raw_token_body: str,
    rxcui: str,
    *,
    lookup: MedicationNormalizationLookup,
) -> bool:
    raw_name_key = normalize_drug_name(raw_token_body)
    if not raw_name_key:
        return False
    record = lookup.rxnorm_index.name_index.rxcui_to_names.get(str(rxcui))
    if record is None:
        return False
    candidate_names = {
        str(record.normalized_preferred_name)
        for _ in [0]
        if str(record.normalized_preferred_name or "").strip()
    }
    candidate_names.update(
        str(name)
        for name in record.normalized_synonyms
        if str(name).strip()
    )
    return raw_name_key in candidate_names


def _meaningful_name_terms(normalized_name: str) -> set[str]:
    return {
        term
        for term in str(normalized_name or "").split("_")
        if term
        and term not in LEXICAL_STOPWORDS
        and not term.isdigit()
        and len(term) > 1
    }


def _build_suspicious_match_example(
    *,
    raw_token: str,
    raw_token_body: str,
    resolved_product_rxcui: str,
    candidate_canonical_rxcuis: Sequence[str],
    lookup: MedicationNormalizationLookup,
    reason: str,
    source: str,
) -> dict[str, Any]:
    return {
        "raw_token": raw_token,
        "raw_token_body": raw_token_body,
        "resolved_product_rxcui": resolved_product_rxcui,
        "resolved_product_name": str(lookup.rxcui_to_preferred_name.get(resolved_product_rxcui, "")).strip(),
        "candidate_canonical_rxcuis": list(candidate_canonical_rxcuis[:20]),
        "candidate_canonical_names": [
            _resolve_canonical_name(candidate_rxcui, lookup=lookup) or ""
            for candidate_rxcui in candidate_canonical_rxcuis[:20]
        ],
        "candidate_canonical_count": int(len(candidate_canonical_rxcuis)),
        "reason": reason,
        "source": source,
    }
