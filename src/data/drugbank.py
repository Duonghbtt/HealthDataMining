from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from src.features.medication_history import canonicalize_medication_text
from src.utils.io import resolve_path


DRUGBANK_SOURCE_FORMAT = "drugbank_xml"
DRUGBANK_DDI_TYPE = "drugbank_knowledge_base_auxiliary"
DRUGBANK_MATCH_PRIORITY = ("primary_name", "synonym", "product_name")
_ENGLISH_LANGUAGE_CODES = {"", "en", "eng", "english"}
_DEFAULT_DRUGBANK_PATHS = {
    "source_path": "data/raw/drugbank/full database.xml",
    "summary_path": "data/processed/drugbank/drugbank_summary.json",
    "records_path": "data/processed/drugbank/drugbank_drugs.jsonl.gz",
    "vocab_metadata_path": "data/interim/vocab/drugbank_drug_metadata.json",
    "ddi_pairs_path": "data/processed/ddi/drugbank_ddi_pairs.jsonl.gz",
    "ddi_matrix_path": "data/processed/ddi/drug_ddi_drugbank.pt",
    "ddi_report_path": "data/processed/ddi/drug_ddi_drugbank_report.json",
}


def resolve_drugbank_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    drugbank_cfg = dict(config.get("drugbank", {}))
    resolved: dict[str, Path] = {}
    for key, default_value in _DEFAULT_DRUGBANK_PATHS.items():
        raw_value = drugbank_cfg.get(key, default_value)
        resolved[key] = resolve_path(config["_project_root"], raw_value).resolve()
    return resolved


def drugbank_source_metadata(source_path: Path | None) -> dict[str, Any]:
    display_name = "DrugBank XML" if source_path is None else source_path.name
    return {
        "kind": DRUGBANK_DDI_TYPE,
        "purpose": "DrugBank knowledge-base-derived auxiliary DDI source; benchmark opt-in only",
        "research_grade": False,
        "pair_schema": "drugbank_drug_interaction_edges",
        "display_name": display_name,
    }


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _normalized_text(value: str | None) -> str:
    return str(value or "").strip()


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_value in values:
        value = _normalized_text(raw_value)
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _tokenize_candidates(values: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        token = canonicalize_medication_text(value)
        if token:
            tokens.append(token)
    return _dedupe_preserve_order(tokens)


def _first_child_text(parent: ET.Element, child_name: str) -> str:
    for child in parent:
        if _local_name(child.tag) == child_name:
            return _normalized_text(child.text)
    return ""


def parse_drugbank_drug_element(drug_elem: ET.Element) -> dict[str, Any]:
    drugbank_ids: list[str] = []
    primary_drugbank_id = ""
    name = ""
    synonyms: list[str] = []
    product_names: list[str] = []
    interactions: list[dict[str, str]] = []

    for child in drug_elem:
        child_name = _local_name(child.tag)
        if child_name == "drugbank-id":
            value = _normalized_text(child.text)
            if value:
                drugbank_ids.append(value)
                if not primary_drugbank_id and str(child.attrib.get("primary", "")).strip().lower() == "true":
                    primary_drugbank_id = value
        elif child_name == "name":
            name = _normalized_text(child.text)
        elif child_name == "synonyms":
            for synonym_elem in child:
                if _local_name(synonym_elem.tag) != "synonym":
                    continue
                language = str(synonym_elem.attrib.get("language", "")).strip().lower()
                if language not in _ENGLISH_LANGUAGE_CODES:
                    continue
                synonym = _normalized_text(synonym_elem.text)
                if synonym:
                    synonyms.append(synonym)
        elif child_name == "products":
            for product_elem in child:
                if _local_name(product_elem.tag) != "product":
                    continue
                product_name = _first_child_text(product_elem, "name")
                if product_name:
                    product_names.append(product_name)
        elif child_name == "drug-interactions":
            for interaction_elem in child:
                if _local_name(interaction_elem.tag) != "drug-interaction":
                    continue
                target_drugbank_id = _first_child_text(interaction_elem, "drugbank-id")
                target_name = _first_child_text(interaction_elem, "name")
                description = _first_child_text(interaction_elem, "description")
                if target_drugbank_id or target_name or description:
                    interactions.append(
                        {
                            "drugbank_id": target_drugbank_id,
                            "name": target_name,
                            "description": description,
                        }
                    )

    deduped_ids = _dedupe_preserve_order(drugbank_ids)
    if not primary_drugbank_id and deduped_ids:
        primary_drugbank_id = deduped_ids[0]

    deduped_synonyms = [value for value in _dedupe_preserve_order(synonyms) if value != name]
    deduped_products = [
        value
        for value in _dedupe_preserve_order(product_names)
        if value != name and value not in deduped_synonyms
    ]
    name_token = canonicalize_medication_text(name) or ""
    synonym_tokens = _tokenize_candidates(deduped_synonyms)
    product_tokens = _tokenize_candidates(deduped_products)

    return {
        "primary_drugbank_id": primary_drugbank_id,
        "drugbank_ids": deduped_ids,
        "alias_drugbank_ids": [value for value in deduped_ids if value != primary_drugbank_id],
        "name": name,
        "name_token": name_token,
        "synonyms": deduped_synonyms,
        "synonym_tokens": synonym_tokens,
        "product_names": deduped_products,
        "product_tokens": product_tokens,
        "interaction_count": len(interactions),
        "interactions": interactions,
    }


def iter_drugbank_records(source_path: str | Path) -> Iterator[dict[str, Any]]:
    xml_path = Path(source_path)
    context = ET.iterparse(xml_path, events=("start", "end"))
    _, root = next(context)
    for event, elem in context:
        if event != "end" or _local_name(elem.tag) != "drug":
            continue
        yield parse_drugbank_drug_element(elem)
        elem.clear()
        root.clear()


def resolve_record_vocab_match(
    record: Mapping[str, Any],
    token_to_idx: Mapping[str, int],
) -> dict[str, Any]:
    match_levels = (
        ("primary_name", [str(record.get("name_token", "")).strip()]),
        ("synonym", [str(value).strip() for value in record.get("synonym_tokens", [])]),
        ("product_name", [str(value).strip() for value in record.get("product_tokens", [])]),
    )

    for match_source, candidate_tokens in match_levels:
        matched_tokens = sorted({token for token in candidate_tokens if token and token in token_to_idx})
        if len(matched_tokens) == 1:
            vocab_token = matched_tokens[0]
            return {
                "status": "matched",
                "match_source": match_source,
                "vocab_token": vocab_token,
                "vocab_idx": int(token_to_idx[vocab_token]),
                "candidate_vocab_tokens": matched_tokens,
            }
        if len(matched_tokens) > 1:
            return {
                "status": "ambiguous",
                "match_source": match_source,
                "candidate_vocab_tokens": matched_tokens,
            }

    return {
        "status": "unmatched",
        "match_source": "",
        "candidate_vocab_tokens": [],
    }

