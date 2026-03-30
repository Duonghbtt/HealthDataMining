from __future__ import annotations

import re
from typing import Iterable, Mapping

from src.data.load_mimic import coerce_event_time


MEDICATION_TIME_PRIORITY = {
    "emar": ("charttime", "scheduletime", "storetime"),
    "prescriptions": ("starttime", "stoptime"),
    "pharmacy": ("starttime", "verifiedtime", "entertime"),
}


def _normalize_free_text(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.upper()).strip("_")
    return cleaned


def canonicalize_medication_text(value: str, *, prefer_code: bool = False) -> str | None:
    raw_value = str(value).strip()
    if not raw_value:
        return None
    if raw_value.startswith(("NAME:", "CODE:")):
        return raw_value
    normalized = _normalize_free_text(raw_value)
    if not normalized:
        return None
    prefix = "CODE" if prefer_code else "NAME"
    return f"{prefix}:{normalized}"


def extract_medication_token(row: Mapping[str, str]) -> str | None:
    for field in ("medication", "drug", "formulary_drug_cd"):
        token = canonicalize_medication_text(
            row.get(field, ""),
            prefer_code=(field == "formulary_drug_cd"),
        )
        if token:
            return token
    return None


def medication_event_time(table_name: str, row: Mapping[str, str]):
    fields = MEDICATION_TIME_PRIORITY.get(table_name, ())
    return coerce_event_time(row, fields)


def dedupe_preserve_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_cumulative_history(bucket_drug_ids: list[list[int]], max_history: int) -> list[list[int]]:
    history: list[int] = []
    output: list[list[int]] = []
    for bucket_ids in bucket_drug_ids:
        output.append(history[:max_history])
        for drug_id in reversed(bucket_ids):
            if drug_id in history:
                history.remove(drug_id)
            history.insert(0, drug_id)
        history = history[:max_history]
    return output
