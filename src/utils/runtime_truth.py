from __future__ import annotations

import copy
from typing import Any, Mapping


def normalize_ddi_context(ddi_context: Mapping[str, Any] | None) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(ddi_context or {}))
    source_metadata = dict(resolved.get("source_metadata") or {})
    active = bool(resolved.get("active", resolved.get("status") == "active"))
    resolved["active"] = active
    resolved["status"] = str(resolved.get("status") or ("active" if active else "inactive"))
    resolved["source"] = str(resolved.get("source") or resolved.get("effective_source") or "")
    resolved["requested_source"] = str(resolved.get("requested_source") or "")
    resolved["requested_source_format"] = str(resolved.get("requested_source_format") or "")
    resolved["effective_source"] = str(resolved.get("effective_source") or resolved["source"])
    resolved["effective_source_format"] = str(
        resolved.get("effective_source_format") or resolved.get("source_format") or ""
    )
    resolved["source_format"] = str(
        resolved.get("source_format") or resolved["effective_source_format"] or ""
    )
    resolved["matched_pairs"] = int(resolved.get("matched_pairs") or 0)
    resolved["nonzero_pairs"] = int(resolved.get("nonzero_pairs") or 0)
    resolved["vocab_size"] = int(resolved.get("vocab_size") or 0)
    resolved["fallback_reason"] = str(resolved.get("fallback_reason") or "")
    if resolved.get("ddi_type") is not None:
        source_metadata["kind"] = str(resolved["ddi_type"])
    if resolved.get("ddi_research_grade") is not None:
        source_metadata["research_grade"] = bool(resolved["ddi_research_grade"])
    if not source_metadata.get("kind"):
        source_metadata["kind"] = "unknown"
    if "research_grade" not in source_metadata:
        source_metadata["research_grade"] = False
    resolved["source_metadata"] = source_metadata
    resolved["ddi_type"] = str(resolved.get("ddi_type") or source_metadata.get("kind") or "unknown")
    resolved["ddi_research_grade"] = bool(
        resolved.get("ddi_research_grade", source_metadata.get("research_grade", False))
    )
    resolved["ddi_source"] = str(resolved.get("ddi_source") or resolved["effective_source"])
    return resolved


def ddi_truth_fields(ddi_context: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_ddi_context(ddi_context)
    source_metadata = dict(normalized.get("source_metadata") or {})
    return {
        "ddi_active": bool(normalized["active"]),
        "ddi_status": str(normalized["status"]),
        "ddi_type": str(source_metadata.get("kind") or "unknown"),
        "ddi_research_grade": bool(source_metadata.get("research_grade", False)),
        "ddi_source": str(normalized.get("ddi_source") or normalized.get("effective_source") or ""),
    }


def normalize_initialization_context(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(payload or {}))
    return {
        "initialization_mode": str(resolved.get("initialization_mode") or ""),
        "warm_start_mode": str(resolved.get("warm_start_mode") or ""),
        "warm_start_checkpoint": str(resolved.get("warm_start_checkpoint") or ""),
        "train_budget_label": str(resolved.get("train_budget_label") or ""),
    }


def build_core_runtime_truth(
    *,
    fusion_strategy: str,
    ddi_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "pipeline_level": "core+history",
        "history_active": True,
        "retrieval_active": False,
        "retrieval_status": "extension_only",
        "fusion_strategy": str(fusion_strategy),
        "train_mode": "core",
        "runtime_mode": "core",
        **ddi_truth_fields(ddi_context),
    }


def build_extension_runtime_truth(
    *,
    fusion_strategy: str,
    ddi_context: Mapping[str, Any] | None,
    retrieval_active: bool,
    group_encoder_active: bool,
    retrieval_scoring_mode: str | None = None,
    retrieval_cross_split_policy: str | None = None,
    retrieval_bank_policy: str | None = None,
) -> dict[str, Any]:
    truth = {
        "pipeline_level": "experimental_extension",
        "history_active": True,
        "retrieval_active": bool(retrieval_active),
        "retrieval_status": "experimental",
        "fusion_strategy": str(fusion_strategy),
        "train_mode": "extended",
        "runtime_mode": "extended",
        "group_encoder_active": bool(group_encoder_active),
        "extension_status": "experimental",
        **ddi_truth_fields(ddi_context),
    }
    if retrieval_scoring_mode is not None:
        truth["retrieval_scoring_mode"] = str(retrieval_scoring_mode)
        truth["retrieval_mode"] = str(retrieval_scoring_mode)
    if retrieval_cross_split_policy is not None:
        truth["retrieval_cross_split_policy"] = str(retrieval_cross_split_policy)
    if retrieval_bank_policy is not None:
        truth["retrieval_bank_policy"] = str(retrieval_bank_policy)
    return truth


__all__ = [
    "build_core_runtime_truth",
    "build_extension_runtime_truth",
    "ddi_truth_fields",
    "normalize_initialization_context",
    "normalize_ddi_context",
]
