from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn

_VALID_SELECTION_MODES = {"none", "visit_only", "visit_attribute"}
_VALID_ATTENTION_TYPES = {"softmax", "softmax_topk", "sparse_simple"}
_ATTRIBUTE_ALIASES = {
    "diagnosis": ("diagnosis", "diag", "diag_history_states", "diagnosis_history_states"),
    "procedure": ("procedure", "proc", "proc_history_states", "procedure_history_states"),
    "lab_vital": ("lab_vital", "lab_vital_history_states"),
    "medication_history": ("medication_history", "med_history", "med_history_states", "medication_history_states"),
}


def _validate_shapes(
    current_state: torch.Tensor,
    history_states: torch.Tensor,
    history_mask: torch.Tensor,
    hidden_dim: int,
) -> None:
    if current_state.ndim != 2:
        raise ValueError(f"current_state must have shape (B, H), got {tuple(current_state.shape)}")
    if history_states.ndim != 3:
        raise ValueError(f"history_states must have shape (B, T, H), got {tuple(history_states.shape)}")
    if history_mask.ndim != 2:
        raise ValueError(f"history_mask must have shape (B, T), got {tuple(history_mask.shape)}")
    if tuple(history_states.shape[:2]) != tuple(history_mask.shape):
        raise ValueError(
            "history_states and history_mask must align on batch/time dimensions: "
            f"got {tuple(history_states.shape[:2])} and {tuple(history_mask.shape)}"
        )
    if current_state.shape[0] != history_states.shape[0]:
        raise ValueError(
            "current_state and history_states must align on batch dimension: "
            f"got {tuple(current_state.shape)} and {tuple(history_states.shape)}"
        )
    if current_state.shape[1] != hidden_dim:
        raise ValueError(f"Expected hidden dimension {hidden_dim}, got {int(current_state.shape[1])}")


def _normalize_mode(selection_mode: str) -> str:
    normalized = str(selection_mode).strip().lower()
    if normalized not in _VALID_SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {_VALID_SELECTION_MODES}, got {selection_mode!r}")
    return normalized


def _normalize_attention_type(attention_type: str) -> str:
    normalized = str(attention_type).strip().lower()
    if normalized not in _VALID_ATTENTION_TYPES:
        raise ValueError(f"attention_type must be one of {_VALID_ATTENTION_TYPES}, got {attention_type!r}")
    return normalized


def _infer_history_mask(visit_mask: torch.Tensor) -> torch.Tensor:
    resolved_mask = visit_mask.to(dtype=torch.bool)
    valid_counts = resolved_mask.sum(dim=1)
    if bool((valid_counts <= 0).any().item()):
        raise ValueError("Each sample must contain at least one valid visit")
    batch_size, time_steps = resolved_mask.shape
    visit_indices = torch.arange(time_steps, device=resolved_mask.device).unsqueeze(0).expand(batch_size, -1)
    last_valid_index = valid_counts.to(dtype=torch.long) - 1
    return resolved_mask & (visit_indices < last_valid_index.unsqueeze(-1))


def _selected_indices_to_mask(selected_visit_indices: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
    if selected_visit_indices.ndim != 2:
        raise ValueError(f"selected_visit_indices must have shape (B, K), got {tuple(selected_visit_indices.shape)}")
    selected_mask = torch.zeros_like(history_mask, dtype=torch.bool)
    valid_selected = selected_visit_indices >= 0
    if bool(valid_selected.any().item()):
        selected_mask.scatter_(1, selected_visit_indices.masked_fill(~valid_selected, 0), valid_selected)
    return selected_mask & history_mask


def _extract_selected_visit_indices(
    attention_weights: torch.Tensor,
    history_mask: torch.Tensor,
    *,
    top_k: int | None,
) -> torch.Tensor:
    batch_size, time_steps = attention_weights.shape
    if top_k is None:
        selection_width = time_steps
    else:
        if int(top_k) <= 0:
            raise ValueError("top_k must be positive when provided")
        selection_width = min(int(top_k), time_steps)
    if selection_width == 0:
        return torch.empty(batch_size, 0, dtype=torch.long, device=attention_weights.device)

    masked_weights = attention_weights.masked_fill(~history_mask, -1.0)
    ordered_indices = torch.argsort(masked_weights, dim=-1, descending=True)
    selected_indices = ordered_indices[:, :selection_width].to(dtype=torch.long)
    selected_mask = history_mask.gather(1, selected_indices)
    return torch.where(selected_mask, selected_indices, torch.full_like(selected_indices, -1))


def _renormalize_rows(weights: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    resolved_weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
    denom = resolved_weights.sum(dim=-1, keepdim=True)
    return resolved_weights / torch.where(denom > 0, denom, torch.ones_like(denom))


def _masked_uniform(valid_mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    weights = valid_mask.to(dtype=dtype)
    denom = weights.sum(dim=-1, keepdim=True)
    return weights / torch.where(denom > 0, denom, torch.ones_like(denom))


class _AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.LazyLinear(self.hidden_dim)
        self.value_proj = nn.LazyLinear(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def _apply_attention(
        self,
        scores: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        attention_type: str,
        prior_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_type == "sparse_simple":
            positive_scores = torch.relu(torch.where(valid_mask, scores, torch.zeros_like(scores)))
            if prior_weights is not None:
                positive_scores = positive_scores * torch.where(valid_mask, prior_weights, torch.zeros_like(positive_scores))
            denom = positive_scores.sum(dim=-1, keepdim=True)
            uniform = _masked_uniform(valid_mask, dtype=scores.dtype)
            return torch.where(denom > 0, positive_scores / denom, uniform)

        masked_scores = scores.masked_fill(~valid_mask, -1.0e9)
        weights = torch.softmax(masked_scores, dim=-1)
        if prior_weights is not None:
            weights = weights * torch.where(valid_mask, prior_weights, torch.zeros_like(weights))
        return _renormalize_rows(weights, valid_mask)

    def forward(
        self,
        *,
        current_state: torch.Tensor,
        history_states: torch.Tensor,
        history_mask: torch.Tensor,
        attention_type: str,
        top_k: int | None,
        prior_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.query_proj(current_state)                                    # [B, H]
        keys = self.key_proj(history_states)                                      # [B, T, H]
        values = self.dropout(torch.tanh(self.value_proj(history_states)))        # [B, T, H]
        scores = torch.einsum("bth,bh->bt", keys, query) / math.sqrt(float(self.hidden_dim))

        valid_mask = history_mask
        if top_k is not None and int(top_k) > 0:
            top_k = min(int(top_k), int(history_mask.shape[1]))
            topk_scores = scores.masked_fill(~history_mask, -1.0e9)
            topk_indices = torch.topk(topk_scores, k=top_k, dim=-1).indices
            topk_mask = torch.zeros_like(history_mask, dtype=torch.bool)
            topk_mask.scatter_(1, topk_indices, True)
            valid_mask = history_mask & topk_mask

        weights = self._apply_attention(
            scores,
            valid_mask,
            attention_type=attention_type,
            prior_weights=prior_weights,
        )
        context = (values * weights.unsqueeze(-1)).sum(dim=1)
        has_history = history_mask.any(dim=1, keepdim=True)
        context = torch.where(has_history, context, torch.zeros_like(context))
        selected_visit_indices = _extract_selected_visit_indices(weights, history_mask, top_k=top_k)
        return context, weights, selected_visit_indices


class _ContextFusion(nn.Module):
    def __init__(self, hidden_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gate = nn.Linear(self.hidden_dim, 1)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, contexts: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not contexts:
            raise ValueError("contexts must not be empty")
        names = list(contexts.keys())
        reference = contexts[names[0]]
        for name in names[1:]:
            if tuple(contexts[name].shape) != tuple(reference.shape):
                raise ValueError(
                    f"All contexts must share the same shape, got {tuple(reference.shape)} and {tuple(contexts[name].shape)}"
                )
        stacked = torch.stack([contexts[name] for name in names], dim=1)
        weights = torch.softmax(self.gate(stacked).squeeze(-1), dim=1)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return self.norm(self.dropout(fused)), {
            name: weights[:, index]
            for index, name in enumerate(names)
        }


class SelfHistorySelector(nn.Module):
    """Relevant-history selector with visit-level and optional attribute-level attention."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        dropout: float = 0.1,
        max_selected_visits: int | None = None,
        self_top_k: int | None = None,
        selection_mode: str = "visit_only",
        top_k: int | None = None,
        attention_type: str = "softmax_topk",
        return_attention_weights: bool = True,
        save_selected_indices: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.selection_mode = _normalize_mode(selection_mode)
        self.attention_type = _normalize_attention_type(attention_type)
        self.return_attention_weights = bool(return_attention_weights)
        self.save_selected_indices = bool(save_selected_indices)
        self.top_k = (
            top_k
            if top_k is not None
            else (self_top_k if self_top_k is not None else max_selected_visits)
        )
        if self.top_k is not None and int(self.top_k) <= 0:
            raise ValueError("top_k must be positive when provided")

        self.visit_attention = _AttentionPool(self.hidden_dim, dropout=float(dropout))
        self.attribute_attention = nn.ModuleDict(
            {
                "diagnosis": _AttentionPool(self.hidden_dim, dropout=float(dropout)),
                "procedure": _AttentionPool(self.hidden_dim, dropout=float(dropout)),
                "lab_vital": _AttentionPool(self.hidden_dim, dropout=float(dropout)),
                "medication_history": _AttentionPool(self.hidden_dim, dropout=float(dropout)),
            }
        )
        self.attribute_fusion = _ContextFusion(self.hidden_dim, dropout=float(dropout))

    def _resolve_history_inputs(
        self,
        *,
        current_state: torch.Tensor,
        state_sequence: torch.Tensor | None,
        history_states: torch.Tensor | None,
        visit_mask: torch.Tensor | None,
        history_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resolved_states = history_states if history_states is not None else state_sequence
        if not isinstance(resolved_states, torch.Tensor):
            raise KeyError("History selector requires `history_states` or `state_sequence`.")

        if history_mask is None:
            if visit_mask is None:
                raise KeyError("History selector requires `history_mask` or `visit_mask`.")
            resolved_history_mask = _infer_history_mask(torch.as_tensor(visit_mask, dtype=torch.bool, device=resolved_states.device))
        else:
            resolved_history_mask = torch.as_tensor(history_mask, dtype=torch.bool, device=resolved_states.device)
            if visit_mask is not None:
                resolved_history_mask = resolved_history_mask & _infer_history_mask(
                    torch.as_tensor(visit_mask, dtype=torch.bool, device=resolved_states.device)
                )
        _validate_shapes(current_state, resolved_states, resolved_history_mask, self.hidden_dim)
        return resolved_states, resolved_history_mask

    def _resolve_modality_history_states(
        self,
        *,
        history_states: torch.Tensor,
        modality_history_states: Mapping[str, torch.Tensor] | None,
        extras: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        resolved: dict[str, torch.Tensor] = {}
        source_mapping = dict(modality_history_states or {})
        for canonical_name, aliases in _ATTRIBUTE_ALIASES.items():
            candidate: torch.Tensor | None = None
            for alias in aliases:
                if alias in source_mapping:
                    candidate = source_mapping[alias]
                    break
                extra_value = extras.get(alias)
                if isinstance(extra_value, torch.Tensor):
                    candidate = extra_value
                    break
            if candidate is None:
                continue
            if candidate.ndim != 3 or tuple(candidate.shape[:2]) != tuple(history_states.shape[:2]):
                raise ValueError(
                    f"{canonical_name} history states must align with history_states on batch/time dimensions: "
                    f"got {tuple(candidate.shape)} and {tuple(history_states.shape)}"
                )
            resolved[canonical_name] = candidate
        return resolved

    def forward(
        self,
        current_state: torch.Tensor,
        state_sequence: torch.Tensor | None = None,
        visit_mask: torch.Tensor | None = None,
        *,
        history_states: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        modality_history_states: Mapping[str, torch.Tensor] | None = None,
        **extras: Any,
    ) -> dict[str, Any]:
        if current_state.ndim != 2:
            raise ValueError(f"current_state must have shape (B, H), got {tuple(current_state.shape)}")

        resolved_history_states, resolved_history_mask = self._resolve_history_inputs(
            current_state=current_state,
            state_sequence=state_sequence,
            history_states=history_states,
            visit_mask=visit_mask,
            history_mask=history_mask,
        )
        batch_size, time_steps, _ = resolved_history_states.shape

        zero_context = torch.zeros(batch_size, self.hidden_dim, device=current_state.device, dtype=current_state.dtype)
        zero_attention = torch.zeros(batch_size, time_steps, device=current_state.device, dtype=current_state.dtype)
        empty_indices = torch.empty(batch_size, 0, device=current_state.device, dtype=torch.long)

        if self.selection_mode == "none":
            return {
                "selection_mode": self.selection_mode,
                "selected_history_context": zero_context,
                "history_context": zero_context,
                "visit_context": zero_context,
                "self_history_summary": zero_context,
                "medication_history_context": zero_context,
                "visit_attention_weights": zero_attention,
                "self_attention_weights": zero_attention,
                "attribute_contexts": {},
                "attribute_attention_weights": {},
                "selected_visit_indices": empty_indices,
                "selected_visit_mask": torch.zeros_like(resolved_history_mask, dtype=torch.bool),
            }

        visit_context, visit_attention_weights, selected_visit_indices = self.visit_attention(
            current_state=current_state,
            history_states=resolved_history_states,
            history_mask=resolved_history_mask,
            attention_type=self.attention_type,
            top_k=self.top_k if self.selection_mode in {"visit_only", "visit_attribute"} else None,
        )
        selected_visit_mask = (
            resolved_history_mask
            if self.top_k is None
            else _selected_indices_to_mask(selected_visit_indices, resolved_history_mask)
        )

        attribute_contexts: dict[str, torch.Tensor] = {}
        attribute_attention_weights: dict[str, torch.Tensor] = {}
        attribute_fusion_weights: dict[str, torch.Tensor] = {}
        medication_history_context = zero_context
        has_medication_history_context = False

        if self.selection_mode == "visit_attribute":
            modality_states = self._resolve_modality_history_states(
                history_states=resolved_history_states,
                modality_history_states=modality_history_states,
                extras=extras,
            )
            for name, modality_states_tensor in modality_states.items():
                context, weights, _ = self.attribute_attention[name](
                    current_state=current_state,
                    history_states=modality_states_tensor,
                    history_mask=selected_visit_mask if bool(selected_visit_mask.any().item()) else resolved_history_mask,
                    attention_type=self.attention_type,
                    top_k=None,
                    prior_weights=visit_attention_weights,
                )
                if name == "diagnosis":
                    attribute_contexts["diag_context"] = context
                elif name == "procedure":
                    attribute_contexts["proc_context"] = context
                elif name == "lab_vital":
                    attribute_contexts["lab_vital_context"] = context
                elif name == "medication_history":
                    attribute_contexts["med_context"] = context
                    medication_history_context = context
                    has_medication_history_context = True
                attribute_attention_weights[name] = weights

            if attribute_contexts:
                fusion_contexts = {"visit_context": visit_context, **attribute_contexts}
                selected_history_context, attribute_fusion_weights = self.attribute_fusion(fusion_contexts)
            else:
                selected_history_context = visit_context
        else:
            selected_history_context = visit_context

        if not has_medication_history_context and "med_context" in attribute_contexts:
            medication_history_context = attribute_contexts["med_context"]

        return {
            "selection_mode": self.selection_mode,
            "selected_history_context": selected_history_context,
            "history_context": selected_history_context,
            "visit_context": visit_context,
            "self_history_summary": selected_history_context,
            "medication_history_context": medication_history_context,
            "visit_attention_weights": visit_attention_weights if self.return_attention_weights else zero_attention,
            "self_attention_weights": visit_attention_weights if self.return_attention_weights else zero_attention,
            "attribute_contexts": attribute_contexts,
            "attribute_attention_weights": attribute_attention_weights if self.return_attention_weights else {},
            "attribute_fusion_weights": attribute_fusion_weights,
            "selected_visit_indices": selected_visit_indices if self.save_selected_indices else empty_indices,
            "selected_visit_mask": selected_visit_mask,
        }


HistorySelector = SelfHistorySelector

__all__ = ["HistorySelector", "SelfHistorySelector"]

