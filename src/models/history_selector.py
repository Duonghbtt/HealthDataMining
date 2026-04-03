from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from src.retrieval.memory_bank import MemoryBank


ATTRIBUTE_ORDER = ("diagnosis", "procedure", "lab", "vital", "medication")
ATTRIBUTE_METADATA_FIELDS = {
    "diagnosis": "diag_code_sets",
    "procedure": "proc_code_sets",
    "lab": "lab_feature_sets",
    "vital": "vital_feature_sets",
    "medication": "target_drugs",
}


def _masked_softmax_dim(scores: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    if scores.shape != mask.shape:
        raise ValueError(f"Score shape {tuple(scores.shape)} must match mask shape {tuple(mask.shape)}")
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(masked_scores, dim=dim)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    normalizer = weights.sum(dim=dim, keepdim=True)
    safe_normalizer = torch.where(normalizer > 0, normalizer, torch.ones_like(normalizer))
    return weights / safe_normalizer


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError(f"Expected (B, N) scores, got shape {tuple(scores.shape)}")
    return _masked_softmax_dim(scores, mask, dim=-1)


def _gather_bank_states(
    memory_bank: MemoryBank,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if indices.ndim != 2:
        raise ValueError(f"Expected neighbor indices with shape (B, K), got {tuple(indices.shape)}")
    flat = indices.clamp(min=0).flatten().cpu()
    gathered = memory_bank.visit_states.index_select(0, flat).to(device=device, dtype=torch.float32)
    return gathered.view(indices.shape[0], indices.shape[1], -1)


def _masked_weighted_sum(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights.unsqueeze(-1)).sum(dim=1)


def _sparsify_weights(
    weights: torch.Tensor,
    mask: torch.Tensor,
    *,
    top_k: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if top_k is None:
        selected_mask = mask.clone()
        sparse_weights = torch.where(mask, weights, torch.zeros_like(weights))
        return sparse_weights, selected_mask
    if top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    sparse_weights = torch.zeros_like(weights)
    selected_mask = torch.zeros_like(mask)
    for row_index in range(weights.shape[0]):
        row_mask = mask[row_index]
        valid_count = int(row_mask.sum().item())
        if valid_count <= 0:
            continue
        keep = min(int(top_k), valid_count)
        row_values = weights[row_index].masked_fill(~row_mask, float("-inf"))
        top_positions = torch.topk(row_values, k=keep, dim=-1).indices
        selected_mask[row_index, top_positions] = True
        sparse_weights[row_index, top_positions] = weights[row_index, top_positions]
        denom = sparse_weights[row_index].sum()
        if float(denom.item()) > 0.0:
            sparse_weights[row_index] = sparse_weights[row_index] / denom
    return sparse_weights, selected_mask


def _empty_attribute_tensor(
    batch_size: int,
    candidate_count: int,
    hidden_dim: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_tensor = torch.zeros(
        batch_size,
        candidate_count,
        len(ATTRIBUTE_ORDER),
        dtype=torch.float32,
        device=device,
    )
    value_tensor = torch.zeros(
        batch_size,
        candidate_count,
        len(ATTRIBUTE_ORDER),
        hidden_dim,
        dtype=torch.float32,
        device=device,
    )
    return weight_tensor, value_tensor


class HistorySelector(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        dropout: float = 0.1,
        score_bias_weight: float = 0.5,
        self_top_k: int | None = 3,
        neighbor_top_k: int | None = 3,
        use_retrieval_bias: bool = True,
        use_attribute_gate: bool = True,
        use_group_reweight: bool = True,
        group_reweight_weight: float = 0.35,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.score_bias_weight = float(score_bias_weight)
        self.self_top_k = self_top_k
        self.neighbor_top_k = neighbor_top_k
        self.use_retrieval_bias = bool(use_retrieval_bias)
        self.use_attribute_gate = bool(use_attribute_gate)
        self.use_group_reweight = bool(use_group_reweight)
        self.group_reweight_weight = float(group_reweight_weight)

        self.self_query = nn.Linear(hidden_dim, hidden_dim)
        self.self_key = nn.Linear(hidden_dim, hidden_dim)
        self.self_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.neighbor_query = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_key = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attribute_query = nn.ModuleDict(
            {
                branch_name: nn.Linear(hidden_dim, hidden_dim)
                for branch_name in ("self", "neighbor")
            }
        )
        self.attribute_key = nn.ModuleDict(
            {
                branch_name: nn.ModuleDict(
                    {attribute_name: nn.Linear(hidden_dim, hidden_dim) for attribute_name in ATTRIBUTE_ORDER}
                )
                for branch_name in ("self", "neighbor")
            }
        )
        self.attribute_value = nn.ModuleDict(
            {
                branch_name: nn.ModuleDict(
                    {
                        attribute_name: nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                        )
                        for attribute_name in ATTRIBUTE_ORDER
                    }
                )
                for branch_name in ("self", "neighbor")
            }
        )
        self.attribute_fallback = nn.ModuleDict(
            {
                branch_name: nn.ModuleDict(
                    {
                        attribute_name: nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                        )
                        for attribute_name in ATTRIBUTE_ORDER
                    }
                )
                for branch_name in ("self", "neighbor")
            }
        )
        self.group_reweight_head = nn.ModuleDict(
            {
                branch_name: nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for branch_name in ("self", "neighbor")
            }
        )

    def _branch_attribute_payload(
        self,
        attribute_payload: Mapping[str, Any] | None,
        *,
        branch_name: str,
    ) -> Mapping[str, Any] | None:
        if attribute_payload is None:
            return None
        branch_payload = attribute_payload.get(branch_name)
        if isinstance(branch_payload, Mapping):
            return branch_payload
        if any(attribute_name in attribute_payload for attribute_name in ATTRIBUTE_ORDER):
            return attribute_payload
        return None

    def _attribute_mask_from_payload(
        self,
        branch_payload: Mapping[str, Any] | None,
        *,
        attribute_name: str,
        expected_shape: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor | None:
        if branch_payload is None:
            return None
        mask_payload = branch_payload.get("masks") or branch_payload.get("mask")
        raw_mask = None
        if isinstance(mask_payload, Mapping):
            raw_mask = mask_payload.get(attribute_name)
        if raw_mask is None:
            raw_mask = branch_payload.get(f"{attribute_name}_mask")
        if raw_mask is None:
            return None
        mask = torch.as_tensor(raw_mask, dtype=torch.bool, device=device)
        if tuple(mask.shape) != expected_shape:
            raise ValueError(
                f"Attribute mask for {attribute_name} must have shape {expected_shape}, got {tuple(mask.shape)}"
            )
        return mask

    def _attribute_representation_from_payload(
        self,
        branch_payload: Mapping[str, Any] | None,
        *,
        attribute_name: str,
        expected_shape: tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if branch_payload is None:
            return None
        representation_payload = branch_payload.get("representations", branch_payload)
        raw_repr = None
        if isinstance(representation_payload, Mapping):
            raw_repr = representation_payload.get(attribute_name)
        if raw_repr is None:
            raw_repr = branch_payload.get(f"{attribute_name}_repr")
        if raw_repr is None:
            return None
        tensor = torch.as_tensor(raw_repr, dtype=dtype, device=device)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Attribute representation for {attribute_name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
            )
        return tensor

    def _neighbor_attribute_masks(
        self,
        memory_bank: MemoryBank,
        neighbor_indices: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        masks = {
            attribute_name: torch.zeros_like(neighbor_mask, dtype=torch.bool)
            for attribute_name in ATTRIBUTE_ORDER
        }
        for row_index in range(neighbor_indices.shape[0]):
            for col_index in range(neighbor_indices.shape[1]):
                if not bool(neighbor_mask[row_index, col_index].item()):
                    continue
                bank_index = int(neighbor_indices[row_index, col_index].item())
                for attribute_name, field_name in ATTRIBUTE_METADATA_FIELDS.items():
                    masks[attribute_name][row_index, col_index] = len(getattr(memory_bank, field_name)[bank_index]) > 0
        return masks

    def _compute_attribute_evidence(
        self,
        *,
        branch_name: str,
        current_state: torch.Tensor,
        candidate_states: torch.Tensor,
        candidate_mask: torch.Tensor,
        branch_attribute_payload: Mapping[str, Any] | None,
        metadata_attribute_masks: Mapping[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor | list[str]]:
        batch_size, candidate_count, hidden_dim = candidate_states.shape
        device = candidate_states.device
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Expected candidate hidden dim {self.hidden_dim}, got {hidden_dim}")
        if candidate_count == 0:
            empty_scores, empty_values = _empty_attribute_tensor(
                batch_size,
                candidate_count,
                hidden_dim,
                device=device,
            )
            empty_mask = torch.zeros_like(empty_scores, dtype=torch.bool)
            empty_candidate_mask = torch.zeros(batch_size, candidate_count, dtype=torch.bool, device=device)
            return {
                "candidate_evidence": torch.zeros_like(candidate_states),
                "attribute_scores": empty_scores,
                "attribute_weights": empty_scores,
                "attribute_values": empty_values,
                "attribute_mask": empty_mask,
                "attribute_available_mask": empty_candidate_mask,
                "attribute_fallback_mask": empty_candidate_mask,
                "attribute_sources": ["fallback"] * len(ATTRIBUTE_ORDER),
            }

        query = self.attribute_query[branch_name](current_state)
        attribute_scores = []
        attribute_values = []
        attribute_masks = []
        attribute_sources = []
        expected_shape = (batch_size, candidate_count, hidden_dim)
        expected_mask_shape = (batch_size, candidate_count)

        for attribute_name in ATTRIBUTE_ORDER:
            provided_repr = self._attribute_representation_from_payload(
                branch_payload=branch_attribute_payload,
                attribute_name=attribute_name,
                expected_shape=expected_shape,
                device=device,
                dtype=candidate_states.dtype,
            )
            if provided_repr is None:
                attribute_repr = self.attribute_fallback[branch_name][attribute_name](candidate_states)
                attribute_sources.append("fallback")
            else:
                attribute_repr = provided_repr
                attribute_sources.append("provided")

            attribute_mask = candidate_mask.clone()
            payload_mask = self._attribute_mask_from_payload(
                branch_payload=branch_attribute_payload,
                attribute_name=attribute_name,
                expected_shape=expected_mask_shape,
                device=device,
            )
            if payload_mask is not None:
                attribute_mask = attribute_mask & payload_mask
            elif metadata_attribute_masks is not None and attribute_name in metadata_attribute_masks:
                attribute_mask = attribute_mask & metadata_attribute_masks[attribute_name].to(device=device, dtype=torch.bool)

            attribute_key = self.attribute_key[branch_name][attribute_name](attribute_repr)
            raw_score = torch.einsum("bnh,bh->bn", attribute_key, query) / (self.hidden_dim ** 0.5)
            attribute_scores.append(raw_score.unsqueeze(-1))
            attribute_values.append(self.attribute_value[branch_name][attribute_name](attribute_repr).unsqueeze(2))
            attribute_masks.append(attribute_mask.unsqueeze(-1))

        attribute_scores_tensor = torch.cat(attribute_scores, dim=-1)
        attribute_values_tensor = torch.cat(attribute_values, dim=2)
        attribute_mask_tensor = torch.cat(attribute_masks, dim=-1)
        score_basis = attribute_scores_tensor if self.use_attribute_gate else torch.zeros_like(attribute_scores_tensor)
        attribute_weights = _masked_softmax_dim(score_basis, attribute_mask_tensor, dim=-1)
        candidate_evidence = (attribute_values_tensor * attribute_weights.unsqueeze(-1)).sum(dim=2)
        attribute_available_mask = attribute_mask_tensor.any(dim=-1)
        attribute_fallback_mask = candidate_mask & ~attribute_available_mask
        candidate_evidence = torch.where(attribute_available_mask.unsqueeze(-1), candidate_evidence, candidate_states)
        candidate_evidence = torch.where(
            candidate_mask.unsqueeze(-1),
            candidate_evidence,
            torch.zeros_like(candidate_evidence),
        )
        return {
            "candidate_evidence": candidate_evidence,
            "attribute_scores": torch.where(attribute_mask_tensor, attribute_scores_tensor, torch.zeros_like(attribute_scores_tensor)),
            "attribute_weights": attribute_weights,
            "attribute_values": attribute_values_tensor,
            "attribute_mask": attribute_mask_tensor,
            "attribute_available_mask": attribute_available_mask,
            "attribute_fallback_mask": attribute_fallback_mask,
            "attribute_sources": attribute_sources,
        }

    def _apply_group_reweight(
        self,
        *,
        branch_name: str,
        candidate_evidence: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_scores: torch.Tensor,
        group_context: torch.Tensor | None,
        group_available_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | bool]:
        batch_size, candidate_count, _ = candidate_evidence.shape
        device = candidate_evidence.device
        zero = torch.zeros(batch_size, candidate_count, dtype=candidate_evidence.dtype, device=device)
        zero_mask = torch.zeros(batch_size, candidate_count, dtype=torch.bool, device=device)
        if (
            not self.use_group_reweight
            or group_context is None
            or group_available_mask is None
            or candidate_count == 0
        ):
            return {
                "scores": base_scores,
                "group_reweight_scores": zero,
                "group_influence": zero,
                "group_cosine_scores": zero,
                "group_used_mask": zero_mask,
                "group_used": False,
            }
        resolved_group_mask = group_available_mask.to(device=device, dtype=torch.bool)
        expanded_group = group_context.unsqueeze(1).expand(-1, candidate_count, -1)
        cosine_scores = F.cosine_similarity(candidate_evidence, expanded_group, dim=-1, eps=1.0e-8)
        learned_scores = self.group_reweight_head[branch_name](
            torch.cat([candidate_evidence, expanded_group], dim=-1)
        ).squeeze(-1)
        group_reweight_scores = 0.5 * (cosine_scores + learned_scores)
        group_used_mask = candidate_mask & resolved_group_mask.unsqueeze(-1)
        group_reweight_scores = torch.where(group_used_mask, group_reweight_scores, torch.zeros_like(group_reweight_scores))
        group_influence = self.group_reweight_weight * group_reweight_scores
        return {
            "scores": base_scores + group_influence,
            "group_reweight_scores": group_reweight_scores,
            "group_influence": group_influence,
            "group_cosine_scores": torch.where(group_used_mask, cosine_scores, torch.zeros_like(cosine_scores)),
            "group_used_mask": group_used_mask,
            "group_used": bool(group_used_mask.any().item()),
        }

    def _select_self_history(
        self,
        current_state: torch.Tensor,
        state_sequence: torch.Tensor,
        visit_mask: torch.Tensor,
        *,
        branch_attribute_payload: Mapping[str, Any] | None,
        group_context: torch.Tensor | None,
        group_available_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | list[str] | bool]:
        batch_size, time_steps, hidden_dim = state_sequence.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected state_sequence hidden dim {self.hidden_dim}, got {hidden_dim}"
            )

        visit_indices = torch.arange(time_steps, device=state_sequence.device).unsqueeze(0).expand(batch_size, -1)
        current_visit_index = visit_mask.sum(dim=-1).to(dtype=torch.long) - 1
        self_mask = visit_mask & (visit_indices < current_visit_index.unsqueeze(-1))

        attribute_outputs = self._compute_attribute_evidence(
            branch_name="self",
            current_state=current_state,
            candidate_states=state_sequence,
            candidate_mask=self_mask,
            branch_attribute_payload=branch_attribute_payload,
            metadata_attribute_masks=None,
        )
        candidate_evidence = attribute_outputs["candidate_evidence"]
        query = self.self_query(current_state).unsqueeze(1)
        keys = self.self_key(candidate_evidence)
        content_scores = torch.einsum("bth,bqh->bt", keys, query) / (self.hidden_dim ** 0.5)
        group_outputs = self._apply_group_reweight(
            branch_name="self",
            candidate_evidence=candidate_evidence,
            candidate_mask=self_mask,
            base_scores=content_scores,
            group_context=group_context,
            group_available_mask=group_available_mask,
        )
        scores = group_outputs["scores"]
        dense_weights = _masked_softmax(scores, self_mask)
        weights, selected_mask = _sparsify_weights(dense_weights, self_mask, top_k=self.self_top_k)
        values = self.self_value(candidate_evidence)
        context = _masked_weighted_sum(values, weights)
        available_mask = self_mask.any(dim=-1)
        top_positions = weights.argmax(dim=-1)
        top_indices = torch.where(
            available_mask,
            visit_indices.gather(1, top_positions.unsqueeze(-1)).squeeze(-1),
            torch.full_like(top_positions, -1),
        )
        return {
            "context": context,
            "available_mask": available_mask,
            "weights": weights,
            "dense_weights": dense_weights,
            "content_scores": torch.where(self_mask, content_scores, torch.zeros_like(content_scores)),
            "scores": torch.where(self_mask, scores, torch.zeros_like(scores)),
            "indices": visit_indices,
            "mask": self_mask,
            "selected_mask": selected_mask,
            "selected_count": selected_mask.sum(dim=-1),
            "top_index": top_indices,
            "current_visit_index": current_visit_index,
            "attribute_scores": attribute_outputs["attribute_scores"],
            "attribute_weights": attribute_outputs["attribute_weights"],
            "attribute_mask": attribute_outputs["attribute_mask"],
            "attribute_available_mask": attribute_outputs["attribute_available_mask"],
            "attribute_fallback_mask": attribute_outputs["attribute_fallback_mask"],
            "attribute_sources": attribute_outputs["attribute_sources"],
            "group_reweight_scores": group_outputs["group_reweight_scores"],
            "group_influence": group_outputs["group_influence"],
            "group_cosine_scores": group_outputs["group_cosine_scores"],
            "group_used_mask": group_outputs["group_used_mask"],
            "group_used": group_outputs["group_used"],
        }

    def _select_neighbor_history(
        self,
        current_state: torch.Tensor,
        retrieval_payload: Mapping[str, Any] | None,
        memory_bank: MemoryBank | None,
        *,
        branch_attribute_payload: Mapping[str, Any] | None,
        group_context: torch.Tensor | None,
        group_available_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | list[str] | bool]:
        batch_size = current_state.shape[0]
        device = current_state.device
        if retrieval_payload is None or memory_bank is None:
            empty = torch.zeros(batch_size, 0, dtype=torch.float32, device=device)
            empty_long = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            empty_bool = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
            empty_attr = torch.zeros(batch_size, 0, len(ATTRIBUTE_ORDER), dtype=torch.float32, device=device)
            return {
                "context": torch.zeros_like(current_state),
                "available_mask": torch.zeros(batch_size, dtype=torch.bool, device=device),
                "weights": empty,
                "dense_weights": empty,
                "content_scores": empty,
                "scores": empty,
                "indices": empty_long,
                "mask": empty_bool,
                "selected_mask": empty_bool,
                "selected_count": torch.zeros(batch_size, dtype=torch.long, device=device),
                "top_index": torch.full((batch_size,), -1, dtype=torch.long, device=device),
                "matched_visit_indices": empty_long,
                "retrieval_scores": empty,
                "retrieval_bias": empty,
                "neighbor_stay_ids": empty_long,
                "attribute_scores": empty_attr,
                "attribute_weights": empty_attr,
                "attribute_mask": torch.zeros_like(empty_attr, dtype=torch.bool),
                "attribute_available_mask": empty_bool,
                "attribute_fallback_mask": empty_bool,
                "attribute_sources": ["fallback"] * len(ATTRIBUTE_ORDER),
                "group_reweight_scores": empty,
                "group_influence": empty,
                "group_cosine_scores": empty,
                "group_used_mask": empty_bool,
                "group_used": False,
            }

        neighbor_indices = torch.as_tensor(retrieval_payload["neighbor_indices"], dtype=torch.long, device=device)
        neighbor_mask = neighbor_indices >= 0
        neighbor_states = _gather_bank_states(memory_bank, neighbor_indices, device=device)
        retrieval_scores = torch.as_tensor(
            retrieval_payload["neighbor_scores"],
            dtype=torch.float32,
            device=device,
        )
        retrieval_scores = torch.where(neighbor_mask, retrieval_scores, torch.zeros_like(retrieval_scores))
        if self.use_retrieval_bias:
            retrieval_bias = self.score_bias_weight * torch.tanh(retrieval_scores)
        else:
            retrieval_bias = torch.zeros_like(retrieval_scores)

        attribute_outputs = self._compute_attribute_evidence(
            branch_name="neighbor",
            current_state=current_state,
            candidate_states=neighbor_states,
            candidate_mask=neighbor_mask,
            branch_attribute_payload=branch_attribute_payload,
            metadata_attribute_masks=self._neighbor_attribute_masks(memory_bank, neighbor_indices, neighbor_mask),
        )
        candidate_evidence = attribute_outputs["candidate_evidence"]
        query = self.neighbor_query(current_state).unsqueeze(1)
        keys = self.neighbor_key(candidate_evidence)
        content_scores = torch.einsum("bkh,bqh->bk", keys, query) / (self.hidden_dim ** 0.5)
        group_outputs = self._apply_group_reweight(
            branch_name="neighbor",
            candidate_evidence=candidate_evidence,
            candidate_mask=neighbor_mask,
            base_scores=content_scores + retrieval_bias,
            group_context=group_context,
            group_available_mask=group_available_mask,
        )
        scores = group_outputs["scores"]
        dense_weights = _masked_softmax(scores, neighbor_mask)
        weights, selected_mask = _sparsify_weights(dense_weights, neighbor_mask, top_k=self.neighbor_top_k)
        values = self.neighbor_value(candidate_evidence)
        context = _masked_weighted_sum(values, weights)
        available_mask = neighbor_mask.any(dim=-1)
        top_positions = weights.argmax(dim=-1)
        top_indices = torch.where(
            available_mask,
            neighbor_indices.gather(1, top_positions.unsqueeze(-1)).squeeze(-1),
            torch.full_like(top_positions, -1),
        )
        matched_visit_indices = torch.as_tensor(
            retrieval_payload.get("matched_visit_indices", torch.full_like(neighbor_indices, -1)),
            dtype=torch.long,
            device=device,
        )
        neighbor_stay_ids = torch.as_tensor(
            retrieval_payload.get("neighbor_stay_ids", torch.full_like(neighbor_indices, -1)),
            dtype=torch.long,
            device=device,
        )
        return {
            "context": context,
            "available_mask": available_mask,
            "weights": weights,
            "dense_weights": dense_weights,
            "content_scores": torch.where(neighbor_mask, content_scores, torch.zeros_like(content_scores)),
            "scores": torch.where(neighbor_mask, scores, torch.zeros_like(scores)),
            "indices": neighbor_indices,
            "mask": neighbor_mask,
            "selected_mask": selected_mask,
            "selected_count": selected_mask.sum(dim=-1),
            "top_index": top_indices,
            "matched_visit_indices": matched_visit_indices,
            "retrieval_scores": retrieval_scores,
            "retrieval_bias": retrieval_bias,
            "neighbor_stay_ids": neighbor_stay_ids,
            "attribute_scores": attribute_outputs["attribute_scores"],
            "attribute_weights": attribute_outputs["attribute_weights"],
            "attribute_mask": attribute_outputs["attribute_mask"],
            "attribute_available_mask": attribute_outputs["attribute_available_mask"],
            "attribute_fallback_mask": attribute_outputs["attribute_fallback_mask"],
            "attribute_sources": attribute_outputs["attribute_sources"],
            "group_reweight_scores": group_outputs["group_reweight_scores"],
            "group_influence": group_outputs["group_influence"],
            "group_cosine_scores": group_outputs["group_cosine_scores"],
            "group_used_mask": group_outputs["group_used_mask"],
            "group_used": group_outputs["group_used"],
        }

    def forward(
        self,
        *,
        current_state: torch.Tensor,
        state_sequence: torch.Tensor,
        visit_mask: torch.Tensor,
        retrieval_payload: Mapping[str, Any] | None = None,
        memory_bank: MemoryBank | None = None,
        group_context: torch.Tensor | None = None,
        group_available_mask: torch.Tensor | None = None,
        attribute_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        device = current_state.device
        if group_context is None:
            resolved_group_context = None
            resolved_group_available_mask = torch.zeros(current_state.shape[0], dtype=torch.bool, device=device)
        else:
            resolved_group_context = torch.as_tensor(group_context, dtype=current_state.dtype, device=device)
            if tuple(resolved_group_context.shape) != tuple(current_state.shape):
                raise ValueError(
                    f"group_context must have shape {tuple(current_state.shape)}, got {tuple(resolved_group_context.shape)}"
                )
            if group_available_mask is None:
                resolved_group_available_mask = torch.ones(current_state.shape[0], dtype=torch.bool, device=device)
            else:
                resolved_group_available_mask = group_available_mask.to(device=device, dtype=torch.bool)

        self_history = self._select_self_history(
            current_state,
            state_sequence,
            visit_mask,
            branch_attribute_payload=self._branch_attribute_payload(attribute_payload, branch_name="self"),
            group_context=resolved_group_context,
            group_available_mask=resolved_group_available_mask,
        )
        neighbor_history = self._select_neighbor_history(
            current_state,
            retrieval_payload,
            memory_bank,
            branch_attribute_payload=self._branch_attribute_payload(attribute_payload, branch_name="neighbor"),
            group_context=resolved_group_context,
            group_available_mask=resolved_group_available_mask,
        )

        evidence_metadata: dict[str, Any] = {
            "attribute_order": list(ATTRIBUTE_ORDER),
            "current_visit_index": self_history["current_visit_index"],
            "self_history_indices": self_history["indices"],
            "self_history_mask": self_history["mask"],
            "self_history_content_scores": self_history["content_scores"],
            "self_history_scores": self_history["scores"],
            "self_history_weights": self_history["weights"],
            "self_history_dense_weights": self_history["dense_weights"],
            "self_history_top_index": self_history["top_index"],
            "self_history_selected_mask": self_history["selected_mask"],
            "self_history_selected_count": self_history["selected_count"],
            "self_history_available_mask": self_history["available_mask"],
            "neighbor_indices": neighbor_history["indices"],
            "neighbor_mask": neighbor_history["mask"],
            "neighbor_content_scores": neighbor_history["content_scores"],
            "neighbor_scores": neighbor_history["scores"],
            "neighbor_weights": neighbor_history["weights"],
            "neighbor_dense_weights": neighbor_history["dense_weights"],
            "neighbor_top_index": neighbor_history["top_index"],
            "neighbor_selected_mask": neighbor_history["selected_mask"],
            "neighbor_selected_count": neighbor_history["selected_count"],
            "neighbor_matched_visit_indices": neighbor_history["matched_visit_indices"],
            "neighbor_retrieval_scores": neighbor_history["retrieval_scores"],
            "neighbor_retrieval_bias": neighbor_history["retrieval_bias"],
            "neighbor_score_gain_from_bias": neighbor_history["retrieval_bias"],
            "neighbor_stay_ids": neighbor_history["neighbor_stay_ids"],
            "neighbor_available_mask": neighbor_history["available_mask"],
            "attribute_scores": {
                "self": self_history["attribute_scores"],
                "neighbor": neighbor_history["attribute_scores"],
            },
            "attribute_weights": {
                "self": self_history["attribute_weights"],
                "neighbor": neighbor_history["attribute_weights"],
            },
            "self_attribute_scores": self_history["attribute_scores"],
            "self_attribute_weights": self_history["attribute_weights"],
            "self_attribute_mask": self_history["attribute_mask"],
            "self_attribute_available_mask": self_history["attribute_available_mask"],
            "self_attribute_fallback_mask": self_history["attribute_fallback_mask"],
            "self_attribute_sources": list(self_history["attribute_sources"]),
            "neighbor_attribute_scores": neighbor_history["attribute_scores"],
            "neighbor_attribute_weights": neighbor_history["attribute_weights"],
            "neighbor_attribute_mask": neighbor_history["attribute_mask"],
            "neighbor_attribute_available_mask": neighbor_history["attribute_available_mask"],
            "neighbor_attribute_fallback_mask": neighbor_history["attribute_fallback_mask"],
            "neighbor_attribute_sources": list(neighbor_history["attribute_sources"]),
            "group_influence": {
                "self": self_history["group_influence"],
                "neighbor": neighbor_history["group_influence"],
            },
            "group_reweight_scores": {
                "self": self_history["group_reweight_scores"],
                "neighbor": neighbor_history["group_reweight_scores"],
            },
            "self_group_influence": self_history["group_influence"],
            "self_group_reweight_scores": self_history["group_reweight_scores"],
            "self_group_cosine_scores": self_history["group_cosine_scores"],
            "self_group_used_mask": self_history["group_used_mask"],
            "neighbor_group_influence": neighbor_history["group_influence"],
            "neighbor_group_reweight_scores": neighbor_history["group_reweight_scores"],
            "neighbor_group_cosine_scores": neighbor_history["group_cosine_scores"],
            "neighbor_group_used_mask": neighbor_history["group_used_mask"],
            "group_aware_selection_used": bool(self_history["group_used"] or neighbor_history["group_used"]),
            "group_aware_selection_mask": resolved_group_available_mask
            & (self_history["available_mask"] | neighbor_history["available_mask"]),
            "group_available_mask": resolved_group_available_mask,
            "selection_config": {
                "self_top_k": self.self_top_k,
                "neighbor_top_k": self.neighbor_top_k,
                "use_retrieval_bias": self.use_retrieval_bias,
                "use_attribute_gate": self.use_attribute_gate,
                "use_group_reweight": self.use_group_reweight,
                "group_reweight_weight": self.group_reweight_weight,
                "attribute_order": list(ATTRIBUTE_ORDER),
            },
        }
        if retrieval_payload is not None:
            for field_name in ("aux_personal_history_indices", "aux_personal_history_scores"):
                if field_name in retrieval_payload:
                    evidence_metadata[field_name] = torch.as_tensor(retrieval_payload[field_name], device=device)

        return {
            "self_history_context": self_history["context"],
            "neighbor_history_context": neighbor_history["context"],
            "group_context": resolved_group_context,
            "evidence_metadata": evidence_metadata,
        }
