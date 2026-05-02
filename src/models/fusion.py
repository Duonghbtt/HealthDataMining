from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


def _ensure_tensor_shape(name: str, value: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.zeros_like(reference)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor when provided, got {type(value)!r}")
    if tuple(value.shape) != tuple(reference.shape):
        raise ValueError(f"{name} must match current_state shape: got {tuple(value.shape)} and {tuple(reference.shape)}")
    return value


class _ContextAggregator(nn.Module):
    def __init__(self, hidden_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Linear(int(hidden_dim), 1)
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, contexts: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not contexts:
            raise ValueError("contexts must not be empty")
        names = list(contexts.keys())
        reference = contexts[names[0]]
        for name in names[1:]:
            if tuple(contexts[name].shape) != tuple(reference.shape):
                raise ValueError(
                    f"All fusion contexts must share the same shape, got {tuple(reference.shape)} and {tuple(contexts[name].shape)}"
                )
        stacked = torch.stack([contexts[name] for name in names], dim=1)  # [B, S, H]
        weights = torch.softmax(self.gate(stacked).squeeze(-1), dim=1)     # [B, S]
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)               # [B, H]
        return self.norm(self.dropout(fused)), {
            name: weights[:, index]
            for index, name in enumerate(names)
        }


class FusionModule(nn.Module):
    """Fuse current state, selected history, and medication history into one context."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        dropout: float = 0.1,
        strategy: str = "gated",
        mode: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.mode = str(mode if mode is not None else strategy).strip().lower()
        if self.mode not in {"concat", "gated"}:
            raise ValueError("FusionModule supports only `concat` and `gated` modes.")

        self.current_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.selected_history_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.medication_history_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.retrieval_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.auxiliary_history_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.concat_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 5, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(self.hidden_dim),
        )
        self.gated_fusion = _ContextAggregator(self.hidden_dim, dropout=float(dropout))

    def _aggregate_attribute_contexts(
        self,
        *,
        current_state: torch.Tensor,
        attribute_contexts: Mapping[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not attribute_contexts:
            return torch.zeros_like(current_state), {}
        filtered_contexts = {
            name: tensor
            for name, tensor in attribute_contexts.items()
            if isinstance(tensor, torch.Tensor)
        }
        if not filtered_contexts:
            return torch.zeros_like(current_state), {}
        for name, tensor in filtered_contexts.items():
            if tuple(tensor.shape) != tuple(current_state.shape):
                raise ValueError(
                    f"attribute_contexts[{name!r}] must match current_state shape: "
                    f"got {tuple(tensor.shape)} and {tuple(current_state.shape)}"
                )
        projected_contexts = {
            name: self.auxiliary_history_proj(tensor)
            for name, tensor in filtered_contexts.items()
        }
        return self.gated_fusion(projected_contexts)

    def forward(
        self,
        *,
        current_state: torch.Tensor,
        self_history_summary: torch.Tensor | None = None,
        selected_self_history: torch.Tensor | None = None,
        medication_history_context: torch.Tensor | None = None,
        retrieval_context: torch.Tensor | None = None,
        attribute_contexts: Mapping[str, torch.Tensor] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if current_state.ndim != 2:
            raise ValueError(f"current_state must have shape (B, H), got {tuple(current_state.shape)}")

        resolved_selected_history = _ensure_tensor_shape(
            "selected_self_history",
            selected_self_history if selected_self_history is not None else self_history_summary,
            current_state,
        )
        resolved_medication_history = _ensure_tensor_shape(
            "medication_history_context",
            medication_history_context,
            current_state,
        )
        resolved_retrieval_context = _ensure_tensor_shape(
            "retrieval_context",
            retrieval_context,
            current_state,
        )
        auxiliary_history, attribute_fusion_gates = self._aggregate_attribute_contexts(
            current_state=current_state,
            attribute_contexts=attribute_contexts,
        )

        projected_components = {
            "current_state": self.current_proj(current_state),
            "selected_self_history": self.selected_history_proj(resolved_selected_history),
            "medication_history_context": self.medication_history_proj(resolved_medication_history),
            "retrieval_context": self.retrieval_proj(resolved_retrieval_context),
            "auxiliary_history": auxiliary_history,
        }

        if self.mode == "concat":
            context_vector = self.concat_projection(
                torch.cat(
                    [
                        projected_components["current_state"],
                        projected_components["selected_self_history"],
                        projected_components["medication_history_context"],
                        projected_components["retrieval_context"],
                        projected_components["auxiliary_history"],
                    ],
                    dim=-1,
                )
            )
            fusion_gates: dict[str, torch.Tensor] = {}
        else:
            context_vector, fusion_gates = self.gated_fusion(projected_components)

        return {
            "context_vector": context_vector,
            "fused_representation": context_vector,
            "fusion_gates": fusion_gates,
            "attribute_fusion_gates": attribute_fusion_gates,
            "fusion_components": {
                "current_state": current_state,
                "selected_self_history": resolved_selected_history,
                "medication_history_context": resolved_medication_history,
                "retrieval_context": resolved_retrieval_context,
                "auxiliary_history": auxiliary_history,
            },
        }


__all__ = ["FusionModule"]
