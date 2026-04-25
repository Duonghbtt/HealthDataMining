from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


BRANCH_ORDER = ("current", "self", "neighbor", "group")


def _safe_entropy(weights: torch.Tensor) -> torch.Tensor:
    return -(weights * torch.log(weights.clamp(min=1.0e-12))).sum(dim=1)


class FusionModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        dropout: float = 0.1,
        strategy: str = "gated",
        current_branch_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        if strategy not in {"gated", "mean", "concat", "gated_residual"}:
            raise ValueError(f"Unsupported fusion strategy: {strategy}")
        if not 0.0 <= float(current_branch_dropout) <= 1.0:
            raise ValueError(
                f"current_branch_dropout must be in [0, 1], got {current_branch_dropout!r}"
            )
        self.strategy = strategy
        self.current_branch_dropout = float(current_branch_dropout)
        self.branch_projection = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for name in BRANCH_ORDER
            }
        )
        self.concat_projection = nn.Sequential(
            nn.Linear(hidden_dim * len(BRANCH_ORDER), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.branch_gate = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for name in BRANCH_ORDER
            }
        )
        self.current_self_balance: nn.Module | None = None
        self.residual_update: nn.Module | None = None
        if strategy == "gated_residual":
            self.current_self_balance = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
            self.residual_update = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def _resolve_branch(
        self,
        branch: torch.Tensor | None,
        *,
        reference: torch.Tensor,
        branch_mask: torch.Tensor | None,
        default_available: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = reference.shape[0]
        device = reference.device
        if branch is None:
            tensor = torch.zeros_like(reference)
            if branch_mask is None:
                mask = torch.full((batch_size,), default_available, dtype=torch.bool, device=device)
            else:
                mask = branch_mask.to(device=device, dtype=torch.bool)
            return tensor, mask

        if branch.shape != reference.shape:
            raise ValueError(
                f"Branch tensor shape {tuple(branch.shape)} must match reference shape {tuple(reference.shape)}"
            )
        if branch_mask is None:
            mask = torch.full((batch_size,), default_available, dtype=torch.bool, device=device)
        else:
            mask = branch_mask.to(device=device, dtype=torch.bool)
        return branch.to(device=device, dtype=reference.dtype), mask

    def forward(
        self,
        *,
        current_state: torch.Tensor,
        self_history_context: torch.Tensor | None = None,
        neighbor_history_context: torch.Tensor | None = None,
        group_context: torch.Tensor | None = None,
        branch_masks: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | list[str] | dict[str, torch.Tensor]]:
        device = current_state.device
        masks = branch_masks or {}
        branch_inputs = {
            "current": self._resolve_branch(current_state, reference=current_state, branch_mask=masks.get("current"), default_available=True),
            "self": self._resolve_branch(self_history_context, reference=current_state, branch_mask=masks.get("self"), default_available=self_history_context is not None),
            "neighbor": self._resolve_branch(neighbor_history_context, reference=current_state, branch_mask=masks.get("neighbor"), default_available=neighbor_history_context is not None),
            "group": self._resolve_branch(group_context, reference=current_state, branch_mask=masks.get("group"), default_available=group_context is not None),
        }

        projected = []
        logits = []
        mask_columns = []
        for branch_name in BRANCH_ORDER:
            branch_value, branch_mask = branch_inputs[branch_name]
            branch_proj = self.branch_projection[branch_name](branch_value)
            branch_logit = self.branch_gate[branch_name](
                torch.cat([current_state, branch_proj], dim=-1)
            ).squeeze(-1)
            projected.append(branch_proj)
            logits.append(branch_logit)
            mask_columns.append(branch_mask)

        projected_tensor = torch.stack(projected, dim=1)
        logits_tensor = torch.stack(logits, dim=1)
        branch_mask_tensor = torch.stack(mask_columns, dim=1).to(device=device, dtype=torch.bool)
        current_drop_mask = torch.zeros(branch_mask_tensor.shape[0], dtype=torch.bool, device=device)
        if self.training and self.current_branch_dropout > 0.0:
            auxiliary_available = branch_mask_tensor[:, 1:].any(dim=1)
            current_available = branch_mask_tensor[:, 0]
            eligible = current_available & auxiliary_available
            if bool(eligible.any().item()):
                sampled_keep = torch.rand(branch_mask_tensor.shape[0], device=device) >= self.current_branch_dropout
                current_drop_mask = eligible & ~sampled_keep
                branch_mask_tensor = branch_mask_tensor.clone()
                branch_mask_tensor[current_drop_mask, 0] = False
        masked_logits = logits_tensor.masked_fill(~branch_mask_tensor, float("-inf"))
        gate_weights = torch.softmax(masked_logits, dim=1)
        gate_weights = torch.where(branch_mask_tensor, gate_weights, torch.zeros_like(gate_weights))
        gate_normalizer = gate_weights.sum(dim=1, keepdim=True)
        gate_weights = gate_weights / torch.where(gate_normalizer > 0, gate_normalizer, torch.ones_like(gate_normalizer))

        balance_weights = torch.zeros(
            projected_tensor.shape[0],
            2,
            dtype=projected_tensor.dtype,
            device=device,
        )
        residual_update_norm = torch.zeros(
            projected_tensor.shape[0],
            dtype=projected_tensor.dtype,
            device=device,
        )
        if self.strategy == "gated":
            fusion_weights = gate_weights
            fused_repr = (projected_tensor * fusion_weights.unsqueeze(-1)).sum(dim=1)
        elif self.strategy == "gated_residual":
            if self.current_self_balance is None or self.residual_update is None:
                raise RuntimeError("gated_residual fusion was initialized without residual modules")
            fusion_weights = gate_weights
            gated_repr = (projected_tensor * fusion_weights.unsqueeze(-1)).sum(dim=1)
            current_proj = projected_tensor[:, BRANCH_ORDER.index("current")]
            self_proj = projected_tensor[:, BRANCH_ORDER.index("self")]
            current_self_logits = self.current_self_balance(torch.cat([current_proj, self_proj], dim=-1))
            self_available = branch_mask_tensor[:, BRANCH_ORDER.index("self")]
            balance_mask = torch.stack(
                [
                    torch.ones_like(self_available, dtype=torch.bool),
                    self_available,
                ],
                dim=1,
            )
            masked_balance_logits = current_self_logits.masked_fill(~balance_mask, float("-inf"))
            balance_weights = torch.softmax(masked_balance_logits, dim=1)
            balance_weights = torch.where(balance_mask, balance_weights, torch.zeros_like(balance_weights))
            balance_normalizer = balance_weights.sum(dim=1, keepdim=True)
            balance_weights = balance_weights / torch.where(
                balance_normalizer > 0,
                balance_normalizer,
                torch.ones_like(balance_normalizer),
            )
            current_self_mix = (
                torch.stack([current_proj, self_proj], dim=1)
                * balance_weights.unsqueeze(-1)
            ).sum(dim=1)
            residual_delta = self.residual_update(
                torch.cat([current_proj, current_self_mix, gated_repr], dim=-1)
            )
            residual_update_norm = residual_delta.norm(dim=-1)
            fused_repr = current_proj + 0.5 * residual_delta
        elif self.strategy == "mean":
            fusion_weights = branch_mask_tensor.to(dtype=projected_tensor.dtype)
            fusion_weights = fusion_weights / fusion_weights.sum(dim=1, keepdim=True).clamp(min=1.0)
            fused_repr = (projected_tensor * fusion_weights.unsqueeze(-1)).sum(dim=1)
        else:
            fusion_weights = branch_mask_tensor.to(dtype=projected_tensor.dtype)
            fusion_weights = fusion_weights / fusion_weights.sum(dim=1, keepdim=True).clamp(min=1.0)
            fused_repr = self.concat_projection(projected_tensor.reshape(projected_tensor.shape[0], -1))

        branch_entropy = _safe_entropy(fusion_weights)
        available_branch_count = branch_mask_tensor.sum(dim=1)
        max_entropy = torch.log(available_branch_count.to(dtype=fusion_weights.dtype).clamp(min=1.0))
        normalized_branch_entropy = torch.where(
            available_branch_count > 1,
            branch_entropy / max_entropy.clamp(min=1.0e-12),
            torch.ones_like(branch_entropy),
        )
        dominant_branch_index = fusion_weights.argmax(dim=1)
        dominant_branch_name = [BRANCH_ORDER[int(index)] for index in dominant_branch_index.tolist()]
        dominant_branch_weight = fusion_weights.max(dim=1).values
        branch_contribution_norms = (projected_tensor * fusion_weights.unsqueeze(-1)).norm(dim=-1)
        ideal_weights = branch_mask_tensor.to(dtype=fusion_weights.dtype)
        ideal_weights = ideal_weights / ideal_weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        branch_balance_error = torch.where(
            branch_mask_tensor,
            torch.abs(fusion_weights - ideal_weights),
            torch.zeros_like(fusion_weights),
        )
        branch_balance_score = 1.0 - (
            branch_balance_error.sum(dim=1)
            / branch_mask_tensor.sum(dim=1).clamp(min=1).to(dtype=fusion_weights.dtype)
        )
        branch_balance_gap = branch_balance_error.max(dim=1).values
        collapse_baseline = ideal_weights.max(dim=1).values
        branch_collapse_score = dominant_branch_weight - collapse_baseline
        branch_collapse_flag = (dominant_branch_weight >= 0.75) | (branch_collapse_score >= 0.35)
        fusion_entropy_loss = 1.0 - normalized_branch_entropy
        fusion_balance_loss = (
            ((fusion_weights - ideal_weights) ** 2) * branch_mask_tensor.to(dtype=fusion_weights.dtype)
        ).sum(dim=1) / branch_mask_tensor.sum(dim=1).clamp(min=1).to(dtype=fusion_weights.dtype)

        return {
            "fused_repr": fused_repr,
            "fusion_weights": fusion_weights,
            "branch_logits": logits_tensor,
            "gate_weights": gate_weights,
            "branch_mask": branch_mask_tensor,
            "current_branch_dropped": current_drop_mask,
            "current_self_balance_weights": balance_weights,
            "current_self_current_weight": balance_weights[:, 0],
            "current_self_history_weight": balance_weights[:, 1],
            "residual_update_norm": residual_update_norm,
            "branch_order": list(BRANCH_ORDER),
            "fusion_strategy": self.strategy,
            "branch_entropy": branch_entropy,
            "normalized_branch_entropy": normalized_branch_entropy,
            "available_branch_count": available_branch_count,
            "dominant_branch_index": dominant_branch_index,
            "dominant_branch_name": dominant_branch_name,
            "dominant_branch_weight": dominant_branch_weight,
            "branch_collapse_flag": branch_collapse_flag,
            "branch_collapse_score": branch_collapse_score,
            "branch_contribution_norms": branch_contribution_norms,
            "branch_balance_error": branch_balance_error,
            "branch_balance_gap": branch_balance_gap,
            "branch_balance_score": branch_balance_score,
            "fusion_entropy_loss": fusion_entropy_loss,
            "fusion_balance_loss": fusion_balance_loss,
            "branch_contexts": {name: projected_tensor[:, index] for index, name in enumerate(BRANCH_ORDER)},
        }
