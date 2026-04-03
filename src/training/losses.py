from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.ddi_regularization import DDIRegularizer


_VALID_REDUCTIONS = {"mean", "sum", "none"}


def _validate_reduction(reduction: str) -> str:
    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(f"reduction must be one of {_VALID_REDUCTIONS}, got {reduction!r}")
    return reduction


def _reduce_tensor(loss_tensor: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return loss_tensor.mean()
    if reduction == "sum":
        return loss_tensor.sum()
    if loss_tensor.ndim <= 1:
        return loss_tensor
    reduce_dims = tuple(range(1, loss_tensor.ndim))
    return loss_tensor.mean(dim=reduce_dims)


def extract_last_valid_targets(target_drugs: torch.Tensor, visit_mask: torch.Tensor) -> torch.Tensor:
    """Extract the last valid visit target for each sample.

    Parameters
    ----------
    target_drugs:
        Batched multi-visit medication targets with shape ``(B, T, D)``.
    visit_mask:
        Visit validity mask with shape ``(B, T)``.

    Returns
    -------
    torch.Tensor
        Final valid visit targets with shape ``(B, D)``.
    """

    if not isinstance(target_drugs, torch.Tensor):
        raise TypeError(f"target_drugs must be a torch.Tensor, got {type(target_drugs)!r}")
    if not isinstance(visit_mask, torch.Tensor):
        raise TypeError(f"visit_mask must be a torch.Tensor, got {type(visit_mask)!r}")
    if target_drugs.ndim != 3:
        raise ValueError(f"target_drugs must have shape (B, T, D), got {tuple(target_drugs.shape)}")
    if visit_mask.ndim != 2:
        raise ValueError(f"visit_mask must have shape (B, T), got {tuple(visit_mask.shape)}")
    if tuple(target_drugs.shape[:2]) != tuple(visit_mask.shape):
        raise ValueError(
            "target_drugs and visit_mask must agree on batch/time dimensions: "
            f"got {tuple(target_drugs.shape[:2])} and {tuple(visit_mask.shape)}"
        )

    resolved_mask = visit_mask.to(device=target_drugs.device, dtype=torch.bool)
    valid_counts = resolved_mask.sum(dim=1)
    if bool((valid_counts <= 0).any().item()):
        raise ValueError("Each sample must contain at least one valid visit")

    last_indices = valid_counts.to(dtype=torch.long) - 1
    batch_indices = torch.arange(target_drugs.shape[0], device=target_drugs.device)
    return target_drugs[batch_indices, last_indices]


def _resolve_targets(
    target_drugs: torch.Tensor,
    visit_mask: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(target_drugs, torch.Tensor):
        raise TypeError(f"target_drugs must be a torch.Tensor, got {type(target_drugs)!r}")

    if target_drugs.ndim == 2:
        targets = target_drugs
    elif target_drugs.ndim == 3:
        if visit_mask is None:
            raise ValueError("visit_mask is required when target_drugs has shape (B, T, D)")
        targets = extract_last_valid_targets(target_drugs, visit_mask)
    else:
        raise ValueError(f"target_drugs must have shape (B, D) or (B, T, D), got {tuple(target_drugs.shape)}")

    targets = targets.to(device=device, dtype=dtype)
    if not torch.isfinite(targets).all():
        raise ValueError("target_drugs must contain only finite values")
    return targets


class MedicationRecommendationLoss(nn.Module):
    """Compute prediction loss and optional DDI regularization for medication recommendation."""

    def __init__(
        self,
        *,
        lambda_ddi: float = 0.0,
        ddi_regularizer: DDIRegularizer | None = None,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if float(lambda_ddi) < 0.0:
            raise ValueError(f"lambda_ddi must be non-negative, got {lambda_ddi!r}")

        self.lambda_ddi = float(lambda_ddi)
        self.ddi_regularizer = ddi_regularizer
        self.reduction = _validate_reduction(reduction)

        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            pos_weight_tensor = torch.as_tensor(pos_weight, dtype=torch.float32)
            if pos_weight_tensor.ndim != 1:
                raise ValueError(f"pos_weight must have shape (D,), got {tuple(pos_weight_tensor.shape)}")
            if not torch.isfinite(pos_weight_tensor).all():
                raise ValueError("pos_weight must contain only finite values")
            self.register_buffer("pos_weight", pos_weight_tensor)

    def forward(
        self,
        *,
        drug_logits: torch.Tensor,
        target_drugs: torch.Tensor,
        visit_mask: torch.Tensor | None = None,
        drug_probs: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Compute total medication recommendation loss.

        Parameters
        ----------
        drug_logits:
            Model logits with shape ``(B, D)``.
        target_drugs:
            Either final-visit targets with shape ``(B, D)`` or trajectory targets
            with shape ``(B, T, D)``.
        visit_mask:
            Required when ``target_drugs`` has shape ``(B, T, D)``.
        drug_probs:
            Optional probability tensor with shape ``(B, D)``. If omitted,
            probabilities are computed from ``drug_logits`` using ``sigmoid``.

        Returns
        -------
        dict[str, Any]
            Dictionary containing total and component losses for trainer logging.
        """

        if not isinstance(drug_logits, torch.Tensor):
            raise TypeError(f"drug_logits must be a torch.Tensor, got {type(drug_logits)!r}")
        if drug_logits.ndim != 2:
            raise ValueError(f"drug_logits must have shape (B, D), got {tuple(drug_logits.shape)}")
        if not torch.isfinite(drug_logits).all():
            raise ValueError("drug_logits must contain only finite values")

        targets = _resolve_targets(
            target_drugs,
            visit_mask,
            device=drug_logits.device,
            dtype=drug_logits.dtype,
        )
        if tuple(targets.shape) != tuple(drug_logits.shape):
            raise ValueError(
                "Resolved targets must match drug_logits shape: "
                f"got {tuple(targets.shape)} and {tuple(drug_logits.shape)}"
            )

        if drug_probs is None:
            resolved_probs = torch.sigmoid(drug_logits)
        else:
            if not isinstance(drug_probs, torch.Tensor):
                raise TypeError(f"drug_probs must be a torch.Tensor, got {type(drug_probs)!r}")
            if tuple(drug_probs.shape) != tuple(drug_logits.shape):
                raise ValueError(
                    "drug_probs must match drug_logits shape: "
                    f"got {tuple(drug_probs.shape)} and {tuple(drug_logits.shape)}"
                )
            if not torch.isfinite(drug_probs).all():
                raise ValueError("drug_probs must contain only finite values")
            resolved_probs = drug_probs.to(device=drug_logits.device, dtype=drug_logits.dtype)

        resolved_pos_weight = None
        if self.pos_weight is not None:
            if self.pos_weight.shape[0] != drug_logits.shape[1]:
                raise ValueError(
                    "pos_weight width must match drug logits width: "
                    f"expected {int(drug_logits.shape[1])}, got {int(self.pos_weight.shape[0])}"
                )
            resolved_pos_weight = self.pos_weight.to(device=drug_logits.device, dtype=drug_logits.dtype)

        prediction_loss_matrix = F.binary_cross_entropy_with_logits(
            drug_logits,
            targets,
            pos_weight=resolved_pos_weight,
            reduction="none",
        )
        prediction_loss = _reduce_tensor(prediction_loss_matrix, self.reduction)

        if self.ddi_regularizer is None:
            ddi_per_sample = torch.zeros(
                drug_logits.shape[0],
                device=drug_logits.device,
                dtype=drug_logits.dtype,
            )
        else:
            ddi_per_sample = self.ddi_regularizer.compute_penalty_per_sample(resolved_probs)
            ddi_per_sample = ddi_per_sample.to(device=drug_logits.device, dtype=drug_logits.dtype)

        ddi_loss = _reduce_tensor(ddi_per_sample, self.reduction)
        weighted_ddi_loss = ddi_loss * self.lambda_ddi
        total_loss = prediction_loss + weighted_ddi_loss

        return {
            "total_loss": total_loss,
            "prediction_loss": prediction_loss,
            "ddi_loss": ddi_loss,
            "weighted_ddi_loss": weighted_ddi_loss,
            "lambda_ddi": torch.tensor(self.lambda_ddi, device=drug_logits.device, dtype=drug_logits.dtype),
        }


__all__ = ["MedicationRecommendationLoss", "extract_last_valid_targets"]
