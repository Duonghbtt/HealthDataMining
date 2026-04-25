from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.ddi_regularization import DDIRegularizer


_VALID_REDUCTIONS = {"mean", "sum", "none"}
_VALID_OBJECTIVES = {"bce", "focal_lite", "asymmetric_focal"}
_VALID_RANKING_OBJECTIVES = {"bpr", "margin"}


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


def _validate_objective(objective: str) -> str:
    resolved = str(objective).strip().lower()
    if resolved not in _VALID_OBJECTIVES:
        raise ValueError(f"objective must be one of {_VALID_OBJECTIVES}, got {objective!r}")
    return resolved


def _validate_ranking_objective(objective: str) -> str:
    resolved = str(objective).strip().lower()
    if resolved not in _VALID_RANKING_OBJECTIVES:
        raise ValueError(
            f"ranking_objective must be one of {_VALID_RANKING_OBJECTIVES}, got {objective!r}"
        )
    return resolved


def _binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    bce_matrix = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    focal_factor = (1.0 - pt).clamp(min=0.0) ** float(gamma)
    return bce_matrix * focal_factor


def _binary_asymmetric_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: torch.Tensor | None,
    gamma_pos: float,
    gamma_neg: float,
    clip: float,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    positive_loss = F.softplus(-logits) * targets
    if pos_weight is not None:
        positive_loss = positive_loss * pos_weight

    if float(clip) > 0.0:
        negative_probs = (probs - float(clip)).clamp(min=0.0, max=1.0 - 1.0e-8)
        negative_probs_for_log = negative_probs.float().clamp(max=1.0 - 1.0e-6)
        negative_loss = -torch.log1p(-negative_probs_for_log).to(dtype=logits.dtype) * (1.0 - targets)
    else:
        negative_probs = probs
        negative_loss = F.softplus(logits) * (1.0 - targets)

    positive_focal = (1.0 - probs).clamp(min=0.0) ** float(gamma_pos)
    negative_focal = negative_probs.clamp(min=0.0) ** float(gamma_neg)
    return positive_loss * positive_focal + negative_loss * negative_focal


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


def _select_negative_logits(
    row_logits: torch.Tensor,
    row_targets: torch.Tensor,
    *,
    num_negatives: int,
    hard_negative_fraction: float,
) -> torch.Tensor:
    negative_logits = row_logits[row_targets <= 0.5]
    if negative_logits.numel() <= 0:
        return negative_logits
    keep = min(int(num_negatives), int(negative_logits.numel()))
    if keep <= 0:
        return negative_logits.new_zeros((0,))

    hard_count = int(round(float(keep) * float(hard_negative_fraction)))
    hard_count = min(max(hard_count, 0), keep)
    selected_parts: list[torch.Tensor] = []
    if hard_count > 0:
        selected_parts.append(torch.topk(negative_logits, k=hard_count, dim=0).values)

    random_count = keep - hard_count
    if random_count > 0:
        if hard_count > 0 and negative_logits.numel() > hard_count:
            hard_threshold = selected_parts[0].min()
            random_pool = negative_logits[negative_logits < hard_threshold]
            if random_pool.numel() < random_count:
                random_pool = negative_logits
        else:
            random_pool = negative_logits
        if random_pool.numel() <= random_count:
            selected_parts.append(random_pool)
        else:
            permutation = torch.randperm(random_pool.numel(), device=random_pool.device)
            selected_parts.append(random_pool.index_select(0, permutation[:random_count]))

    return torch.cat(selected_parts, dim=0) if selected_parts else negative_logits.new_zeros((0,))


def _sampled_pairwise_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_negatives: int,
    objective: str,
    margin: float,
    hard_negative_fraction: float,
) -> torch.Tensor:
    if int(num_negatives) <= 0:
        return logits.new_zeros(())
    row_losses: list[torch.Tensor] = []
    for row_index in range(logits.shape[0]):
        row_logits = logits[row_index]
        row_targets = targets[row_index]
        positive_logits = row_logits[row_targets > 0.5]
        if positive_logits.numel() <= 0:
            continue
        negative_logits = _select_negative_logits(
            row_logits,
            row_targets,
            num_negatives=int(num_negatives),
            hard_negative_fraction=float(hard_negative_fraction),
        )
        if negative_logits.numel() <= 0:
            continue
        pairwise_margin = positive_logits.unsqueeze(1) - negative_logits.unsqueeze(0)
        if objective == "bpr":
            row_loss = F.softplus(-pairwise_margin).mean()
        else:
            row_loss = torch.relu(float(margin) - pairwise_margin).mean()
        row_losses.append(row_loss)
    if not row_losses:
        return logits.new_zeros(())
    return torch.stack(row_losses, dim=0).mean()


class MedicationRecommendationLoss(nn.Module):
    """Compute prediction loss and optional DDI regularization for medication recommendation."""

    def __init__(
        self,
        *,
        lambda_ddi: float = 0.0,
        ddi_regularizer: DDIRegularizer | None = None,
        ddi_context: dict[str, Any] | None = None,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
        objective: str = "bce",
        focal_gamma: float = 1.5,
        asymmetric_gamma_pos: float = 0.0,
        asymmetric_gamma_neg: float = 4.0,
        asymmetric_clip: float = 0.05,
        fusion_entropy_lambda: float = 0.0,
        fusion_balance_lambda: float = 0.0,
        ranking_lambda: float = 0.0,
        ranking_objective: str = "bpr",
        ranking_margin: float = 1.0,
        ranking_num_negatives: int = 32,
        ranking_hard_negative_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        if float(lambda_ddi) < 0.0:
            raise ValueError(f"lambda_ddi must be non-negative, got {lambda_ddi!r}")
        if float(focal_gamma) < 0.0:
            raise ValueError(f"focal_gamma must be non-negative, got {focal_gamma!r}")
        if float(asymmetric_gamma_pos) < 0.0:
            raise ValueError(
                f"asymmetric_gamma_pos must be non-negative, got {asymmetric_gamma_pos!r}"
            )
        if float(asymmetric_gamma_neg) < 0.0:
            raise ValueError(
                f"asymmetric_gamma_neg must be non-negative, got {asymmetric_gamma_neg!r}"
            )
        if not 0.0 <= float(asymmetric_clip) < 1.0:
            raise ValueError(f"asymmetric_clip must be in [0, 1), got {asymmetric_clip!r}")
        if float(fusion_entropy_lambda) < 0.0:
            raise ValueError(
                f"fusion_entropy_lambda must be non-negative, got {fusion_entropy_lambda!r}"
            )
        if float(fusion_balance_lambda) < 0.0:
            raise ValueError(
                f"fusion_balance_lambda must be non-negative, got {fusion_balance_lambda!r}"
            )
        if float(ranking_lambda) < 0.0:
            raise ValueError(f"ranking_lambda must be non-negative, got {ranking_lambda!r}")
        if float(ranking_margin) < 0.0:
            raise ValueError(f"ranking_margin must be non-negative, got {ranking_margin!r}")
        if int(ranking_num_negatives) < 0:
            raise ValueError(
                f"ranking_num_negatives must be non-negative, got {ranking_num_negatives!r}"
            )
        if not 0.0 <= float(ranking_hard_negative_fraction) <= 1.0:
            raise ValueError(
                "ranking_hard_negative_fraction must be in [0, 1], "
                f"got {ranking_hard_negative_fraction!r}"
            )

        self.configured_lambda_ddi = float(lambda_ddi)
        self.ddi_regularizer = ddi_regularizer
        self.ddi_context = copy.deepcopy(ddi_context or {})
        self.ddi_active = self.ddi_regularizer is not None and bool(self.ddi_context.get("active", True))
        self.effective_lambda_ddi = self.configured_lambda_ddi if self.ddi_active else 0.0
        self.reduction = _validate_reduction(reduction)
        self.objective = _validate_objective(objective)
        self.focal_gamma = float(focal_gamma)
        self.asymmetric_gamma_pos = float(asymmetric_gamma_pos)
        self.asymmetric_gamma_neg = float(asymmetric_gamma_neg)
        self.asymmetric_clip = float(asymmetric_clip)
        self.fusion_entropy_lambda = float(fusion_entropy_lambda)
        self.fusion_balance_lambda = float(fusion_balance_lambda)
        self.ranking_lambda = float(ranking_lambda)
        self.ranking_objective = _validate_ranking_objective(ranking_objective)
        self.ranking_margin = float(ranking_margin)
        self.ranking_num_negatives = int(ranking_num_negatives)
        self.ranking_hard_negative_fraction = float(ranking_hard_negative_fraction)

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
        fusion_entropy_loss: torch.Tensor | None = None,
        fusion_balance_loss: torch.Tensor | None = None,
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

        if self.objective == "bce":
            prediction_loss_matrix = F.binary_cross_entropy_with_logits(
                drug_logits,
                targets,
                pos_weight=resolved_pos_weight,
                reduction="none",
            )
        elif self.objective == "focal_lite":
            prediction_loss_matrix = _binary_focal_loss_with_logits(
                drug_logits,
                targets,
                pos_weight=resolved_pos_weight,
                gamma=self.focal_gamma,
            )
        else:
            prediction_loss_matrix = _binary_asymmetric_focal_loss_with_logits(
                drug_logits,
                targets,
                pos_weight=resolved_pos_weight,
                gamma_pos=self.asymmetric_gamma_pos,
                gamma_neg=self.asymmetric_gamma_neg,
                clip=self.asymmetric_clip,
            )
        prediction_loss = _reduce_tensor(prediction_loss_matrix, self.reduction)
        if self.ranking_lambda > 0.0 and self.ranking_num_negatives > 0:
            ranking_loss = _sampled_pairwise_ranking_loss(
                drug_logits,
                targets,
                num_negatives=self.ranking_num_negatives,
                objective=self.ranking_objective,
                margin=self.ranking_margin,
                hard_negative_fraction=self.ranking_hard_negative_fraction,
            )
        else:
            ranking_loss = drug_logits.new_zeros(())
        weighted_ranking_loss = ranking_loss * self.ranking_lambda

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
        weighted_ddi_loss = ddi_loss * self.effective_lambda_ddi
        if fusion_entropy_loss is None:
            resolved_fusion_entropy = torch.zeros(
                (),
                device=drug_logits.device,
                dtype=drug_logits.dtype,
            )
        else:
            resolved_fusion_entropy = _reduce_tensor(
                fusion_entropy_loss.to(device=drug_logits.device, dtype=drug_logits.dtype),
                self.reduction,
            )
        if fusion_balance_loss is None:
            resolved_fusion_balance = torch.zeros(
                (),
                device=drug_logits.device,
                dtype=drug_logits.dtype,
            )
        else:
            resolved_fusion_balance = _reduce_tensor(
                fusion_balance_loss.to(device=drug_logits.device, dtype=drug_logits.dtype),
                self.reduction,
            )
        weighted_fusion_entropy_loss = resolved_fusion_entropy * self.fusion_entropy_lambda
        weighted_fusion_balance_loss = resolved_fusion_balance * self.fusion_balance_lambda
        total_loss = (
            prediction_loss
            + weighted_ranking_loss
            + weighted_ddi_loss
            + weighted_fusion_entropy_loss
            + weighted_fusion_balance_loss
        )

        return {
            "total_loss": total_loss,
            "prediction_loss": prediction_loss,
            "ranking_loss": ranking_loss,
            "weighted_ranking_loss": weighted_ranking_loss,
            "ddi_loss": ddi_loss,
            "weighted_ddi_loss": weighted_ddi_loss,
            "fusion_entropy_loss": resolved_fusion_entropy,
            "fusion_balance_loss": resolved_fusion_balance,
            "weighted_fusion_entropy_loss": weighted_fusion_entropy_loss,
            "weighted_fusion_balance_loss": weighted_fusion_balance_loss,
            "objective": self.objective,
            "asymmetric_gamma_pos": torch.tensor(
                self.asymmetric_gamma_pos,
                device=drug_logits.device,
                dtype=drug_logits.dtype,
            ),
            "asymmetric_gamma_neg": torch.tensor(
                self.asymmetric_gamma_neg,
                device=drug_logits.device,
                dtype=drug_logits.dtype,
            ),
            "asymmetric_clip": torch.tensor(
                self.asymmetric_clip,
                device=drug_logits.device,
                dtype=drug_logits.dtype,
            ),
            "ranking_objective": self.ranking_objective,
            "ranking_num_negatives": torch.tensor(
                self.ranking_num_negatives,
                device=drug_logits.device,
                dtype=drug_logits.dtype,
            ),
            "configured_ddi_lambda": torch.tensor(
                self.configured_lambda_ddi,
                device=drug_logits.device,
                dtype=drug_logits.dtype,
            ),
            "effective_ddi_lambda": torch.tensor(
                self.effective_lambda_ddi,
                device=drug_logits.device,
                dtype=drug_logits.dtype,
            ),
            "ranking_lambda": torch.tensor(
                self.ranking_lambda,
                device=drug_logits.device,
                dtype=drug_logits.dtype,
            ),
        }


__all__ = ["MedicationRecommendationLoss", "extract_last_valid_targets"]
