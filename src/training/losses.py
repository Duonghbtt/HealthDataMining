from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

_VALID_REDUCTIONS = {"mean", "sum", "none"}
_VALID_DDI_SCHEDULE_TYPES = {"constant", "linear", "cosine", "step"}
_LAMBDA_DDI_DISABLED_WARNING_SHOWN = False


def _tensor_debug_summary(name: str, value: torch.Tensor) -> str:
    resolved = value.detach()
    shape = tuple(resolved.shape)
    dtype = resolved.dtype
    if resolved.numel() == 0:
        return f"{name}: shape={shape} dtype={dtype} numel=0"
    if not (resolved.is_floating_point() or resolved.is_complex()):
        return (
            f"{name}: shape={shape} dtype={dtype} "
            f"min={resolved.min().item()} max={resolved.max().item()}"
        )
    nan_count = int(torch.isnan(resolved).sum().item())
    inf_count = int(torch.isinf(resolved).sum().item())
    finite = resolved[torch.isfinite(resolved)]
    if finite.numel() == 0:
        return (
            f"{name}: shape={shape} dtype={dtype} nan_count={nan_count} "
            f"inf_count={inf_count} finite_values=0"
        )
    return (
        f"{name}: shape={shape} dtype={dtype} nan_count={nan_count} "
        f"inf_count={inf_count} min={finite.min().item():.6g} "
        f"max={finite.max().item():.6g} mean={finite.mean().item():.6g}"
    )


def _validate_reduction(reduction: str) -> str:
    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(f"reduction must be one of {_VALID_REDUCTIONS}, got {reduction!r}")
    return reduction


def _reduce_per_sample(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError(f"Expected per-sample tensor with shape (B,), got {tuple(values.shape)}")
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    return values


def extract_last_valid_targets(target_drugs: torch.Tensor, visit_mask: torch.Tensor) -> torch.Tensor:
    """Extract current-visit targets from [B, T, D] or passthrough [B, D]."""

    if not isinstance(target_drugs, torch.Tensor):
        raise TypeError(f"target_drugs must be a torch.Tensor, got {type(target_drugs)!r}")
    if target_drugs.ndim == 2:
        return target_drugs
    if target_drugs.ndim != 3:
        raise ValueError(f"target_drugs must have shape (B, D) or (B, T, D), got {tuple(target_drugs.shape)}")
    if not isinstance(visit_mask, torch.Tensor):
        raise TypeError(f"visit_mask must be a torch.Tensor, got {type(visit_mask)!r}")
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


def _build_pos_weight_from_avg_pos(
    *,
    vocab_size: int,
    avg_pos: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")

    resolved_avg_pos = float(avg_pos)
    if resolved_avg_pos <= 0.0:
        raise ValueError(f"avg_pos must be > 0 to build pos_weight, got {resolved_avg_pos}")
    if resolved_avg_pos >= float(vocab_size):
        raise ValueError(
            "avg_pos must be smaller than vocab_size to build a positive pos_weight: "
            f"got avg_pos={resolved_avg_pos} and vocab_size={vocab_size}"
        )

    pos_weight_scalar = (float(vocab_size) - resolved_avg_pos) / resolved_avg_pos
    return torch.full((vocab_size,), pos_weight_scalar, device=device, dtype=dtype)


def _resolve_pos_weight(
    pos_weight: torch.Tensor | None,
    avg_pos: float | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    width: int,
) -> torch.Tensor | None:
    """Resolve class imbalance weighting from either an explicit tensor or avg_pos."""

    if pos_weight is not None and avg_pos is not None:
        raise ValueError("Provide either explicit pos_weight or avg_pos, but not both")
    if pos_weight is None:
        if avg_pos is None:
            return None
        return _build_pos_weight_from_avg_pos(
            vocab_size=width,
            avg_pos=avg_pos,
            device=device,
            dtype=dtype,
        )

    resolved = torch.as_tensor(pos_weight, device=device, dtype=dtype)
    if resolved.ndim != 1 or resolved.shape[0] != width:
        raise ValueError(
            "pos_weight must have shape (D,) matching drug logits width: "
            f"got {tuple(resolved.shape)} for {width}"
        )
    if not torch.isfinite(resolved).all():
        raise ValueError("pos_weight must contain only finite values")
    return resolved


def _warn_if_ddi_disabled(lambda_ddi: float) -> None:
    global _LAMBDA_DDI_DISABLED_WARNING_SHOWN
    if float(lambda_ddi) == 0.0 and not _LAMBDA_DDI_DISABLED_WARNING_SHOWN:
        print("WARNING: lambda_ddi=0, DDI loss is disabled. DDI rate will not be optimized.")
        _LAMBDA_DDI_DISABLED_WARNING_SHOWN = True


def _resolve_ddi_matrix(
    ddi_matrix: torch.Tensor | None,
    *,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if ddi_matrix is None:
        return None

    resolved = torch.as_tensor(ddi_matrix, device=device, dtype=dtype)
    if resolved.ndim != 2:
        raise ValueError(f"ddi_matrix must have shape (D, D), got {tuple(resolved.shape)}")
    if resolved.shape[0] != resolved.shape[1]:
        raise ValueError(f"ddi_matrix must be square, got {tuple(resolved.shape)}")
    if int(resolved.shape[0]) != int(width):
        raise ValueError(
            "ddi_matrix width must match drug logits width: "
            f"got {int(resolved.shape[0])} and {int(width)}"
        )
    if not torch.isfinite(resolved).all():
        raise ValueError("ddi_matrix must contain only finite values")
    return resolved


def _validate_optional_drug_probs(
    drug_probs: torch.Tensor,
    *,
    drug_logits: torch.Tensor,
    expected_probs: torch.Tensor,
) -> None:
    if tuple(drug_probs.shape) != tuple(drug_logits.shape):
        raise ValueError(
            "drug_probs must match drug_logits shape when provided: "
            f"got {tuple(drug_probs.shape)} and {tuple(drug_logits.shape)}"
        )
    if not torch.isfinite(drug_probs).all():
        raise ValueError("drug_probs must contain only finite values")
    if bool(((drug_probs < 0.0) | (drug_probs > 1.0)).any().item()):
        raise ValueError("drug_probs must contain values in [0, 1]")
    if not torch.allclose(drug_probs, expected_probs, rtol=1e-4, atol=1e-6):
        raise ValueError(
            "drug_probs must match torch.sigmoid(drug_logits); external probabilities are not authoritative"
        )


def _normalize_schedule_config(schedule_config: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(schedule_config or {})
    schedule_type = str(payload.get("type", "constant")).strip().lower()
    if schedule_type not in _VALID_DDI_SCHEDULE_TYPES:
        raise ValueError(
            f"ddi_schedule.type must be one of {_VALID_DDI_SCHEDULE_TYPES}, got {schedule_type!r}"
        )
    return {
        "enabled": bool(payload.get("enabled", False)),
        "type": schedule_type,
        "start_epoch": None if payload.get("start_epoch") is None else int(payload["start_epoch"]),
        "end_epoch": None if payload.get("end_epoch") is None else int(payload["end_epoch"]),
        "start_value": None if payload.get("start_value") is None else float(payload["start_value"]),
        "end_value": None if payload.get("end_value") is None else float(payload["end_value"]),
    }


def build_medication_loss_config(
    *,
    loss_config: Mapping[str, Any] | None = None,
    training_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize medication loss configuration for stage-wise training."""

    loss_payload = dict(loss_config or {})
    training_payload = dict(training_config or {})
    use_bce = bool(loss_payload.get("use_bce", True))
    if not use_bce:
        raise ValueError("Medication recommendation training requires BCEWithLogits as the primary loss.")

    margin_weight = float(loss_payload.get("margin_weight", 0.0))
    if margin_weight < 0.0:
        raise ValueError(f"margin_weight must be non-negative, got {margin_weight}")

    lambda_ddi = float(loss_payload.get("lambda_ddi", 0.0))
    if lambda_ddi < 0.0:
        raise ValueError(f"lambda_ddi must be non-negative, got {lambda_ddi}")

    stage1_epochs = int(loss_payload.get("stage1_epochs", training_payload.get("stage1_epochs", 0)))
    stage2_epochs = int(loss_payload.get("stage2_epochs", training_payload.get("stage2_epochs", 0)))
    if stage1_epochs < 0 or stage2_epochs < 0:
        raise ValueError("stage1_epochs and stage2_epochs must be non-negative")

    return {
        "use_bce": True,
        "use_margin_loss": bool(loss_payload.get("use_margin_loss", False)),
        "margin_weight": margin_weight,
        "use_ddi": bool(loss_payload.get("use_ddi", lambda_ddi > 0.0)),
        "lambda_ddi": lambda_ddi,
        "stage_training": bool(loss_payload.get("stage_training", False)),
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "ddi_schedule": _normalize_schedule_config(loss_payload.get("ddi_schedule")),
    }


def resolve_lambda_ddi_current(
    *,
    current_epoch: int | None,
    loss_config: Mapping[str, Any] | None,
) -> float:
    """Resolve the DDI regularization weight at the current epoch."""

    resolved_loss_config = build_medication_loss_config(loss_config=loss_config)
    if not bool(resolved_loss_config["use_ddi"]):
        return 0.0

    target_lambda = float(resolved_loss_config["lambda_ddi"])
    if target_lambda <= 0.0:
        return 0.0

    schedule_cfg = dict(resolved_loss_config["ddi_schedule"])
    stage_training = bool(resolved_loss_config["stage_training"])
    schedule_enabled = bool(schedule_cfg.get("enabled", False))
    if not stage_training and not schedule_enabled:
        return target_lambda

    stage2_start_epoch = int(schedule_cfg.get("start_epoch") or max(1, int(resolved_loss_config["stage1_epochs"]) + 1))
    if current_epoch is None:
        return 0.0 if stage_training or schedule_enabled else target_lambda
    epoch_value = int(current_epoch)
    if epoch_value < stage2_start_epoch:
        return 0.0
    if not schedule_enabled:
        return target_lambda

    schedule_type = str(schedule_cfg["type"])
    start_value = float(schedule_cfg["start_value"] if schedule_cfg["start_value"] is not None else 0.0)
    end_value = float(schedule_cfg["end_value"] if schedule_cfg["end_value"] is not None else target_lambda)
    end_epoch = int(
        schedule_cfg["end_epoch"]
        if schedule_cfg["end_epoch"] is not None
        else max(stage2_start_epoch, stage2_start_epoch + int(resolved_loss_config["stage2_epochs"]) - 1)
    )
    if schedule_type == "constant" or end_epoch <= stage2_start_epoch:
        return end_value
    if schedule_type == "step":
        return end_value if epoch_value >= end_epoch else start_value

    progress = float(epoch_value - stage2_start_epoch) / float(max(end_epoch - stage2_start_epoch, 1))
    progress = min(max(progress, 0.0), 1.0)
    if schedule_type == "linear":
        return start_value + progress * (end_value - start_value)
    if schedule_type == "cosine":
        cosine_progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        return start_value + cosine_progress * (end_value - start_value)
    raise ValueError(f"Unsupported ddi schedule type: {schedule_type!r}")


def _build_margin_targets(target_current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    positive_mask = target_current > 0.0
    margin_targets = torch.full(
        target_current.shape,
        fill_value=-1,
        device=target_current.device,
        dtype=torch.long,
    )
    for sample_index in range(target_current.shape[0]):
        positive_indices = torch.nonzero(positive_mask[sample_index], as_tuple=False).flatten()
        if positive_indices.numel() > 0:
            margin_targets[sample_index, : positive_indices.numel()] = positive_indices
    return margin_targets, positive_mask.any(dim=1)


def _compute_margin_loss_per_sample(
    *,
    drug_logits: torch.Tensor,
    target_current: torch.Tensor,
) -> torch.Tensor:
    margin_targets, has_positive_labels = _build_margin_targets(target_current)
    if not bool(has_positive_labels.any().item()):
        return torch.zeros(drug_logits.shape[0], device=drug_logits.device, dtype=drug_logits.dtype)

    margin_losses = F.multilabel_margin_loss(
        drug_logits,
        margin_targets,
        reduction="none",
    ).to(dtype=drug_logits.dtype)
    return torch.where(
        has_positive_labels,
        margin_losses,
        torch.zeros_like(margin_losses),
    )


def _coerce_current_epoch(current_epoch: int | torch.Tensor | None) -> int | None:
    if current_epoch is None:
        return None
    if isinstance(current_epoch, torch.Tensor):
        if current_epoch.numel() != 1:
            raise ValueError(f"current_epoch tensor must be scalar, got {tuple(current_epoch.shape)}")
        return int(current_epoch.detach().cpu().item())
    return int(current_epoch)


def _resolve_loss_config(
    *,
    loss_config: Mapping[str, Any] | None,
    lambda_ddi: float | None,
    use_margin_loss: bool | None,
    margin_weight: float | None,
    use_ddi: bool | None,
    stage_training: bool | None,
    stage1_epochs: int | None,
    stage2_epochs: int | None,
    ddi_schedule: Mapping[str, Any] | None,
) -> dict[str, Any]:
    resolved = build_medication_loss_config(loss_config=loss_config)
    if lambda_ddi is not None:
        resolved["lambda_ddi"] = float(lambda_ddi)
    if use_margin_loss is not None:
        resolved["use_margin_loss"] = bool(use_margin_loss)
    if margin_weight is not None:
        resolved["margin_weight"] = float(margin_weight)
    if resolved["margin_weight"] < 0.0:
        raise ValueError(f"margin_weight must be non-negative, got {resolved['margin_weight']}")
    if use_ddi is not None:
        resolved["use_ddi"] = bool(use_ddi)
    if stage_training is not None:
        resolved["stage_training"] = bool(stage_training)
    if stage1_epochs is not None:
        resolved["stage1_epochs"] = int(stage1_epochs)
    if stage2_epochs is not None:
        resolved["stage2_epochs"] = int(stage2_epochs)
    if ddi_schedule is not None:
        resolved["ddi_schedule"] = _normalize_schedule_config(ddi_schedule)
    return resolved


def compute_medication_losses(
    *,
    drug_logits: torch.Tensor,
    target_drugs: torch.Tensor,
    visit_mask: torch.Tensor | None = None,
    drug_probs: torch.Tensor | None = None,
    ddi_matrix: torch.Tensor | None = None,
    lambda_ddi: float | None = None,
    pos_weight: torch.Tensor | None = None,
    avg_pos: float | None = None,
    reduction: str = "mean",
    training: bool = True,
    loss_config: Mapping[str, Any] | None = None,
    current_epoch: int | torch.Tensor | None = None,
    use_margin_loss: bool | None = None,
    margin_weight: float | None = None,
    use_ddi: bool | None = None,
    stage_training: bool | None = None,
    stage1_epochs: int | None = None,
    stage2_epochs: int | None = None,
    ddi_schedule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute medication losses from raw logits.

    Class imbalance weighting can be supplied either via an explicit ``pos_weight``
    tensor with shape ``(D,)`` or via ``avg_pos`` so the function auto-builds a
    uniform per-class weight using the current medication vocabulary width.
    """

    resolved_reduction = _validate_reduction(reduction)
    if not isinstance(drug_logits, torch.Tensor):
        raise TypeError(f"drug_logits must be a torch.Tensor, got {type(drug_logits)!r}")
    if drug_logits.ndim != 2:
        raise ValueError(f"drug_logits must have shape (B, D), got {tuple(drug_logits.shape)}")
    if not torch.isfinite(drug_logits).all():
        raise ValueError(
            "drug_logits must contain only finite values; "
            f"{_tensor_debug_summary('drug_logits', drug_logits)}"
        )
    training = bool(training and torch.is_grad_enabled())
    assert drug_logits.requires_grad or not training, (
        "drug_logits must be raw logits, not sigmoid output"
    )
    resolved_loss_config = _resolve_loss_config(
        loss_config=loss_config,
        lambda_ddi=lambda_ddi,
        use_margin_loss=use_margin_loss,
        margin_weight=margin_weight,
        use_ddi=use_ddi,
        stage_training=stage_training,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        ddi_schedule=ddi_schedule,
    )
    if not bool(resolved_loss_config["use_ddi"]) or float(resolved_loss_config["lambda_ddi"]) == 0.0:
        _warn_if_ddi_disabled(float(resolved_loss_config["lambda_ddi"]))

    target_current = _resolve_targets(
        target_drugs,
        visit_mask,
        device=drug_logits.device,
        dtype=drug_logits.dtype,
    )
    if tuple(target_current.shape) != tuple(drug_logits.shape):
        raise ValueError(
            "Resolved targets must match drug_logits shape: "
            f"got {tuple(target_current.shape)} and {tuple(drug_logits.shape)}"
        )

    resolved_probs = torch.sigmoid(drug_logits)
    if drug_probs is not None:
        provided_probs = torch.as_tensor(drug_probs, device=drug_logits.device, dtype=drug_logits.dtype)
        _validate_optional_drug_probs(
            provided_probs,
            drug_logits=drug_logits,
            expected_probs=resolved_probs,
        )

    resolved_pos_weight = _resolve_pos_weight(
        pos_weight,
        avg_pos,
        device=drug_logits.device,
        dtype=drug_logits.dtype,
        width=int(drug_logits.shape[1]),
    )
    resolved_ddi_matrix = _resolve_ddi_matrix(
        ddi_matrix,
        width=int(drug_logits.shape[1]),
        device=resolved_probs.device,
        dtype=resolved_probs.dtype,
    )
    if (
        bool(resolved_loss_config["use_ddi"])
        and float(resolved_loss_config["lambda_ddi"]) > 0.0
        and resolved_ddi_matrix is None
    ):
        raise ValueError("DDI regularization is enabled but ddi_matrix was not provided.")
    lambda_ddi_current = resolve_lambda_ddi_current(
        current_epoch=_coerce_current_epoch(current_epoch),
        loss_config=resolved_loss_config,
    )
    margin_weight_value = (
        float(resolved_loss_config["margin_weight"])
        if bool(resolved_loss_config["use_margin_loss"])
        else 0.0
    )

    # Multi-label BCE prediction loss is computed directly on raw pre-sigmoid logits.
    prediction_loss_matrix = F.binary_cross_entropy_with_logits(
        drug_logits,
        target_current,
        pos_weight=resolved_pos_weight,
        reduction="none",
    )
    prediction_loss_per_sample = prediction_loss_matrix.mean(dim=1)
    prediction_loss = _reduce_per_sample(prediction_loss_per_sample, resolved_reduction)

    if bool(resolved_loss_config["use_margin_loss"]):
        margin_loss_per_sample = _compute_margin_loss_per_sample(
            drug_logits=drug_logits,
            target_current=target_current,
        )
    else:
        margin_loss_per_sample = torch.zeros(
            drug_logits.shape[0],
            device=drug_logits.device,
            dtype=drug_logits.dtype,
        )
    margin_loss = _reduce_per_sample(margin_loss_per_sample, resolved_reduction)
    weighted_margin_loss = margin_loss * margin_weight_value

    # DDI loss computes expected co-medication pair strength under predicted probabilities
    # and penalizes predicted pairs that are known to interact in the DDI matrix.
    if resolved_ddi_matrix is None or not bool(resolved_loss_config["use_ddi"]):
        ddi_per_sample = torch.zeros(drug_logits.shape[0], device=drug_logits.device, dtype=drug_logits.dtype)
    else:
        pred_pairs = resolved_probs.unsqueeze(2) * resolved_probs.unsqueeze(1)
        ddi_per_sample = (pred_pairs * resolved_ddi_matrix.unsqueeze(0)).sum(dim=(1, 2))

    ddi_loss = _reduce_per_sample(ddi_per_sample, resolved_reduction)
    weighted_ddi_loss = ddi_loss * float(lambda_ddi_current)
    # Total loss combines prediction quality and DDI regularization.
    total_loss = prediction_loss + weighted_margin_loss + weighted_ddi_loss
    lambda_ddi_tensor = torch.tensor(
        float(lambda_ddi_current),
        device=drug_logits.device,
        dtype=drug_logits.dtype,
    )
    return {
        "pred_loss": prediction_loss,
        "prediction_loss": prediction_loss,
        "pred_bce_loss": prediction_loss,
        "margin_loss": margin_loss,
        "weighted_margin_loss": weighted_margin_loss,
        "ddi_loss": ddi_loss,
        "weighted_ddi_loss": weighted_ddi_loss,
        "lambda_ddi_current": lambda_ddi_tensor,
        "total_loss": total_loss,
        "target_current": target_current,
    }


class MedicationRecommendationLoss(nn.Module):
    """Compatibility wrapper around ``compute_medication_losses``."""

    def __init__(
        self,
        *,
        lambda_ddi: float | None = None,
        ddi_matrix: torch.Tensor | None = None,
        pos_weight: torch.Tensor | None = None,
        avg_pos: float | None = None,
        reduction: str = "mean",
        loss_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.reduction = _validate_reduction(reduction)
        if pos_weight is not None and avg_pos is not None:
            raise ValueError("Provide either explicit pos_weight or avg_pos, but not both")
        self.avg_pos = None if avg_pos is None else float(avg_pos)
        self.loss_config = build_medication_loss_config(loss_config=loss_config)
        if lambda_ddi is not None:
            self.loss_config["lambda_ddi"] = float(lambda_ddi)
        if ddi_matrix is None:
            self.register_buffer("ddi_matrix", None)
        else:
            self.register_buffer("ddi_matrix", torch.as_tensor(ddi_matrix, dtype=torch.float32))
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32))

    def forward(
        self,
        *,
        drug_logits: torch.Tensor,
        target_drugs: torch.Tensor,
        visit_mask: torch.Tensor | None = None,
        drug_probs: torch.Tensor | None = None,
        current_epoch: int | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        return compute_medication_losses(
            drug_logits=drug_logits,
            target_drugs=target_drugs,
            visit_mask=visit_mask,
            drug_probs=drug_probs,
            ddi_matrix=self.ddi_matrix,
            lambda_ddi=float(self.loss_config["lambda_ddi"]),
            pos_weight=self.pos_weight,
            avg_pos=self.avg_pos,
            reduction=self.reduction,
            training=self.training,
            loss_config=self.loss_config,
            current_epoch=current_epoch,
        )


__all__ = [
    "MedicationRecommendationLoss",
    "build_medication_loss_config",
    "compute_medication_losses",
    "extract_last_valid_targets",
    "resolve_lambda_ddi_current",
]
