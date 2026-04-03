from __future__ import annotations

from typing import Any

import torch


def _validate_binary_matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape (N, D), got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_same_shape(name_a: str, a: torch.Tensor, name_b: str, b: torch.Tensor) -> None:
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(
            f"{name_a} and {name_b} must share the same shape, got {tuple(a.shape)} and {tuple(b.shape)}"
        )


def _as_bool_matrix(value: torch.Tensor) -> torch.Tensor:
    return value.to(dtype=torch.bool)


def _resolve_ddi_upper(ddi_matrix: torch.Tensor) -> torch.Tensor:
    matrix = _validate_binary_matrix("ddi_matrix", torch.as_tensor(ddi_matrix, dtype=torch.float32))
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"ddi_matrix must be square, got {tuple(matrix.shape)}")
    ddi_bool = (matrix > 0).to(dtype=torch.bool)
    ddi_bool = torch.logical_or(ddi_bool, ddi_bool.transpose(0, 1))
    ddi_bool.fill_diagonal_(False)
    return torch.triu(ddi_bool, diagonal=1)


def compute_samplewise_jaccard(
    y_true: torch.Tensor,
    y_pred_binary: torch.Tensor,
) -> torch.Tensor:
    """Compute sample-wise Jaccard scores for multi-label predictions."""

    true_tensor = _validate_binary_matrix("y_true", y_true)
    pred_tensor = _validate_binary_matrix("y_pred_binary", y_pred_binary)
    _validate_same_shape("y_true", true_tensor, "y_pred_binary", pred_tensor)

    true_mask = _as_bool_matrix(true_tensor)
    pred_mask = _as_bool_matrix(pred_tensor)
    intersection = torch.logical_and(true_mask, pred_mask).sum(dim=1, dtype=torch.float32)
    union = torch.logical_or(true_mask, pred_mask).sum(dim=1, dtype=torch.float32)
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def compute_samplewise_f1(
    y_true: torch.Tensor,
    y_pred_binary: torch.Tensor,
) -> torch.Tensor:
    """Compute sample-wise F1 scores for multi-label predictions."""

    true_tensor = _validate_binary_matrix("y_true", y_true)
    pred_tensor = _validate_binary_matrix("y_pred_binary", y_pred_binary)
    _validate_same_shape("y_true", true_tensor, "y_pred_binary", pred_tensor)

    true_mask = _as_bool_matrix(true_tensor)
    pred_mask = _as_bool_matrix(pred_tensor)
    true_positive = torch.logical_and(true_mask, pred_mask).sum(dim=1, dtype=torch.float32)
    false_positive = torch.logical_and(~true_mask, pred_mask).sum(dim=1, dtype=torch.float32)
    false_negative = torch.logical_and(true_mask, ~pred_mask).sum(dim=1, dtype=torch.float32)
    denominator = 2.0 * true_positive + false_positive + false_negative
    return torch.where(denominator > 0, (2.0 * true_positive) / denominator, torch.ones_like(denominator))


def binarize_predictions(drug_probs: torch.Tensor, threshold: float) -> torch.Tensor:
    """Threshold model probabilities into binary multi-label predictions."""

    probs = _validate_binary_matrix("drug_probs", drug_probs)
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")
    return probs >= float(threshold)


def multilabel_jaccard(y_true: torch.Tensor, y_pred_binary: torch.Tensor) -> float:
    """Sample-wise Jaccard averaged over the batch."""

    return float(compute_samplewise_jaccard(y_true, y_pred_binary).mean().item())


def multilabel_f1(y_true: torch.Tensor, y_pred_binary: torch.Tensor) -> float:
    """Sample-wise F1 averaged over the batch."""

    return float(compute_samplewise_f1(y_true, y_pred_binary).mean().item())


def _binary_average_precision(y_true: torch.Tensor, y_score: torch.Tensor) -> float | None:
    labels = torch.as_tensor(y_true, dtype=torch.float32)
    scores = torch.as_tensor(y_score, dtype=torch.float32)
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError(
            f"Average precision expects 1D tensors, got {tuple(labels.shape)} and {tuple(scores.shape)}"
        )
    if labels.shape[0] != scores.shape[0]:
        raise ValueError(
            f"Average precision expects aligned shapes, got {tuple(labels.shape)} and {tuple(scores.shape)}"
        )

    positive_count = int((labels > 0).sum().item())
    if positive_count <= 0:
        return None

    sorted_indices = torch.argsort(scores, descending=True, stable=True)
    sorted_labels = (labels[sorted_indices] > 0).to(dtype=torch.float32)
    cumulative_true_positive = torch.cumsum(sorted_labels, dim=0)
    ranks = torch.arange(1, sorted_labels.shape[0] + 1, dtype=torch.float32, device=sorted_labels.device)
    precision_at_k = cumulative_true_positive / ranks
    average_precision = (precision_at_k * sorted_labels).sum() / float(positive_count)
    return float(average_precision.item())


def multilabel_prauc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """Macro average precision over drug labels with at least one positive example."""

    true_tensor = _validate_binary_matrix("y_true", y_true)
    score_tensor = _validate_binary_matrix("y_score", y_score)
    _validate_same_shape("y_true", true_tensor, "y_score", score_tensor)

    per_label_scores: list[float] = []
    for label_index in range(true_tensor.shape[1]):
        ap_score = _binary_average_precision(true_tensor[:, label_index], score_tensor[:, label_index])
        if ap_score is not None:
            per_label_scores.append(ap_score)
    if not per_label_scores:
        return 0.0
    return float(sum(per_label_scores) / float(len(per_label_scores)))


def compute_ddi_flags(y_pred_binary: torch.Tensor, ddi_matrix: torch.Tensor) -> torch.Tensor:
    """Return a boolean flag for whether each sample contains at least one DDI pair."""

    pred_tensor = _validate_binary_matrix("y_pred_binary", y_pred_binary)
    ddi_upper = _resolve_ddi_upper(ddi_matrix)
    if pred_tensor.shape[1] != ddi_upper.shape[0]:
        raise ValueError(
            "y_pred_binary width must match ddi_matrix width: "
            f"got {int(pred_tensor.shape[1])} and {int(ddi_upper.shape[0])}"
        )

    pred_mask = _as_bool_matrix(pred_tensor).to(device=ddi_upper.device)
    flags = torch.zeros(pred_mask.shape[0], dtype=torch.bool, device=ddi_upper.device)
    for sample_index in range(pred_mask.shape[0]):
        predicted_indices = torch.nonzero(pred_mask[sample_index], as_tuple=False).flatten()
        if predicted_indices.numel() < 2:
            continue
        sample_ddi = ddi_upper.index_select(0, predicted_indices).index_select(1, predicted_indices)
        flags[sample_index] = bool(sample_ddi.any().item())
    return flags


def compute_ddi_rate(y_pred_binary: torch.Tensor, ddi_matrix: torch.Tensor) -> dict[str, float]:
    """Compute dataset-level DDI rate from binary medication predictions."""

    pred_tensor = _validate_binary_matrix("y_pred_binary", y_pred_binary)
    ddi_upper = _resolve_ddi_upper(ddi_matrix)
    if pred_tensor.shape[1] != ddi_upper.shape[0]:
        raise ValueError(
            "y_pred_binary width must match ddi_matrix width: "
            f"got {int(pred_tensor.shape[1])} and {int(ddi_upper.shape[0])}"
        )

    pred_mask = _as_bool_matrix(pred_tensor).to(device=ddi_upper.device)
    total_predicted_pairs = 0.0
    total_interacting_pairs = 0.0
    patients_with_ddi = 0.0

    for sample_index in range(pred_mask.shape[0]):
        predicted_indices = torch.nonzero(pred_mask[sample_index], as_tuple=False).flatten()
        predicted_count = int(predicted_indices.numel())
        if predicted_count < 2:
            continue

        sample_total_pairs = float(predicted_count * (predicted_count - 1) // 2)
        sample_ddi = ddi_upper.index_select(0, predicted_indices).index_select(1, predicted_indices)
        sample_interacting_pairs = float(sample_ddi.sum(dtype=torch.float32).item())

        total_predicted_pairs += sample_total_pairs
        total_interacting_pairs += sample_interacting_pairs
        if sample_interacting_pairs > 0.0:
            patients_with_ddi += 1.0

    ddi_rate = 0.0 if total_predicted_pairs <= 0.0 else total_interacting_pairs / total_predicted_pairs
    return {
        "ddi_rate": float(ddi_rate),
        "total_predicted_pairs": float(total_predicted_pairs),
        "total_interacting_pairs": float(total_interacting_pairs),
        "patients_with_ddi": float(patients_with_ddi),
        "num_samples": float(pred_tensor.shape[0]),
    }


def compute_core_metrics(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    threshold: float,
    ddi_matrix: torch.Tensor,
) -> dict[str, float]:
    """Compute the core evaluation metric bundle for medication recommendation."""

    y_pred_binary = binarize_predictions(y_score, threshold)
    ddi_summary = compute_ddi_rate(y_pred_binary, ddi_matrix)
    return {
        "jaccard": multilabel_jaccard(y_true, y_pred_binary),
        "f1": multilabel_f1(y_true, y_pred_binary),
        "prauc": multilabel_prauc(y_true, y_score),
        **ddi_summary,
    }


__all__ = [
    "binarize_predictions",
    "compute_core_metrics",
    "compute_ddi_flags",
    "compute_ddi_rate",
    "compute_samplewise_f1",
    "compute_samplewise_jaccard",
    "multilabel_f1",
    "multilabel_jaccard",
    "multilabel_prauc",
]
