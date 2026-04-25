from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Mapping, Sequence

import torch

from .metrics import (
    binarize_predictions,
    compute_ddi_rate,
    compute_samplewise_f1,
    compute_samplewise_jaccard,
    multilabel_prauc,
)


_VALID_THRESHOLD_METRICS = {"f1", "jaccard"}
_VALIDATION_SWEEP_SOURCE = "validation_sweep"


def normalize_threshold_tuning_config(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(payload or {})
    metric = str(raw.get("metric", "f1")).strip().lower()
    tie_breaker = str(raw.get("tie_breaker", "jaccard")).strip().lower()
    if metric not in _VALID_THRESHOLD_METRICS:
        raise ValueError(f"Unsupported threshold tuning metric `{metric}`")
    if tie_breaker not in _VALID_THRESHOLD_METRICS:
        raise ValueError(f"Unsupported threshold tuning tie_breaker `{tie_breaker}`")

    raw_candidates = raw.get("candidates", [0.5])
    candidates = [float(value) for value in raw_candidates]
    if not candidates:
        raise ValueError("threshold tuning candidates must not be empty")
    for threshold in candidates:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"threshold candidate must be within [0, 1], got {threshold!r}")

    return {
        "enabled": bool(raw.get("enabled", False)),
        "metric": metric,
        "tie_breaker": tie_breaker,
        "split": str(raw.get("split", "val")).strip().lower(),
        "candidates": candidates,
    }


def _candidate_metric_value(candidate: Mapping[str, Any], metric_name: str) -> float:
    value = candidate.get(metric_name)
    if value is None:
        return float("-inf")
    return float(value)


def sweep_multilabel_thresholds(
    *,
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    candidates: Sequence[float],
    metric: str = "f1",
    tie_breaker: str = "jaccard",
    ddi_matrix: torch.Tensor | None = None,
) -> dict[str, Any]:
    resolved_metric = str(metric).strip().lower()
    resolved_tie_breaker = str(tie_breaker).strip().lower()
    if resolved_metric not in _VALID_THRESHOLD_METRICS:
        raise ValueError(f"Unsupported sweep metric `{resolved_metric}`")
    if resolved_tie_breaker not in _VALID_THRESHOLD_METRICS:
        raise ValueError(f"Unsupported sweep tie_breaker `{resolved_tie_breaker}`")

    candidate_metrics: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    for raw_threshold in candidates:
        threshold = float(raw_threshold)
        y_pred_binary = binarize_predictions(y_score, threshold)
        sample_f1 = compute_samplewise_f1(y_true, y_pred_binary)
        sample_jaccard = compute_samplewise_jaccard(y_true, y_pred_binary)
        row: dict[str, Any] = {
            "threshold": threshold,
            "f1": float(sample_f1.mean().item()),
            "jaccard": float(sample_jaccard.mean().item()),
            "avg_predicted_drugs": float(y_pred_binary.sum(dim=1, dtype=torch.float32).mean().item()),
            "avg_true_drugs": float(y_true.sum(dim=1, dtype=torch.float32).mean().item()),
        }
        if ddi_matrix is not None:
            row.update(compute_ddi_rate(y_pred_binary, ddi_matrix))

        candidate_metrics.append(row)
        if best_row is None:
            best_row = row
            continue

        best_metric = _candidate_metric_value(best_row, resolved_metric)
        current_metric = _candidate_metric_value(row, resolved_metric)
        if current_metric > best_metric:
            best_row = row
            continue
        if current_metric < best_metric:
            continue

        best_tie = _candidate_metric_value(best_row, resolved_tie_breaker)
        current_tie = _candidate_metric_value(row, resolved_tie_breaker)
        if current_tie > best_tie:
            best_row = row

    assert best_row is not None
    return {
        "metric": resolved_metric,
        "tie_breaker": resolved_tie_breaker,
        "best_threshold": float(best_row["threshold"]),
        "best_metrics": copy.deepcopy(best_row),
        "candidate_metrics": candidate_metrics,
        "prauc": float(multilabel_prauc(y_true, y_score)),
    }


def _thresholds_match(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-8)


def _normalized_checkpoint_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(selection))
    if normalized.get("best_threshold") is not None:
        normalized["best_threshold"] = float(normalized["best_threshold"])
    return normalized


def _build_threshold_metadata_mismatch(
    *,
    effective_threshold: float,
    selection_best_threshold: float,
    resolved_threshold: float,
    resolved_threshold_source: str,
) -> dict[str, Any]:
    return {
        "effective_threshold": float(effective_threshold),
        "threshold_selection_best_threshold": float(selection_best_threshold),
        "resolved_threshold": float(resolved_threshold),
        "resolved_threshold_source": str(resolved_threshold_source),
    }


def resolve_threshold_from_checkpoint(checkpoint_payload: Mapping[str, Any]) -> tuple[float | None, str | None, dict[str, Any] | None]:
    selection = checkpoint_payload.get("threshold_selection")
    selection_payload = None
    selection_best_threshold = None
    selection_source = ""
    if isinstance(selection, Mapping) and selection.get("best_threshold") is not None:
        selection_payload = _normalized_checkpoint_selection(selection)
        selection_best_threshold = float(selection_payload["best_threshold"])
        selection_source = str(selection_payload.get("source", "")).strip().lower()

    effective_threshold = checkpoint_payload.get("effective_threshold")
    resolved_effective_threshold = None if effective_threshold is None else float(effective_threshold)

    if selection_best_threshold is not None and selection_source == _VALIDATION_SWEEP_SOURCE:
        if resolved_effective_threshold is not None:
            selection_payload["effective_threshold"] = float(resolved_effective_threshold)
            if not _thresholds_match(resolved_effective_threshold, selection_best_threshold):
                selection_payload["checkpoint_threshold_metadata_mismatch"] = _build_threshold_metadata_mismatch(
                    effective_threshold=resolved_effective_threshold,
                    selection_best_threshold=selection_best_threshold,
                    resolved_threshold=selection_best_threshold,
                    resolved_threshold_source="checkpoint.threshold_selection.best_threshold",
                )
                warnings.warn(
                    "Checkpoint threshold metadata mismatch: "
                    f"effective_threshold={resolved_effective_threshold} "
                    f"but threshold_selection.best_threshold={selection_best_threshold}. "
                    "Preferring checkpoint.threshold_selection.best_threshold because "
                    "threshold_selection.source=validation_sweep.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return (
            selection_best_threshold,
            "checkpoint.threshold_selection.best_threshold",
            selection_payload,
        )

    if resolved_effective_threshold is not None:
        resolved_selection = {
            "source": "checkpoint.effective_threshold",
            "best_threshold": float(resolved_effective_threshold),
        }
        if selection_payload is not None:
            resolved_selection["checkpoint_threshold_selection"] = selection_payload
            if selection_best_threshold is not None and not _thresholds_match(
                resolved_effective_threshold,
                selection_best_threshold,
            ):
                resolved_selection["checkpoint_threshold_metadata_mismatch"] = _build_threshold_metadata_mismatch(
                    effective_threshold=resolved_effective_threshold,
                    selection_best_threshold=selection_best_threshold,
                    resolved_threshold=resolved_effective_threshold,
                    resolved_threshold_source="checkpoint.effective_threshold",
                )
        return (
            resolved_effective_threshold,
            "checkpoint.effective_threshold",
            resolved_selection,
        )

    if selection_payload is not None:
        return (
            selection_best_threshold,
            "checkpoint.threshold_selection.best_threshold",
            selection_payload,
        )
    return None, None, None


def resolve_effective_threshold(
    *,
    cli_threshold: float | None,
    checkpoint_payload: Mapping[str, Any] | None,
    config_threshold: float,
) -> tuple[float, str, dict[str, Any] | None]:
    if cli_threshold is not None:
        threshold = float(cli_threshold)
        return (
            threshold,
            "cli",
            {
                "source": "cli",
                "best_threshold": threshold,
            },
        )

    if checkpoint_payload is not None:
        checkpoint_threshold, threshold_source, selection = resolve_threshold_from_checkpoint(checkpoint_payload)
        if checkpoint_threshold is not None and threshold_source is not None:
            return float(checkpoint_threshold), threshold_source, selection

    threshold = float(config_threshold)
    return (
        threshold,
        "config.prediction.threshold",
        {
            "source": "config.prediction.threshold",
            "best_threshold": threshold,
        },
    )


__all__ = [
    "normalize_threshold_tuning_config",
    "resolve_effective_threshold",
    "resolve_threshold_from_checkpoint",
    "sweep_multilabel_thresholds",
]
