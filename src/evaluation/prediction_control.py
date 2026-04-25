from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from .metrics import binarize_predictions
from .thresholding import (
    normalize_threshold_tuning_config,
    resolve_effective_threshold,
    sweep_multilabel_thresholds,
)


_VALID_PREDICTION_MODES = {"threshold", "calibrated_threshold", "top_k"}
_VALID_TOP_K_STRATEGIES = {"fixed", "avg_train_drugs"}


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def normalize_prediction_config(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = copy.deepcopy(dict(payload or {}))
    mode = str(raw.get("mode", "threshold")).strip().lower()
    if mode not in _VALID_PREDICTION_MODES:
        raise ValueError(f"Unsupported prediction mode `{mode}`")

    top_k_strategy = str(raw.get("top_k_strategy", "fixed")).strip().lower()
    if top_k_strategy not in _VALID_TOP_K_STRATEGIES:
        raise ValueError(f"Unsupported top_k strategy `{top_k_strategy}`")

    top_k = raw.get("top_k", 10)
    resolved_top_k = None if top_k is None else int(top_k)
    if resolved_top_k is not None and resolved_top_k <= 0:
        raise ValueError(f"prediction.top_k must be positive, got {resolved_top_k!r}")

    threshold = float(raw.get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"prediction.threshold must be within [0, 1], got {threshold!r}")

    calibration_raw = dict(raw.get("calibration") or {})
    calibration = normalize_threshold_tuning_config(calibration_raw)
    calibration["enabled"] = bool(calibration_raw.get("enabled", False))

    return {
        "mode": mode,
        "top_k": resolved_top_k,
        "top_k_strategy": top_k_strategy,
        "threshold": threshold,
        "calibration": calibration,
    }


def resolve_requested_prediction_mode(
    *,
    cli_prediction_mode: str | None,
    cli_prediction_top_k: int | None,
    cli_threshold: float | None,
    prediction_config: Mapping[str, Any],
) -> str:
    if cli_prediction_top_k is not None:
        return "top_k"
    if cli_threshold is not None:
        return "threshold"

    normalized = normalize_prediction_config(prediction_config)
    if cli_prediction_mode is not None:
        resolved = str(cli_prediction_mode).strip().lower()
        if resolved not in _VALID_PREDICTION_MODES:
            raise ValueError(f"Unsupported prediction mode `{resolved}`")
        return resolved

    if normalized["mode"] == "top_k":
        return "top_k"
    if normalized["mode"] == "calibrated_threshold":
        return "calibrated_threshold"
    if bool(normalized["calibration"]["enabled"]):
        return "calibrated_threshold"
    return "threshold"


def prediction_mode_requires_calibration(
    *,
    cli_prediction_mode: str | None,
    cli_prediction_top_k: int | None,
    cli_threshold: float | None,
    prediction_config: Mapping[str, Any],
) -> bool:
    return (
        resolve_requested_prediction_mode(
            cli_prediction_mode=cli_prediction_mode,
            cli_prediction_top_k=cli_prediction_top_k,
            cli_threshold=cli_threshold,
            prediction_config=prediction_config,
        )
        == "calibrated_threshold"
    )


def prediction_mode_requires_train_cardinality(
    *,
    cli_prediction_mode: str | None,
    cli_prediction_top_k: int | None,
    cli_threshold: float | None,
    prediction_config: Mapping[str, Any],
) -> bool:
    normalized = normalize_prediction_config(prediction_config)
    resolved_mode = resolve_requested_prediction_mode(
        cli_prediction_mode=cli_prediction_mode,
        cli_prediction_top_k=cli_prediction_top_k,
        cli_threshold=cli_threshold,
        prediction_config=prediction_config,
    )
    return (
        resolved_mode == "top_k"
        and cli_prediction_top_k is None
        and str(normalized["top_k_strategy"]).strip().lower() == "avg_train_drugs"
    )


def collect_model_predictions(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    decoder_top_k: int | None,
    include_num_steps: bool = False,
) -> dict[str, Any]:
    collected_probs: list[torch.Tensor] = []
    collected_targets: list[torch.Tensor] = []
    collected_num_steps: list[torch.Tensor] = []
    subject_ids: list[int] = []
    hadm_ids: list[int] = []
    stay_ids: list[int] = []

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch_on_device = _move_batch_to_device(batch, device)
            outputs = model(
                batch_on_device,
                mode="core",
                decoder_top_k=decoder_top_k,
            )
            drug_probs = outputs.get("drug_probs")
            final_target_drugs = outputs.get("final_target_drugs")
            if drug_probs is None:
                raise RuntimeError("Model did not return `drug_probs` during evaluation.")
            if final_target_drugs is None:
                raise RuntimeError("Model did not return `final_target_drugs` during evaluation.")

            collected_probs.append(drug_probs.detach().cpu())
            collected_targets.append(final_target_drugs.detach().cpu())
            subject_ids.extend(int(value) for value in batch.get("subject_ids", []))
            hadm_ids.extend(int(value) for value in batch.get("hadm_ids", []))
            stay_ids.extend(int(value) for value in batch.get("stay_ids", []))
            if include_num_steps:
                collected_num_steps.append(batch_on_device["visit_mask"].sum(dim=1).detach().cpu())

    if not collected_probs or not collected_targets:
        raise ValueError("Evaluation dataloader produced no batches")

    payload: dict[str, Any] = {
        "probs": torch.cat(collected_probs, dim=0),
        "targets": torch.cat(collected_targets, dim=0),
        "subject_ids": subject_ids,
        "hadm_ids": hadm_ids,
        "stay_ids": stay_ids,
    }
    if include_num_steps:
        payload["num_steps"] = torch.cat(collected_num_steps, dim=0).to(dtype=torch.long)
    return payload


def compute_average_target_cardinality_from_dataloader(dataloader: DataLoader) -> float:
    total_positive = 0.0
    total_samples = 0
    for batch in dataloader:
        targets = batch.get("final_target_drugs")
        if targets is None:
            raise KeyError("Expected dataloader batch to include `final_target_drugs`.")
        targets_tensor = torch.as_tensor(targets, dtype=torch.float32)
        if targets_tensor.ndim != 2:
            raise ValueError(
                f"Expected `final_target_drugs` to have shape (N, D), got {tuple(targets_tensor.shape)}"
            )
        total_positive += float(targets_tensor.sum().item())
        total_samples += int(targets_tensor.shape[0])
    if total_samples <= 0:
        raise ValueError("Cannot compute average target cardinality from an empty dataloader")
    return total_positive / float(total_samples)


def _resolve_fixed_top_k(value: int, *, num_labels: int) -> int:
    return max(1, min(int(value), int(num_labels)))


def _resolve_avg_train_drugs_top_k(value: float, *, num_labels: int) -> int:
    rounded = int(math.floor(float(value) + 0.5))
    return _resolve_fixed_top_k(rounded, num_labels=num_labels)


def binarize_top_k_predictions(drug_probs: torch.Tensor, top_k: int) -> torch.Tensor:
    probs = torch.as_tensor(drug_probs, dtype=torch.float32)
    if probs.ndim != 2:
        raise ValueError(f"drug_probs must have shape (N, D), got {tuple(probs.shape)}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k!r}")
    if probs.shape[1] <= 0:
        raise ValueError("drug_probs must contain at least one label column")

    resolved_top_k = min(int(top_k), int(probs.shape[1]))
    top_indices = torch.topk(probs, k=resolved_top_k, dim=1, largest=True, sorted=False).indices
    predictions = torch.zeros_like(probs, dtype=torch.bool)
    predictions.scatter_(1, top_indices, True)
    return predictions


def resolve_prediction_control(
    *,
    prediction_config: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any] | None,
    cli_threshold: float | None,
    cli_prediction_mode: str | None,
    cli_prediction_top_k: int | None,
    eval_probs: torch.Tensor,
    eval_targets: torch.Tensor,
    ddi_matrix: torch.Tensor | None,
    calibration_probs: torch.Tensor | None = None,
    calibration_targets: torch.Tensor | None = None,
    avg_train_drugs: float | None = None,
) -> dict[str, Any]:
    normalized = normalize_prediction_config(prediction_config)
    resolved_mode = resolve_requested_prediction_mode(
        cli_prediction_mode=cli_prediction_mode,
        cli_prediction_top_k=cli_prediction_top_k,
        cli_threshold=cli_threshold,
        prediction_config=normalized,
    )

    if resolved_mode == "top_k":
        if cli_prediction_top_k is not None:
            top_k = _resolve_fixed_top_k(cli_prediction_top_k, num_labels=int(eval_probs.shape[1]))
            top_k_source = "cli"
            top_k_strategy = "fixed"
        elif normalized["top_k_strategy"] == "avg_train_drugs":
            if avg_train_drugs is None:
                raise ValueError("avg_train_drugs is required when prediction.top_k_strategy=avg_train_drugs")
            top_k = _resolve_avg_train_drugs_top_k(avg_train_drugs, num_labels=int(eval_probs.shape[1]))
            top_k_source = "train.avg_true_drugs"
            top_k_strategy = "avg_train_drugs"
        else:
            if normalized["top_k"] is None:
                raise ValueError("prediction.top_k must be set when prediction.top_k_strategy=fixed")
            top_k = _resolve_fixed_top_k(normalized["top_k"], num_labels=int(eval_probs.shape[1]))
            top_k_source = "config.prediction.top_k"
            top_k_strategy = "fixed"

        return {
            "prediction_mode": "top_k",
            "prediction_control": {
                "mode": "top_k",
                "top_k": int(top_k),
                "top_k_source": str(top_k_source),
                "top_k_strategy": str(top_k_strategy),
                "avg_train_drugs": None if avg_train_drugs is None else float(avg_train_drugs),
                "avg_train_drugs_source": (
                    None
                    if avg_train_drugs is None
                    else "computed_from_train_split"
                ),
            },
            "binary_predictions": binarize_top_k_predictions(eval_probs, top_k),
            "threshold": None,
            "threshold_source": None,
            "threshold_selection": None,
        }

    if resolved_mode == "calibrated_threshold":
        calibration = dict(normalized["calibration"])
        if calibration_probs is None or calibration_targets is None:
            raise ValueError("Calibration predictions are required for calibrated_threshold mode")
        sweep_payload = sweep_multilabel_thresholds(
            y_true=calibration_targets,
            y_score=calibration_probs,
            candidates=calibration["candidates"],
            metric=calibration["metric"],
            tie_breaker=calibration["tie_breaker"],
            ddi_matrix=ddi_matrix,
        )
        threshold = float(sweep_payload["best_threshold"])
        threshold_selection = {
            "source": "evaluation.calibration",
            "split": str(calibration["split"]),
            "metric": str(sweep_payload["metric"]),
            "tie_breaker": str(sweep_payload["tie_breaker"]),
            "candidates": [float(value) for value in calibration["candidates"]],
            "best_threshold": threshold,
            "best_metrics": copy.deepcopy(dict(sweep_payload["best_metrics"])),
            "candidate_metrics": copy.deepcopy(list(sweep_payload["candidate_metrics"])),
            "prauc": float(sweep_payload["prauc"]),
        }
        return {
            "prediction_mode": "calibrated_threshold",
            "prediction_control": {
                "mode": "calibrated_threshold",
                "threshold": threshold,
                "threshold_source": "evaluation.calibration.best_threshold",
                "threshold_selection": threshold_selection,
            },
            "binary_predictions": binarize_predictions(eval_probs, threshold),
            "threshold": threshold,
            "threshold_source": "evaluation.calibration.best_threshold",
            "threshold_selection": threshold_selection,
        }

    threshold, threshold_source, threshold_selection = resolve_effective_threshold(
        cli_threshold=cli_threshold,
        checkpoint_payload=checkpoint_payload,
        config_threshold=float(normalized["threshold"]),
    )
    return {
        "prediction_mode": "threshold",
        "prediction_control": {
            "mode": "threshold",
            "threshold": float(threshold),
            "threshold_source": str(threshold_source),
            "threshold_selection": copy.deepcopy(threshold_selection),
        },
        "binary_predictions": binarize_predictions(eval_probs, threshold),
        "threshold": float(threshold),
        "threshold_source": str(threshold_source),
        "threshold_selection": threshold_selection,
    }


__all__ = [
    "binarize_top_k_predictions",
    "collect_model_predictions",
    "compute_average_target_cardinality_from_dataloader",
    "normalize_prediction_config",
    "prediction_mode_requires_calibration",
    "prediction_mode_requires_train_cardinality",
    "resolve_prediction_control",
    "resolve_requested_prediction_mode",
]
