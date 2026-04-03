from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.utils.io import ensure_dir


_LOSS_KEYS = ("total_loss", "prediction_loss", "ddi_loss", "weighted_ddi_loss")


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor for logging, got shape {tuple(value.shape)}")
        return float(value.detach().cpu().item())
    return float(value)


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


class Trainer:
    """Minimal trainer for stable core-model optimization."""

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        checkpoint_dir: str | Path,
        log_dir: str | Path,
        scheduler: Any | None = None,
        monitor_metric: str = "val_total_loss",
        monitor_mode: str = "min",
        decoder_top_k: int | None = None,
    ) -> None:
        if monitor_mode not in {"min", "max"}:
            raise ValueError(f"monitor_mode must be 'min' or 'max', got {monitor_mode!r}")

        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device) if isinstance(loss_fn, nn.Module) else loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.monitor_metric = str(monitor_metric)
        self.monitor_mode = monitor_mode
        self.decoder_top_k = decoder_top_k

        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.log_dir = ensure_dir(log_dir)
        self.best_checkpoint_path = self.checkpoint_dir / "train_core_best.pt"
        self.metrics_log_path = self.log_dir / "train_core_metrics.jsonl"
        self.best_metric = float("inf") if monitor_mode == "min" else float("-inf")

    def _run_one_epoch(
        self,
        dataloader: DataLoader,
        *,
        training: bool,
    ) -> dict[str, float]:
        phase = "train" if training else "val"
        totals = {key: 0.0 for key in _LOSS_KEYS}
        total_examples = 0

        self.model.train(mode=training)
        grad_context = torch.enable_grad if training else torch.no_grad

        for batch in dataloader:
            batch_on_device = _move_batch_to_device(batch, self.device)
            batch_size = int(batch_on_device["visit_mask"].shape[0])
            if batch_size <= 0:
                continue

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with grad_context():
                outputs = self.model(
                    batch_on_device,
                    mode="core",
                    decoder_top_k=self.decoder_top_k,
                )
                drug_logits = outputs.get("drug_logits")
                drug_probs = outputs.get("drug_probs")
                if drug_logits is None or drug_probs is None:
                    raise RuntimeError(
                        "Model did not return `drug_logits` and `drug_probs`. "
                        "Ensure a medication decoder is attached in core training."
                    )

                loss_outputs = self.loss_fn(
                    drug_logits=drug_logits,
                    drug_probs=drug_probs,
                    target_drugs=batch_on_device["target_drugs"],
                    visit_mask=batch_on_device["visit_mask"],
                )

                if training:
                    loss_outputs["total_loss"].backward()
                    self.optimizer.step()

            total_examples += batch_size
            for key in _LOSS_KEYS:
                totals[key] += _to_float(loss_outputs[key]) * batch_size

        if total_examples <= 0:
            raise ValueError(f"{phase} dataloader produced zero valid examples")

        return {
            f"{phase}_{key}": totals[key] / float(total_examples)
            for key in _LOSS_KEYS
        }

    def train_one_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        return self._run_one_epoch(dataloader, training=True)

    def validate_one_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        return self._run_one_epoch(dataloader, training=False)

    def save_best_checkpoint(
        self,
        *,
        epoch: int,
        epoch_metrics: Mapping[str, float],
        extra_state: Mapping[str, Any] | None = None,
    ) -> Path | None:
        if self.monitor_metric not in epoch_metrics:
            raise KeyError(f"Missing monitor metric `{self.monitor_metric}` in epoch metrics")

        current_metric = float(epoch_metrics[self.monitor_metric])
        is_better = (
            current_metric < self.best_metric
            if self.monitor_mode == "min"
            else current_metric > self.best_metric
        )
        if not is_better:
            return None

        self.best_metric = current_metric
        checkpoint_payload: dict[str, Any] = {
            "epoch": int(epoch),
            "best_metric": current_metric,
            "monitor_metric": self.monitor_metric,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint_payload["scheduler_state_dict"] = self.scheduler.state_dict()
        if extra_state:
            checkpoint_payload.update(dict(extra_state))

        torch.save(checkpoint_payload, self.best_checkpoint_path)
        return self.best_checkpoint_path

    def log_metrics(self, *, epoch: int, metrics: Mapping[str, Any]) -> None:
        summary = (
            f"Epoch {epoch}: "
            f"train_total_loss={float(metrics['train_total_loss']):.6f} "
            f"train_prediction_loss={float(metrics['train_prediction_loss']):.6f} "
            f"train_ddi_loss={float(metrics['train_ddi_loss']):.6f} "
            f"val_total_loss={float(metrics['val_total_loss']):.6f} "
            f"val_prediction_loss={float(metrics['val_prediction_loss']):.6f} "
            f"val_ddi_loss={float(metrics['val_ddi_loss']):.6f}"
        )
        print(summary)

        log_payload = {"epoch": int(epoch), **{key: _to_float(value) for key, value in metrics.items()}}
        with self.metrics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_payload, ensure_ascii=True, sort_keys=True))
            handle.write("\n")

    def fit(
        self,
        *,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int,
        extra_checkpoint_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if int(epochs) <= 0:
            raise ValueError(f"epochs must be positive, got {epochs!r}")

        history: list[dict[str, float]] = []
        best_checkpoint_path: Path | None = None

        for epoch in range(1, int(epochs) + 1):
            train_metrics = self.train_one_epoch(train_dataloader)
            val_metrics = self.validate_one_epoch(val_dataloader)
            epoch_metrics = {**train_metrics, **val_metrics}

            if self.scheduler is not None:
                self.scheduler.step()

            maybe_best = self.save_best_checkpoint(
                epoch=epoch,
                epoch_metrics=epoch_metrics,
                extra_state=extra_checkpoint_state,
            )
            if maybe_best is not None:
                best_checkpoint_path = maybe_best

            self.log_metrics(epoch=epoch, metrics=epoch_metrics)
            history.append({"epoch": float(epoch), **epoch_metrics})

        return {
            "history": history,
            "best_metric": self.best_metric,
            "best_checkpoint_path": None if best_checkpoint_path is None else str(best_checkpoint_path),
            "monitor_metric": self.monitor_metric,
        }


__all__ = ["Trainer"]
