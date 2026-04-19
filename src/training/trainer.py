from __future__ import annotations

import contextlib
import copy
import itertools
import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.utils.io import ensure_dir


_LOSS_KEYS = ("total_loss", "prediction_loss", "ddi_loss", "weighted_ddi_loss")
_TIMING_KEYS = (
    "data_time",
    "transfer_time",
    "forward_time",
    "loss_time",
    "backward_time",
    "optimizer_time",
    "step_time",
    "samples_per_sec",
)
_KEEP_CPU_BATCH_KEYS = frozenset({"visit_lengths"})


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor for logging, got shape {tuple(value.shape)}")
        return float(value.detach().cpu().item())
    return float(value)


def _move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    non_blocking: bool = False,
    keep_cpu_keys: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    resolved_keep_cpu_keys = _KEEP_CPU_BATCH_KEYS if keep_cpu_keys is None else keep_cpu_keys
    return {
        key: (
            value
            if key in resolved_keep_cpu_keys
            else value.to(device, non_blocking=non_blocking)
        )
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _resolve_target_tensor(batch: Mapping[str, Any]) -> torch.Tensor:
    target = batch.get("final_target_drugs")
    if isinstance(target, torch.Tensor):
        return target
    target = batch.get("target_drugs")
    if isinstance(target, torch.Tensor):
        return target
    raise KeyError("Batch must contain either `final_target_drugs` or `target_drugs`.")


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
        run_context: Mapping[str, Any] | None = None,
        amp: bool = False,
        grad_accum_steps: int = 1,
        non_blocking_transfer: bool = False,
        log_interval: int = 50,
        profile_steps: int | None = None,
        timing_enabled: bool = True,
    ) -> None:
        if monitor_mode not in {"min", "max"}:
            raise ValueError(f"monitor_mode must be 'min' or 'max', got {monitor_mode!r}")
        if int(grad_accum_steps) <= 0:
            raise ValueError(f"grad_accum_steps must be positive, got {grad_accum_steps!r}")
        if int(log_interval) <= 0:
            raise ValueError(f"log_interval must be positive, got {log_interval!r}")
        if profile_steps is not None and int(profile_steps) <= 0:
            raise ValueError(f"profile_steps must be positive when provided, got {profile_steps!r}")

        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device) if isinstance(loss_fn, nn.Module) else loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.monitor_metric = str(monitor_metric)
        self.monitor_mode = monitor_mode
        self.decoder_top_k = decoder_top_k
        self.run_context = copy.deepcopy(dict(run_context or {}))

        self.use_amp = bool(amp) and device.type == "cuda"
        self.grad_accum_steps = int(grad_accum_steps)
        self.non_blocking_transfer = bool(non_blocking_transfer)
        self.log_interval = int(log_interval)
        self.profile_steps = None if profile_steps is None else int(profile_steps)
        self.timing_enabled = bool(timing_enabled)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:  # pragma: no cover - compatibility path for older torch
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.log_dir = ensure_dir(log_dir)
        self.best_checkpoint_path = self.checkpoint_dir / "train_core_best.pt"
        self.metrics_log_path = self.log_dir / "train_core_metrics.jsonl"
        self.best_metric = float("inf") if monitor_mode == "min" else float("-inf")

    def _sync_timing(self) -> None:
        if self.device.type == "cuda" and self.timing_enabled:
            torch.cuda.synchronize(self.device)

    def _autocast_context(self):
        if not self.use_amp:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.float16)

    def _max_epoch_steps(self, dataloader: DataLoader) -> int | None:
        if self.profile_steps is None:
            return None
        try:
            return min(int(len(dataloader)), int(self.profile_steps))
        except TypeError:
            return int(self.profile_steps)

    def _create_progress(
        self,
        dataloader: DataLoader,
        *,
        phase: str,
        training: bool,
        max_steps: int | None,
    ) -> Any | None:
        _ = dataloader
        _ = phase
        _ = training
        _ = max_steps
        return None

    def _close_progress(self, progress: Any | None) -> None:
        _ = progress

    def _update_progress(
        self,
        progress: Any | None,
        *,
        phase: str,
        step_index: int,
        total_examples: int,
        totals: Mapping[str, float],
        timing_totals: Mapping[str, float],
    ) -> None:
        _ = progress
        _ = phase
        _ = step_index
        _ = total_examples
        _ = totals
        _ = timing_totals

    def _forward_model(self, batch_on_device: Mapping[str, Any]) -> dict[str, Any]:
        return self.model(
            batch_on_device,
            mode="core",
            decoder_top_k=self.decoder_top_k,
            compute_ddi_metrics=False,
        )

    def _compute_loss_outputs(
        self,
        *,
        outputs: Mapping[str, Any],
        batch_on_device: Mapping[str, Any],
    ) -> dict[str, Any]:
        drug_logits = outputs.get("drug_logits")
        drug_probs = outputs.get("drug_probs")
        if drug_logits is None or drug_probs is None:
            raise RuntimeError(
                "Model did not return `drug_logits` and `drug_probs`. "
                "Ensure a medication decoder is attached in core training."
            )
        return self.loss_fn(
            drug_logits=drug_logits,
            drug_probs=drug_probs,
            target_drugs=_resolve_target_tensor(batch_on_device),
            visit_mask=batch_on_device["visit_mask"],
        )

    def _optimizer_step(self) -> None:
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _timing_metric_payload(
        self,
        *,
        phase: str,
        timing_totals: Mapping[str, float],
        total_examples: int,
        step_count: int,
    ) -> dict[str, float]:
        if not self.timing_enabled or step_count <= 0:
            return {}
        total_loop_time = float(timing_totals["data_time"]) + float(timing_totals["step_time"])
        samples_per_sec = 0.0 if total_loop_time <= 0.0 else float(total_examples) / total_loop_time
        return {
            f"{phase}_{key}": float(timing_totals[key]) / float(step_count)
            for key in _TIMING_KEYS
            if key != "samples_per_sec"
        } | {f"{phase}_samples_per_sec": samples_per_sec}

    def _run_one_epoch(
        self,
        dataloader: DataLoader,
        *,
        training: bool,
    ) -> dict[str, float]:
        phase = "train" if training else "val"
        totals = {key: 0.0 for key in _LOSS_KEYS}
        timing_totals = {
            "data_time": 0.0,
            "transfer_time": 0.0,
            "forward_time": 0.0,
            "loss_time": 0.0,
            "backward_time": 0.0,
            "optimizer_time": 0.0,
            "step_time": 0.0,
        }
        total_examples = 0
        step_count = 0
        max_steps = self._max_epoch_steps(dataloader)

        self.model.train(mode=training)
        grad_context = torch.enable_grad if training else torch.no_grad
        progress = self._create_progress(
            dataloader,
            phase=phase,
            training=training,
            max_steps=max_steps,
        )
        iterable_source = dataloader if progress is None else progress
        iterable = (
            iterable_source
            if max_steps is None
            else itertools.islice(iterable_source, max_steps)
        )
        batches_since_step = 0
        self.optimizer.zero_grad(set_to_none=True)
        last_step_end = time.perf_counter()

        try:
            for step_index, batch in enumerate(iterable, start=1):
                data_time = time.perf_counter() - last_step_end

                transfer_start = time.perf_counter()
                batch_on_device = _move_batch_to_device(
                    batch,
                    self.device,
                    non_blocking=self.non_blocking_transfer,
                )
                self._sync_timing()
                transfer_time = time.perf_counter() - transfer_start

                batch_size = int(batch_on_device["visit_mask"].shape[0])
                if batch_size <= 0:
                    last_step_end = time.perf_counter()
                    continue

                with grad_context():
                    forward_start = time.perf_counter()
                    with self._autocast_context():
                        outputs = self._forward_model(batch_on_device)
                    self._sync_timing()
                    forward_time = time.perf_counter() - forward_start

                    loss_start = time.perf_counter()
                    with self._autocast_context():
                        loss_outputs = self._compute_loss_outputs(
                            outputs=outputs,
                            batch_on_device=batch_on_device,
                        )
                    self._sync_timing()
                    loss_time = time.perf_counter() - loss_start

                    backward_time = 0.0
                    optimizer_time = 0.0
                    if training:
                        backward_start = time.perf_counter()
                        scaled_loss = loss_outputs["total_loss"] / float(self.grad_accum_steps)
                        if self.use_amp:
                            self.scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        self._sync_timing()
                        backward_time = time.perf_counter() - backward_start
                        batches_since_step += 1

                        if batches_since_step >= self.grad_accum_steps:
                            optimizer_start = time.perf_counter()
                            self._optimizer_step()
                            self._sync_timing()
                            optimizer_time = time.perf_counter() - optimizer_start
                            batches_since_step = 0

                step_compute_time = transfer_time + forward_time + loss_time + backward_time + optimizer_time
                total_examples += batch_size
                step_count += 1
                for key in _LOSS_KEYS:
                    totals[key] += _to_float(loss_outputs[key]) * batch_size
                timing_totals["data_time"] += data_time
                timing_totals["transfer_time"] += transfer_time
                timing_totals["forward_time"] += forward_time
                timing_totals["loss_time"] += loss_time
                timing_totals["backward_time"] += backward_time
                timing_totals["optimizer_time"] += optimizer_time
                timing_totals["step_time"] += step_compute_time

                if step_index == 1 or step_index % self.log_interval == 0:
                    self._update_progress(
                        progress,
                        phase=phase,
                        step_index=step_index,
                        total_examples=total_examples,
                        totals=totals,
                        timing_totals=timing_totals,
                    )
                last_step_end = time.perf_counter()
        finally:
            self._close_progress(progress)

        if training and batches_since_step > 0:
            optimizer_start = time.perf_counter()
            self._optimizer_step()
            self._sync_timing()
            optimizer_flush_time = time.perf_counter() - optimizer_start
            timing_totals["optimizer_time"] += optimizer_flush_time
            timing_totals["step_time"] += optimizer_flush_time

        if total_examples <= 0:
            raise ValueError(f"{phase} dataloader produced zero valid examples")

        epoch_metrics = {
            f"{phase}_{key}": totals[key] / float(total_examples)
            for key in _LOSS_KEYS
        }
        epoch_metrics.update(
            self._timing_metric_payload(
                phase=phase,
                timing_totals=timing_totals,
                total_examples=total_examples,
                step_count=step_count,
            )
        )
        return epoch_metrics

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
        ddi_context = dict(self.run_context.get("ddi_context", {}))
        ddi_status = ddi_context.get("status", "active" if ddi_context.get("active") else "inactive")
        ddi_reason = ddi_context.get("reason", "")
        source_metadata = dict(ddi_context.get("source_metadata") or {})
        ddi_kind = source_metadata.get("kind", "")
        ddi_research_grade = source_metadata.get("research_grade")
        ddi_purpose = source_metadata.get("purpose", "")
        effective_ddi_lambda = self.run_context.get("effective_ddi_lambda", 0.0)
        summary = (
            f"Epoch {epoch}: "
            f"train_total_loss={float(metrics['train_total_loss']):.6f} "
            f"train_prediction_loss={float(metrics['train_prediction_loss']):.6f} "
            f"train_ddi_loss={float(metrics['train_ddi_loss']):.6f} "
            f"val_total_loss={float(metrics['val_total_loss']):.6f} "
            f"val_prediction_loss={float(metrics['val_prediction_loss']):.6f} "
            f"val_ddi_loss={float(metrics['val_ddi_loss']):.6f} "
            f"ddi_status={ddi_status} "
            f"effective_ddi_lambda={float(effective_ddi_lambda):.6f}"
        )
        if "train_step_time" in metrics and "train_samples_per_sec" in metrics:
            summary = (
                f"{summary} "
                f"train_step_time={float(metrics['train_step_time']):.4f} "
                f"train_sps={float(metrics['train_samples_per_sec']):.2f}"
            )
        if "val_step_time" in metrics and "val_samples_per_sec" in metrics:
            summary = (
                f"{summary} "
                f"val_step_time={float(metrics['val_step_time']):.4f} "
                f"val_sps={float(metrics['val_samples_per_sec']):.2f}"
            )
        if ddi_reason:
            summary = f"{summary} ddi_reason={ddi_reason}"
        if ddi_kind:
            summary = f"{summary} ddi_kind={ddi_kind}"
        if ddi_research_grade is not None:
            summary = f"{summary} ddi_research_grade={bool(ddi_research_grade)}"
        if ddi_purpose:
            summary = f"{summary} ddi_purpose={str(ddi_purpose)}"
        print(summary)

        log_payload = {"epoch": int(epoch), **{key: _to_float(value) for key, value in metrics.items()}}
        if self.run_context:
            log_payload["run_context"] = copy.deepcopy(self.run_context)
        with self.metrics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_payload, ensure_ascii=True, sort_keys=True))
            handle.write("\n")

    def _set_dataloader_epoch(self, dataloader: DataLoader, *, epoch: int) -> None:
        batch_sampler = getattr(dataloader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)

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
            self._set_dataloader_epoch(train_dataloader, epoch=epoch)
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


__all__ = ["Trainer", "_LOSS_KEYS", "_move_batch_to_device", "_resolve_target_tensor", "_to_float"]
