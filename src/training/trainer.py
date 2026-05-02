from __future__ import annotations

import csv
import json
import time
from collections.abc import Mapping as MappingABC, Sequence
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_core_metrics
from src.training.losses import extract_last_valid_targets

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.utils.io import ensure_dir


_LOSS_KEYS = (
    "total_loss",
    "prediction_loss",
    "pred_bce_loss",
    "margin_loss",
    "weighted_margin_loss",
    "ddi_loss",
    "weighted_ddi_loss",
    "lambda_ddi_current",
)
_TIME_KEYS = ("data_time", "step_time")
_GRAD_KEYS = ("grad_norm", "clipped_grad_norm")
_CODE_EMBEDDING_SPECS = (
    {
        "label": "diagnosis",
        "parameter_name": "encoder.diagnosis_encoder.embedding.weight",
        "batch_key": "diag_codes",
        "mask_key": "diag_mask",
    },
    {
        "label": "procedure",
        "parameter_name": "encoder.procedure_encoder.embedding.weight",
        "batch_key": "proc_codes",
        "mask_key": "proc_mask",
    },
    {
        "label": "medication_history",
        "parameter_name": "encoder.medication_history_encoder.embedding.weight",
        "batch_key": "med_history",
        "mask_key": "med_history_mask",
    },
)


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


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _log_line(message: str) -> None:
    writer = getattr(tqdm, "write", None)
    if callable(writer):
        writer(message)
        return
    print(message)


def _resolve_loss_output(
    outputs: Mapping[str, Any],
    *,
    key: str,
    fallback_key: str | None = None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = outputs.get(key)
    if value is None and fallback_key is not None:
        value = outputs.get(fallback_key)
    if value is None:
        return torch.zeros((), device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


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


def assert_finite(name: str, value: torch.Tensor) -> None:
    if value.is_floating_point() or value.is_complex():
        if not torch.isfinite(value).all():
            raise ValueError(
                f"Non-finite tensor detected at `{name}`; {_tensor_debug_summary(name, value)}"
            )


def _assert_finite_tree(prefix: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        assert_finite(prefix, value)
        return
    if isinstance(value, MappingABC):
        for key, child in value.items():
            _assert_finite_tree(f"{prefix}.{key}", child)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_finite_tree(f"{prefix}[{index}]", child)


def _get_named_parameter(model: nn.Module, parameter_name: str) -> nn.Parameter | None:
    for name, parameter in model.named_parameters():
        if name == parameter_name:
            return parameter
    return None


def _assert_finite_gradients(model: nn.Module) -> None:
    first_nonfinite_name: str | None = None
    first_nonfinite_grad: torch.Tensor | None = None
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        if grad.is_floating_point() or grad.is_complex():
            if not torch.isfinite(grad).all():
                first_nonfinite_name = f"gradients.{name}"
                first_nonfinite_grad = grad
                break
    if first_nonfinite_name is not None:
        finite_global_norm = _compute_global_grad_norm(model, finite_only=True)
        extra_context = ""
        if finite_global_norm is not None:
            extra_context = f" finite_global_grad_norm={finite_global_norm:.6f}"
        raise ValueError(
            f"Non-finite gradients detected; first offending parameter=`{first_nonfinite_name}`."
            f"{extra_context} {_tensor_debug_summary(first_nonfinite_name, first_nonfinite_grad)}"
        )


def _compute_global_grad_norm(model: nn.Module, *, finite_only: bool = False) -> float | None:
    total_sq_norm = 0.0
    saw_gradient = False
    for _, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        if not (grad.is_floating_point() or grad.is_complex()):
            continue
        resolved_grad = grad.detach()
        if finite_only:
            finite_grad = resolved_grad[torch.isfinite(resolved_grad)]
            if finite_grad.numel() == 0:
                continue
            grad_norm = torch.linalg.vector_norm(finite_grad)
        else:
            if not torch.isfinite(resolved_grad).all():
                return None
            grad_norm = torch.linalg.vector_norm(resolved_grad)
        total_sq_norm += float(grad_norm.item()) ** 2
        saw_gradient = True
    if not saw_gradient:
        return 0.0
    return total_sq_norm ** 0.5


def _count_valid_code_tokens(
    batch: Mapping[str, Any] | None,
    *,
    batch_key: str,
    mask_key: str,
) -> int:
    if batch is None:
        return 0
    code_tensor = batch.get(batch_key)
    if not isinstance(code_tensor, torch.Tensor):
        return 0
    code_mask = batch.get(mask_key)
    if isinstance(code_mask, torch.Tensor) and tuple(code_mask.shape) == tuple(code_tensor.shape):
        resolved_mask = code_mask.to(device=code_tensor.device, dtype=torch.bool)
    elif code_tensor.dtype.is_floating_point:
        resolved_mask = code_tensor > 0
    else:
        resolved_mask = code_tensor.ne(0)
    return int(resolved_mask.sum().item())


def _resolve_debug_check_now(
    *,
    enabled: bool,
    light_mode: bool,
    check_every_n_steps: int,
    step_index: int,
) -> bool:
    if not enabled:
        return False
    if not light_mode:
        return True
    resolved_every = max(int(check_every_n_steps), 1)
    return int(step_index) == 1 or int(step_index) % resolved_every == 0


def _sanitize_code_embedding_gradients(
    model: nn.Module,
    *,
    batch: Mapping[str, Any] | None,
    max_norm: float | None,
    phase: str,
    epoch: int,
    step: int,
) -> dict[str, dict[str, float | int | bool]]:
    reports: dict[str, dict[str, float | int | bool]] = {}
    for spec in _CODE_EMBEDDING_SPECS:
        label = str(spec["label"])
        parameter_name = str(spec["parameter_name"])
        valid_code_tokens = _count_valid_code_tokens(
            batch,
            batch_key=str(spec["batch_key"]),
            mask_key=str(spec["mask_key"]),
        )
        parameter = _get_named_parameter(model, parameter_name)
        if parameter is None or parameter.grad is None:
            reports[label] = {
                "had_nonfinite": False,
                "pre_clip_norm": 0.0,
                "post_clip_norm": 0.0,
                "valid_code_tokens": valid_code_tokens,
            }
            continue
        grad = parameter.grad
        if not (grad.is_floating_point() or grad.is_complex()):
            reports[label] = {
                "had_nonfinite": False,
                "pre_clip_norm": 0.0,
                "post_clip_norm": 0.0,
                "valid_code_tokens": valid_code_tokens,
            }
            continue

        raw_grad = grad.detach().clone()
        had_nonfinite = not torch.isfinite(raw_grad).all()
        if had_nonfinite:
            sanitized_grad = torch.nan_to_num(raw_grad, nan=0.0, posinf=0.0, neginf=0.0)
            parameter.grad.copy_(sanitized_grad)
        working_grad = parameter.grad.detach()
        pre_clip_norm = (
            float(torch.linalg.vector_norm(working_grad).item()) if working_grad.numel() > 0 else 0.0
        )
        if max_norm is not None and float(max_norm) > 0.0 and pre_clip_norm > float(max_norm):
            clip_scale = float(max_norm) / max(pre_clip_norm, 1.0e-12)
            parameter.grad.mul_(clip_scale)
            working_grad = parameter.grad.detach()
        post_clip_norm = (
            float(torch.linalg.vector_norm(working_grad).item()) if working_grad.numel() > 0 else 0.0
        )
        if had_nonfinite:
            _log_line(
                "[code-embedding-grad-sanitize] "
                f"branch={label} phase={phase} epoch={epoch} step={step} "
                f"valid_code_tokens={valid_code_tokens} "
                f"raw={_tensor_debug_summary(parameter_name, raw_grad)} "
                f"post_clip_norm={post_clip_norm:.6f}"
            )
        reports[label] = {
            "had_nonfinite": had_nonfinite,
            "pre_clip_norm": pre_clip_norm,
            "post_clip_norm": post_clip_norm,
            "valid_code_tokens": valid_code_tokens,
        }
    return reports


class Trainer:
    """Trainer for the self-history-only core pipeline."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        checkpoint_dir: str | Path,
        log_dir: str | Path,
        scheduler: Any | None = None,
        monitor_metric: str = "val_total_loss",
        monitor_mode: str = "min",
        decoder_top_k: int | None = None,
        loss_fn: nn.Module | None = None,
        validation_threshold: float = 0.5,
        use_self_history: bool = True,
        use_ddi: bool = True,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
        max_grad_norm: float | None = 1.0,
        sanitize_code_embedding_grads: bool = True,
        code_embedding_grad_max_norm: float | None = 0.5,
        freeze_code_embedding_epochs: int = 1,
        debug_checks_enabled: bool = True,
        debug_checks_light_mode: bool = True,
        debug_check_every_n_steps: int = 100,
        sync_timing: bool = False,
    ) -> None:
        if monitor_mode not in {"min", "max"}:
            raise ValueError(f"monitor_mode must be 'min' or 'max', got {monitor_mode!r}")
        if not 0.0 <= float(validation_threshold) <= 1.0:
            raise ValueError(f"validation_threshold must be in [0, 1], got {validation_threshold!r}")

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.monitor_metric = str(monitor_metric)
        self.monitor_mode = str(monitor_mode)
        self.decoder_top_k = decoder_top_k
        self.loss_fn = loss_fn
        self.validation_threshold = float(validation_threshold)
        self.use_self_history = bool(use_self_history)
        self.use_ddi = bool(use_ddi)
        self.use_retrieval = bool(getattr(self.model, "use_retrieval", False))
        self.history_mode = str(getattr(self.model, "history_mode", "self_only"))
        self.max_train_batches = None if max_train_batches is None else int(max_train_batches)
        self.max_val_batches = None if max_val_batches is None else int(max_val_batches)
        self.max_grad_norm = (
            None if max_grad_norm is None or float(max_grad_norm) <= 0.0 else float(max_grad_norm)
        )
        self.sanitize_code_embedding_grads = bool(sanitize_code_embedding_grads)
        self.code_embedding_grad_max_norm = (
            None
            if code_embedding_grad_max_norm is None or float(code_embedding_grad_max_norm) <= 0.0
            else float(code_embedding_grad_max_norm)
        )
        self.freeze_code_embedding_epochs = max(int(freeze_code_embedding_epochs), 0)
        self.debug_checks_enabled = bool(debug_checks_enabled)
        self.debug_checks_light_mode = bool(debug_checks_light_mode)
        self.debug_check_every_n_steps = max(int(debug_check_every_n_steps), 1)
        self.sync_timing = bool(sync_timing)
        self._retrieval_policy_logged = False
        self._code_embeddings_are_frozen: bool | None = None

        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.log_dir = ensure_dir(log_dir)
        self.best_checkpoint_path = self.checkpoint_dir / "train_core_best.pt"
        self.metrics_log_path = self.log_dir / "train_core_metrics.jsonl"
        self.metrics_per_epoch_json_path = self.log_dir / "metrics_per_epoch.json"
        self.metrics_per_epoch_csv_path = self.log_dir / "metrics_per_epoch.csv"
        self.best_metrics_path = self.log_dir / "best_metrics.json"
        self.best_metric = float("inf") if monitor_mode == "min" else float("-inf")
        self.best_epoch_metrics: dict[str, float] | None = None

    def _resolve_validation_ddi_matrix(self, batch: Mapping[str, Any]) -> torch.Tensor | None:
        ddi_matrix = batch.get("ddi_adj")
        if ddi_matrix is None:
            ddi_matrix = getattr(self.model, "ddi_matrix", None)
        if ddi_matrix is None:
            return None

        resolved = torch.as_tensor(ddi_matrix, dtype=torch.float32).detach().cpu()
        if resolved.ndim != 2:
            raise ValueError(f"Validation ddi_matrix must have shape (D, D), got {tuple(resolved.shape)}")
        return resolved

    def _resolve_validation_targets(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
    ) -> torch.Tensor | None:
        target_current = outputs.get("final_target_drugs")
        if target_current is None:
            target_current = outputs.get("target_current")
        if target_current is None:
            target_current = batch.get("target_drugs")
            if isinstance(target_current, torch.Tensor) and target_current.ndim == 3:
                visit_mask = batch.get("visit_mask")
                if not isinstance(visit_mask, torch.Tensor):
                    raise RuntimeError("visit_mask is required to resolve validation targets from [B, T, D].")
                target_current = extract_last_valid_targets(target_current, visit_mask)
        return None if target_current is None else torch.as_tensor(target_current)

    def _set_code_embeddings_trainable(self, *, trainable: bool, epoch: int) -> None:
        if self._code_embeddings_are_frozen is not None and self._code_embeddings_are_frozen == (not trainable):
            return
        affected_branches: list[str] = []
        for spec in _CODE_EMBEDDING_SPECS:
            parameter = _get_named_parameter(self.model, str(spec["parameter_name"]))
            if parameter is None:
                continue
            parameter.requires_grad_(trainable)
            if not trainable:
                parameter.grad = None
            affected_branches.append(str(spec["label"]))
        if not affected_branches:
            return
        self._code_embeddings_are_frozen = not trainable
        state = "unfrozen" if trainable else "frozen"
        _log_line(
            f"Code embeddings {state} for epoch {epoch}: {', '.join(affected_branches)} "
            f"(freeze_epochs={self.freeze_code_embedding_epochs})"
        )

    def _run_one_epoch(
        self,
        dataloader: DataLoader,
        *,
        training: bool,
        epoch: int,
    ) -> dict[str, float]:
        phase = "train" if training else "val"
        totals = {key: 0.0 for key in _LOSS_KEYS}
        timing_totals = {key: 0.0 for key in _TIME_KEYS}
        total_examples = 0
        total_batches = 0
        collected_probs: list[torch.Tensor] = []
        collected_targets: list[torch.Tensor] = []
        validation_ddi_matrix: torch.Tensor | None = None
        grad_totals = {key: 0.0 for key in _GRAD_KEYS}
        grad_batches = 0
        code_embedding_grad_norm_totals = {
            str(spec["label"]): 0.0
            for spec in _CODE_EMBEDDING_SPECS
        }
        code_embedding_sanitized_events = {
            str(spec["label"]): 0.0
            for spec in _CODE_EMBEDDING_SPECS
        }

        self.model.train(mode=training)
        grad_context = torch.enable_grad if training else torch.no_grad
        progress = tqdm(
            range(len(dataloader)),
            desc=f"{phase} batches",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            mininterval=2.0,
        )

        dataloader_iter = iter(dataloader)
        for step_index in progress:
            max_batches = self.max_train_batches if training else self.max_val_batches
            if max_batches is not None and total_batches >= max_batches:
                break
            data_start = time.perf_counter()
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break
            data_time = time.perf_counter() - data_start

            step_start = time.perf_counter()
            batch_on_device = _move_batch_to_device(batch, self.device)
            batch_on_device["_current_epoch"] = int(epoch)
            step_number = total_batches + 1
            debug_check_now = _resolve_debug_check_now(
                enabled=self.debug_checks_enabled,
                light_mode=self.debug_checks_light_mode,
                check_every_n_steps=self.debug_check_every_n_steps,
                step_index=step_number,
            )
            batch_on_device["_debug_check_now"] = bool(debug_check_now)
            if debug_check_now:
                _assert_finite_tree(f"{phase}.batch", batch_on_device)
            batch_size = int(batch_on_device["visit_mask"].shape[0])
            if batch_size <= 0:
                continue

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with grad_context():
                outputs = self.model(batch_on_device)
                if debug_check_now:
                    _assert_finite_tree(f"{phase}.outputs", outputs)
                output_dtype = torch.float32
                logits = outputs.get("drug_logits")
                if isinstance(logits, torch.Tensor):
                    output_dtype = logits.dtype
                    assert_finite(f"{phase}.drug_logits", logits)
                drug_probs_output = outputs.get("drug_probs")
                if isinstance(drug_probs_output, torch.Tensor):
                    assert_finite(f"{phase}.drug_probs", drug_probs_output)
                total_loss = _resolve_loss_output(
                    outputs,
                    key="total_loss",
                    device=self.device,
                    dtype=output_dtype,
                )
                prediction_loss = _resolve_loss_output(
                    outputs,
                    key="prediction_loss",
                    fallback_key="pred_bce_loss",
                    device=self.device,
                    dtype=output_dtype,
                )
                pred_bce_loss = _resolve_loss_output(
                    outputs,
                    key="pred_bce_loss",
                    fallback_key="prediction_loss",
                    device=self.device,
                    dtype=output_dtype,
                )
                margin_loss = _resolve_loss_output(
                    outputs,
                    key="margin_loss",
                    device=self.device,
                    dtype=output_dtype,
                )
                weighted_margin_loss = _resolve_loss_output(
                    outputs,
                    key="weighted_margin_loss",
                    device=self.device,
                    dtype=output_dtype,
                )
                ddi_loss = _resolve_loss_output(
                    outputs,
                    key="ddi_loss",
                    device=self.device,
                    dtype=output_dtype,
                )
                weighted_ddi_loss = _resolve_loss_output(
                    outputs,
                    key="weighted_ddi_loss",
                    device=self.device,
                    dtype=output_dtype,
                )
                lambda_ddi_current = _resolve_loss_output(
                    outputs,
                    key="lambda_ddi_current",
                    device=self.device,
                    dtype=output_dtype,
                )
                if outputs.get("total_loss") is None:
                    raise RuntimeError(
                        "Model forward must return `total_loss`, `prediction_loss`, and `ddi_loss` "
                        "for the new training pipeline."
                    )
                if not training:
                    drug_probs = outputs.get("drug_probs")
                    target_current = self._resolve_validation_targets(outputs, batch_on_device)
                    if drug_probs is None or target_current is None:
                        raise RuntimeError(
                            "Model forward must return `drug_probs` and current-visit targets for validation metrics."
                        )
                    collected_probs.append(drug_probs.detach().cpu())
                    collected_targets.append(target_current.detach().cpu())
                    batch_ddi_matrix = self._resolve_validation_ddi_matrix(batch_on_device)
                    if batch_ddi_matrix is None:
                        raise RuntimeError("Validation metrics require an available ddi_matrix.")
                    if validation_ddi_matrix is None:
                        validation_ddi_matrix = batch_ddi_matrix
                    elif not torch.equal(validation_ddi_matrix, batch_ddi_matrix):
                        raise ValueError("Validation batches produced inconsistent ddi_matrix values.")

                if training:
                    total_loss.backward()
                    code_embedding_grad_reports = {
                        str(spec["label"]): {
                            "had_nonfinite": False,
                            "pre_clip_norm": 0.0,
                            "post_clip_norm": 0.0,
                            "valid_code_tokens": _count_valid_code_tokens(
                                batch_on_device,
                                batch_key=str(spec["batch_key"]),
                                mask_key=str(spec["mask_key"]),
                            ),
                        }
                        for spec in _CODE_EMBEDDING_SPECS
                    }
                    if self.sanitize_code_embedding_grads:
                        code_embedding_grad_reports = _sanitize_code_embedding_gradients(
                            self.model,
                            batch=batch_on_device,
                            max_norm=self.code_embedding_grad_max_norm,
                            phase=phase,
                            epoch=epoch,
                            step=step_number,
                        )
                    if debug_check_now:
                        _assert_finite_gradients(self.model)
                    grad_norm = 0.0
                    clipped_grad_norm = 0.0
                    if self.max_grad_norm is not None:
                        try:
                            grad_norm_value = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.max_grad_norm,
                                error_if_nonfinite=True,
                            )
                        except RuntimeError as exc:
                            _assert_finite_gradients(self.model)
                            raise RuntimeError(
                                "clip_grad_norm_ detected non-finite gradients after sanitize."
                            ) from exc
                        grad_norm = _to_float(grad_norm_value)
                        clipped_grad_norm = min(grad_norm, self.max_grad_norm)
                    elif debug_check_now:
                        grad_norm_value = _compute_global_grad_norm(self.model, finite_only=False)
                        if grad_norm_value is None:
                            _assert_finite_gradients(self.model)
                            raise RuntimeError("Expected finite gradients before optimizer.step, but grad norm was undefined.")
                        grad_norm = float(grad_norm_value)
                        clipped_grad_norm = grad_norm
                    grad_totals["grad_norm"] += float(grad_norm) * batch_size
                    grad_totals["clipped_grad_norm"] += float(clipped_grad_norm) * batch_size
                    for label, report in code_embedding_grad_reports.items():
                        code_embedding_grad_norm_totals[label] += float(report["post_clip_norm"]) * batch_size
                        code_embedding_sanitized_events[label] += float(bool(report["had_nonfinite"]))
                    grad_batches += 1
                    self.optimizer.step()
            if self.sync_timing:
                _synchronize_device(self.device)
            step_time = time.perf_counter() - step_start

            total_batches += 1
            total_examples += batch_size
            totals["total_loss"] += _to_float(total_loss) * batch_size
            totals["prediction_loss"] += _to_float(prediction_loss) * batch_size
            totals["pred_bce_loss"] += _to_float(pred_bce_loss) * batch_size
            totals["margin_loss"] += _to_float(margin_loss) * batch_size
            totals["weighted_margin_loss"] += _to_float(weighted_margin_loss) * batch_size
            totals["ddi_loss"] += _to_float(ddi_loss) * batch_size
            totals["weighted_ddi_loss"] += _to_float(weighted_ddi_loss) * batch_size
            totals["lambda_ddi_current"] += _to_float(lambda_ddi_current) * batch_size
            timing_totals["data_time"] += data_time
            timing_totals["step_time"] += step_time
            batch_count = max(total_batches, 1)
            if hasattr(progress, "set_postfix") and (total_batches == 1 or total_batches % 20 == 0):
                progress.set_postfix(
                    total_loss=f"{totals['total_loss'] / float(total_examples):.4f}",
                    pred_loss=f"{totals['pred_bce_loss'] / float(total_examples):.4f}",
                    margin=f"{totals['margin_loss'] / float(total_examples):.4f}",
                    ddi_loss=f"{totals['ddi_loss'] / float(total_examples):.4f}",
                    lambda_ddi=f"{totals['lambda_ddi_current'] / float(total_examples):.4f}",
                    grad_norm=(
                        f"{grad_totals['grad_norm'] / float(total_examples):.4f}"
                        if training and grad_batches > 0
                        else "0.0000"
                    ),
                    data_time=f"{timing_totals['data_time'] / float(batch_count):.3f}s",
                    step_time=f"{timing_totals['step_time'] / float(batch_count):.3f}s",
                )

        if total_examples <= 0:
            raise ValueError(f"{phase} dataloader produced zero valid examples")

        metrics = {f"{phase}_{key}": totals[key] / float(total_examples) for key in _LOSS_KEYS}
        average_batches = float(max(total_batches, 1))
        metrics.update({f"{phase}_{key}": timing_totals[key] / average_batches for key in _TIME_KEYS})
        if training and grad_batches > 0:
            metrics.update(
                {
                    "train_grad_norm": grad_totals["grad_norm"] / float(total_examples),
                    "train_clipped_grad_norm": grad_totals["clipped_grad_norm"] / float(total_examples),
                    "train_code_embedding_sanitized_events": sum(code_embedding_sanitized_events.values()),
                }
            )
            for label in code_embedding_grad_norm_totals:
                metrics[f"train_{label}_embedding_grad_norm"] = (
                    code_embedding_grad_norm_totals[label] / float(total_examples)
                )
                metrics[f"train_{label}_embedding_sanitized_events"] = code_embedding_sanitized_events[label]
        elif training:
            metrics.update(
                {
                    "train_grad_norm": 0.0,
                    "train_clipped_grad_norm": 0.0,
                    "train_code_embedding_sanitized_events": 0.0,
                }
            )
            for spec in _CODE_EMBEDDING_SPECS:
                label = str(spec["label"])
                metrics[f"train_{label}_embedding_grad_norm"] = 0.0
                metrics[f"train_{label}_embedding_sanitized_events"] = 0.0
        if not training:
            if not collected_probs or not collected_targets or validation_ddi_matrix is None:
                raise ValueError("Validation epoch did not produce metric inputs")
            validation_metrics = compute_core_metrics(
                y_true=torch.cat(collected_targets, dim=0),
                y_score=torch.cat(collected_probs, dim=0),
                threshold=self.validation_threshold,
                ddi_matrix=validation_ddi_matrix,
            )
            metrics.update(
                {
                    "val_jaccard": float(validation_metrics["jaccard"]),
                    "val_f1": float(validation_metrics["f1"]),
                    "val_prauc": float(validation_metrics["prauc"]),
                    "val_ddi_rate": float(validation_metrics["ddi_rate"]),
                    "val_ddi": float(validation_metrics["ddi_rate"]),
                    "val_avg_drugs": float(validation_metrics["avg_predicted_drugs"]),
                    "val_avg_true_drugs": float(validation_metrics["avg_true_drugs"]),
                }
            )
        return metrics

    def train_one_epoch(self, dataloader: DataLoader, *, epoch: int) -> dict[str, float]:
        return self._run_one_epoch(dataloader, training=True, epoch=epoch)

    def validate_one_epoch(self, dataloader: DataLoader, *, epoch: int) -> dict[str, float]:
        return self._run_one_epoch(dataloader, training=False, epoch=epoch)

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
        self.best_epoch_metrics = {key: _to_float(value) for key, value in epoch_metrics.items()}
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
        write_payload = {
            "epoch": int(epoch),
            "monitor_metric": self.monitor_metric,
            "best_metric": current_metric,
            "use_self_history": self.use_self_history,
            "use_ddi": self.use_ddi,
            "use_retrieval": self.use_retrieval,
            "history_mode": self.history_mode,
            **self.best_epoch_metrics,
        }
        with self.best_metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(write_payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        return self.best_checkpoint_path

    def log_metrics(self, *, epoch: int, metrics: Mapping[str, Any]) -> None:
        bottleneck = (
            "data_loader"
            if float(metrics["train_data_time"]) > float(metrics["train_step_time"])
            else "model_step"
        )
        summary = (
            f"Epoch {epoch}: "
            f"train_total_loss={float(metrics['train_total_loss']):.6f} "
            f"train_pred_bce_loss={float(metrics['train_pred_bce_loss']):.6f} "
            f"train_margin_loss={float(metrics['train_margin_loss']):.6f} "
            f"train_ddi_loss={float(metrics['train_ddi_loss']):.6f} "
            f"train_lambda_ddi_current={float(metrics['train_lambda_ddi_current']):.6f} "
            f"train_grad_norm={float(metrics.get('train_grad_norm', 0.0)):.6f} "
            f"train_clipped_grad_norm={float(metrics.get('train_clipped_grad_norm', 0.0)):.6f} "
            f"train_diagnosis_embedding_grad_norm="
            f"{float(metrics.get('train_diagnosis_embedding_grad_norm', 0.0)):.6f} "
            f"train_procedure_embedding_grad_norm="
            f"{float(metrics.get('train_procedure_embedding_grad_norm', 0.0)):.6f} "
            f"train_medication_history_embedding_grad_norm="
            f"{float(metrics.get('train_medication_history_embedding_grad_norm', 0.0)):.6f} "
            f"train_code_embedding_sanitized_events="
            f"{float(metrics.get('train_code_embedding_sanitized_events', 0.0)):.0f} "
            f"train_data_time={float(metrics['train_data_time']):.3f}s "
            f"train_step_time={float(metrics['train_step_time']):.3f}s "
            f"val_total_loss={float(metrics['val_total_loss']):.6f} "
            f"val_pred_bce_loss={float(metrics['val_pred_bce_loss']):.6f} "
            f"val_margin_loss={float(metrics['val_margin_loss']):.6f} "
            f"val_ddi_loss={float(metrics['val_ddi_loss']):.6f} "
            f"val_lambda_ddi_current={float(metrics['val_lambda_ddi_current']):.6f} "
            f"val_jaccard={float(metrics['val_jaccard']):.6f} "
            f"val_f1={float(metrics['val_f1']):.6f} "
            f"val_prauc={float(metrics['val_prauc']):.6f} "
            f"val_ddi_rate={float(metrics['val_ddi_rate']):.6f} "
            f"val_avg_drugs={float(metrics['val_avg_drugs']):.6f} "
            f"val_data_time={float(metrics['val_data_time']):.3f}s "
            f"val_step_time={float(metrics['val_step_time']):.3f}s "
            f"history_mode={self.history_mode} "
            f"use_retrieval={self.use_retrieval} "
            f"bottleneck={bottleneck}"
        )
        _log_line(summary)
        log_payload = {
            "epoch": int(epoch),
            "use_self_history": self.use_self_history,
            "use_ddi": self.use_ddi,
            "use_retrieval": self.use_retrieval,
            "history_mode": self.history_mode,
            "train_loss": _to_float(metrics["train_total_loss"]),
            "val_loss": _to_float(metrics["val_total_loss"]),
            **{key: _to_float(value) for key, value in metrics.items()},
        }
        with self.metrics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_payload, ensure_ascii=True, sort_keys=True))
            handle.write("\n")

    def _write_history_artifacts(self, history: list[dict[str, float]]) -> None:
        normalized_history = [
            {
                "epoch": int(record["epoch"]),
                "use_self_history": self.use_self_history,
                "use_ddi": self.use_ddi,
                "use_retrieval": self.use_retrieval,
                "history_mode": self.history_mode,
                "train_loss": float(record["train_total_loss"]),
                "val_loss": float(record["val_total_loss"]),
                **{key: float(value) for key, value in record.items() if key != "epoch"},
            }
            for record in history
        ]
        with self.metrics_per_epoch_json_path.open("w", encoding="utf-8") as handle:
            json.dump(normalized_history, handle, ensure_ascii=True, indent=2, sort_keys=True)

        fieldnames: list[str] = []
        for row in normalized_history:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with self.metrics_per_epoch_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in normalized_history:
                writer.writerow(row)

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

        epoch_progress = tqdm(
            range(1, int(epochs) + 1),
            desc="Training epochs",
            unit="epoch",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        for epoch in epoch_progress:
            code_embeddings_trainable = epoch > self.freeze_code_embedding_epochs
            self._set_code_embeddings_trainable(
                trainable=code_embeddings_trainable,
                epoch=epoch,
            )
            if self.use_retrieval and hasattr(self.model, "refresh_retrieval_memory_bank"):
                retrieval_bank = self.model.refresh_retrieval_memory_bank(
                    train_dataloader,
                    split_name="train",
                    device=self.device,
                    progress_desc=f"Refreshing retrieval bank (train, epoch {epoch})",
                )
                if retrieval_bank is not None:
                    _log_line(
                        f"Refreshed retrieval memory bank for epoch {epoch} "
                        f"(visits={retrieval_bank.num_visits}, history_mode={self.history_mode})"
                    )
                    if not self._retrieval_policy_logged and hasattr(self.model, "get_retrieval_policy"):
                        retrieval_policy = self.model.get_retrieval_policy()
                        _log_line(
                            "Retrieval policy: "
                            f"absolute_time={bool(retrieval_policy.get('has_absolute_time', False))} "
                            f"same_patient_future_blocked={bool(retrieval_policy.get('same_patient_future_blocked', False))} "
                            f"cross_patient_absolute_temporal_filter="
                            f"{bool(retrieval_policy.get('cross_patient_absolute_temporal_filter', False))}"
                        )
                        _log_line(str(retrieval_policy.get("notes", "")))
                        self._retrieval_policy_logged = True
            train_metrics = self.train_one_epoch(train_dataloader, epoch=epoch)
            if self.use_retrieval and hasattr(self.model, "refresh_retrieval_memory_bank"):
                retrieval_bank = self.model.refresh_retrieval_memory_bank(
                    train_dataloader,
                    split_name="train",
                    device=self.device,
                    progress_desc=f"Refreshing retrieval bank (val, epoch {epoch})",
                )
                if retrieval_bank is not None:
                    _log_line(
                        f"Refreshed retrieval memory bank for validation of epoch {epoch} "
                        f"(visits={retrieval_bank.num_visits}, history_mode={self.history_mode})"
                    )
            val_metrics = self.validate_one_epoch(val_dataloader, epoch=epoch)
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

            if hasattr(epoch_progress, "set_postfix"):
                epoch_progress.set_postfix(
                    train_total=f"{float(epoch_metrics['train_total_loss']):.4f}",
                    val_total=f"{float(epoch_metrics['val_total_loss']):.4f}",
                    val_jac=f"{float(epoch_metrics['val_jaccard']):.4f}",
                    val_ddi=f"{float(epoch_metrics['val_ddi_rate']):.4f}",
                    data=f"{float(epoch_metrics['train_data_time']):.3f}s",
                    step=f"{float(epoch_metrics['train_step_time']):.3f}s",
                )
            self.log_metrics(epoch=epoch, metrics=epoch_metrics)
            history.append({"epoch": float(epoch), **epoch_metrics})

        self._write_history_artifacts(history)
        return {
            "history": history,
            "best_metric": self.best_metric,
            "best_checkpoint_path": None if best_checkpoint_path is None else str(best_checkpoint_path),
            "monitor_metric": self.monitor_metric,
            "metrics_per_epoch_json": str(self.metrics_per_epoch_json_path),
            "metrics_per_epoch_csv": str(self.metrics_per_epoch_csv_path),
            "best_metrics_json": str(self.best_metrics_path),
        }


__all__ = ["Trainer", "_LOSS_KEYS", "_move_batch_to_device", "_to_float"]
