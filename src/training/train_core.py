from __future__ import annotations

import argparse
import copy
import itertools
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    class _PlainProgress:
        def __init__(self, iterable, **_: Any) -> None:
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, **_: Any) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(iterable, **kwargs: Any):
        return _PlainProgress(iterable, **kwargs)
else:
    def tqdm(iterable, **kwargs: Any):
        return _tqdm(iterable, **kwargs)

from src.data.build_vocab import load_vocab_bundle
from src.data.dataset import (
    DirectParquetTrajectoryDataset,
    MIMICTrajectoryDataset,
    ShardLengthBatchSampler,
    build_collate_fn,
    collate_batch,
    detect_trajectory_layout,
)
from src.evaluation.thresholding import normalize_threshold_tuning_config, sweep_multilabel_thresholds
from src.losses.contrastive import compute_contrastive_loss, compute_embedding_similarity_stats
from src.models.ddi_regularization import DDIRegularizer, load_ddi_artifact
from src.models.full_model import RetrievalEvidenceFusionModel
from src.models.fusion import FusionModule
from src.models.history_selector import HistorySelector
from src.models.medication_decoder import MedicationDecoder
from src.models.patient_state_encoder import PatientStateEncoder
from src.retrieval.memory_bank import MemoryBank
from src.training.losses import MedicationRecommendationLoss, extract_last_valid_targets
from src.training.trainer import (
    Trainer,
    _LOSS_KEYS,
    _accumulate_fusion_diagnostics,
    _accumulate_retrieval_diagnostics,
    _move_batch_to_device,
    _to_float,
)
from src.utils.io import ensure_dir, load_yaml_config, read_json, resolve_path
from src.utils.runtime_truth import build_core_runtime_truth, normalize_ddi_context


_CORE_CONTRASTIVE_METRIC_KEYS = (
    "contrastive_loss",
    "weighted_contrastive_loss",
    "embedding_norm_mean",
    "embedding_norm_std",
    "similarity_mean",
    "similarity_std",
)
_CORE_EPOCH_METRIC_KEYS = (*_LOSS_KEYS, *_CORE_CONTRASTIVE_METRIC_KEYS)


class TqdmCoreTrainer(Trainer):
    """Core trainer that adds tqdm progress bars for train and validation."""

    def __init__(
        self,
        *args: Any,
        retrieval_memory_bank: MemoryBank | None = None,
        retrieval_enabled: bool = False,
        contrastive_lambda: float = 0.0,
        contrastive_temperature: float = 0.07,
        **kwargs: Any,
    ) -> None:
        if float(contrastive_lambda) < 0.0:
            raise ValueError(f"contrastive_lambda must be non-negative, got {contrastive_lambda!r}")
        if float(contrastive_temperature) <= 0.0:
            raise ValueError(
                f"contrastive_temperature must be positive, got {contrastive_temperature!r}"
            )
        super().__init__(*args, **kwargs)
        self.retrieval_memory_bank = retrieval_memory_bank
        self.retrieval_enabled = bool(retrieval_enabled)
        self.contrastive_lambda = float(contrastive_lambda)
        self.contrastive_temperature = float(contrastive_temperature)

    def _forward_model(self, batch_on_device: Mapping[str, Any]) -> dict[str, Any]:
        if not self.retrieval_enabled:
            return super()._forward_model(batch_on_device)
        return self.model(
            batch_on_device,
            mode="core",
            decoder_top_k=self.decoder_top_k,
            compute_ddi_metrics=False,
            memory_bank=self.retrieval_memory_bank if self.retrieval_enabled else None,
            records=batch_on_device.get("records") if self.retrieval_enabled else None,
        )

    def _compute_loss_outputs(
        self,
        *,
        outputs: Mapping[str, Any],
        batch_on_device: Mapping[str, Any],
    ) -> dict[str, Any]:
        loss_outputs = dict(
            super()._compute_loss_outputs(
                outputs=outputs,
                batch_on_device=batch_on_device,
            )
        )
        pooled_state = outputs.get("pooled_state")
        if not isinstance(pooled_state, torch.Tensor):
            raise RuntimeError("Model did not return `pooled_state` for contrastive diagnostics.")

        if self.contrastive_lambda > 0.0:
            if "subject_ids" not in batch_on_device:
                raise KeyError("Batch is missing `subject_ids` required for contrastive loss.")
            contrastive_loss = compute_contrastive_loss(
                pooled_state,
                batch_on_device["subject_ids"],
                temperature=self.contrastive_temperature,
            )
        else:
            contrastive_loss = loss_outputs["total_loss"].new_zeros(())

        weighted_contrastive_loss = contrastive_loss * self.contrastive_lambda
        loss_outputs["total_loss"] = loss_outputs["total_loss"] + weighted_contrastive_loss
        loss_outputs["contrastive_loss"] = contrastive_loss
        loss_outputs["weighted_contrastive_loss"] = weighted_contrastive_loss
        loss_outputs.update(compute_embedding_similarity_stats(pooled_state))
        return loss_outputs

    def _create_progress(
        self,
        dataloader: DataLoader,
        *,
        phase: str,
        training: bool,
        max_steps: int | None,
    ) -> Any | None:
        _ = dataloader
        _ = training
        epoch_index = int(getattr(self, "_current_epoch", 0))
        total_epochs = int(getattr(self, "_fit_total_epochs", 0))
        epoch_label = f"{epoch_index}/{total_epochs}" if total_epochs > 0 else str(epoch_index)
        desc = f"{phase.upper()} {epoch_label}"
        return tqdm(
            dataloader,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            total=max_steps,
        )

    def _close_progress(self, progress: Any | None) -> None:
        if progress is not None:
            progress.close()

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
        _ = phase
        _ = step_index
        if progress is None or total_examples <= 0:
            return
        running_total_loss = float(totals["total_loss"]) / float(total_examples)
        running_prediction_loss = float(totals["prediction_loss"]) / float(total_examples)
        running_ddi_loss = float(totals["ddi_loss"]) / float(total_examples)
        total_time = float(timing_totals["data_time"]) + float(timing_totals["step_time"])
        samples_per_sec = 0.0 if total_time <= 0.0 else float(total_examples) / total_time
        postfix = {
            "total_loss": f"{running_total_loss:.4f}",
            "pred_loss": f"{running_prediction_loss:.4f}",
            "sps": f"{samples_per_sec:.2f}",
        }
        if bool(getattr(self.loss_fn, "ddi_active", False)):
            postfix["ddi_loss"] = f"{running_ddi_loss:.4f}"
        else:
            postfix["ddi"] = "inactive"
        if self.contrastive_lambda > 0.0 and "contrastive_loss" in totals:
            postfix["ctr_loss"] = f"{float(totals['contrastive_loss']) / float(total_examples):.4f}"
        progress.set_postfix(**postfix)

    def _run_one_epoch(
        self,
        dataloader: DataLoader,
        *,
        training: bool,
    ) -> dict[str, float]:
        phase = "train" if training else "val"
        totals = {key: 0.0 for key in _CORE_EPOCH_METRIC_KEYS}
        timing_totals = {
            "data_time": 0.0,
            "transfer_time": 0.0,
            "forward_time": 0.0,
            "loss_time": 0.0,
            "backward_time": 0.0,
            "optimizer_time": 0.0,
            "step_time": 0.0,
        }
        detailed_timing_totals: dict[str, float] = {}
        fusion_diagnostic_totals: dict[str, float] = {}
        total_examples = 0
        step_count = 0
        max_steps = self._max_epoch_steps(dataloader)
        prediction_payload_collector = self._prediction_payload_collector(training=training)

        self.model.train(mode=training)
        grad_context = torch.enable_grad if training else torch.no_grad
        progress = self._create_progress(
            dataloader,
            phase=phase,
            training=training,
            max_steps=max_steps,
        )
        iterable_source = dataloader if progress is None else progress
        iterable = iterable_source if max_steps is None else itertools.islice(iterable_source, max_steps)
        batches_since_step = 0
        self.optimizer.zero_grad(set_to_none=True)
        last_step_end = time.perf_counter()

        try:
            for step_index, batch in enumerate(iterable, start=1):
                step_context = f"{phase} step {step_index}"
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
                self._validate_batch_inputs_finite(
                    batch_on_device,
                    context=f"{step_context} before forward",
                )

                with grad_context():
                    forward_start = time.perf_counter()
                    with self._autocast_context():
                        outputs = self._forward_model(batch_on_device)
                    self._validate_model_outputs_finite(
                        outputs,
                        context=f"{step_context} after forward",
                    )
                    self._collect_runtime_timing(detailed_timing_totals, outputs)
                    _accumulate_fusion_diagnostics(
                        fusion_diagnostic_totals,
                        outputs,
                        batch_size=batch_size,
                    )
                    _accumulate_retrieval_diagnostics(
                        fusion_diagnostic_totals,
                        outputs,
                        batch_size=batch_size,
                    )
                    if prediction_payload_collector is not None:
                        self._collect_prediction_payload_batch(
                            collector=prediction_payload_collector,
                            outputs=outputs,
                            batch_on_device=batch_on_device,
                        )
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
                        if self.grad_scaler_enabled and self.scaler is not None:
                            self.scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        self._sync_timing()
                        backward_time = time.perf_counter() - backward_start
                        batches_since_step += 1

                        if batches_since_step >= self.grad_accum_steps:
                            optimizer_start = time.perf_counter()
                            self._optimizer_step(
                                context=f"{step_context} after optimizer step",
                            )
                            self._sync_timing()
                            optimizer_time = time.perf_counter() - optimizer_start
                            batches_since_step = 0

                step_compute_time = transfer_time + forward_time + loss_time + backward_time + optimizer_time
                total_examples += batch_size
                step_count += 1
                for key in _CORE_EPOCH_METRIC_KEYS:
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
            self._finalize_prediction_payload_collector(
                collector=prediction_payload_collector,
                dataloader=dataloader,
                training=training,
            )

        if training and batches_since_step > 0:
            optimizer_start = time.perf_counter()
            self._optimizer_step(context=f"{phase} epoch-end optimizer flush")
            self._sync_timing()
            optimizer_flush_time = time.perf_counter() - optimizer_start
            timing_totals["optimizer_time"] += optimizer_flush_time
            timing_totals["step_time"] += optimizer_flush_time

        if total_examples <= 0:
            raise ValueError(f"{phase} dataloader produced zero valid examples")

        epoch_metrics = {
            f"{phase}_{key}": totals[key] / float(total_examples)
            for key in _CORE_EPOCH_METRIC_KEYS
        }
        epoch_metrics.update(
            {
                f"{phase}_{key}": value / float(total_examples)
                for key, value in fusion_diagnostic_totals.items()
            }
        )
        epoch_metrics.update(
            self._timing_metric_payload(
                phase=phase,
                timing_totals=timing_totals,
                total_examples=total_examples,
                step_count=step_count,
            )
        )
        epoch_metrics.update(
            self._detailed_timing_metric_payload(
                phase=phase,
                timing_totals=detailed_timing_totals,
                step_count=step_count,
            )
        )
        return epoch_metrics

    def _threshold_tuning_config(self) -> dict[str, Any]:
        raw_config = self.run_context.get("threshold_tuning")
        if isinstance(raw_config, Mapping):
            return normalize_threshold_tuning_config(raw_config)
        return normalize_threshold_tuning_config({})

    def _prediction_payload_collector(self, *, training: bool) -> Any | None:
        if training:
            return None
        tuning_cfg = self._threshold_tuning_config()
        if not bool(tuning_cfg.get("enabled", False)):
            return None
        return {"drug_probs": [], "targets": []}

    def _resolve_prediction_targets(
        self,
        outputs: Mapping[str, Any],
        batch_on_device: Mapping[str, Any],
    ) -> torch.Tensor:
        final_target_drugs = outputs.get("final_target_drugs")
        if isinstance(final_target_drugs, torch.Tensor):
            return final_target_drugs
        final_target_drugs = batch_on_device.get("final_target_drugs")
        if isinstance(final_target_drugs, torch.Tensor):
            return final_target_drugs
        resolved_targets = outputs.get("target_drugs")
        if not isinstance(resolved_targets, torch.Tensor):
            resolved_targets = batch_on_device.get("target_drugs")
        if not isinstance(resolved_targets, torch.Tensor):
            raise RuntimeError("Validation batch is missing final_target_drugs and target_drugs.")
        if resolved_targets.ndim == 3:
            return extract_last_valid_targets(resolved_targets, batch_on_device["visit_mask"])
        return resolved_targets

    def _collect_prediction_payload_batch(
        self,
        *,
        collector: Any,
        outputs: Mapping[str, Any],
        batch_on_device: Mapping[str, Any],
    ) -> None:
        if collector is None:
            return
        drug_probs = outputs.get("drug_probs")
        if not isinstance(drug_probs, torch.Tensor):
            raise RuntimeError("Model did not return `drug_probs` during threshold tuning.")
        targets = self._resolve_prediction_targets(outputs, batch_on_device)
        collector["drug_probs"].append(drug_probs.detach().cpu())
        collector["targets"].append(targets.detach().cpu())

    def _finalize_prediction_payload_collector(
        self,
        *,
        collector: Any,
        dataloader: DataLoader,
        training: bool,
    ) -> None:
        super()._finalize_prediction_payload_collector(
            collector=collector,
            dataloader=dataloader,
            training=training,
        )
        if training or collector is None:
            return
        max_steps = self._max_epoch_steps(dataloader)
        try:
            total_steps = int(len(dataloader))
        except TypeError:
            total_steps = None
        if max_steps is not None and total_steps is not None and int(max_steps) < total_steps:
            return
        if not collector["drug_probs"] or not collector["targets"]:
            return
        self._cached_validation_prediction_payload = (
            torch.cat(collector["drug_probs"], dim=0),
            torch.cat(collector["targets"], dim=0),
        )
        self._cached_validation_payload_dataloader_id = id(dataloader)

    def _collect_prediction_payload(self, dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        cached_payload = self._cached_validation_prediction_payload_for(dataloader)
        if cached_payload is not None:
            return cached_payload

        collected_probs: list[torch.Tensor] = []
        collected_targets: list[torch.Tensor] = []

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch_on_device = {
                    key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                outputs = self._forward_model(batch_on_device)
                drug_probs = outputs.get("drug_probs")
                if drug_probs is None:
                    raise RuntimeError("Model did not return `drug_probs` during threshold tuning.")
                targets = self._resolve_prediction_targets(outputs, batch_on_device)
                collected_probs.append(drug_probs.detach().cpu())
                collected_targets.append(targets.detach().cpu())

        if not collected_probs or not collected_targets:
            raise ValueError("Threshold tuning requires at least one validation batch")
        return torch.cat(collected_probs, dim=0), torch.cat(collected_targets, dim=0)

    def _run_threshold_tuning(self, val_dataloader: DataLoader) -> dict[str, float]:
        tuning_start = time.perf_counter()
        tuning_cfg = self._threshold_tuning_config()
        if not bool(tuning_cfg.get("enabled", False)):
            return {}
        if str(tuning_cfg.get("split", "val")).strip().lower() != "val":
            raise ValueError(
                "Threshold tuning must run on the validation split only. "
                f"Received split={tuning_cfg.get('split')!r}."
            )

        probabilities, targets = self._collect_prediction_payload(val_dataloader)
        ddi_matrix = None
        ddi_regularizer = getattr(self.loss_fn, "ddi_regularizer", None)
        if ddi_regularizer is not None and hasattr(ddi_regularizer, "ddi_matrix"):
            ddi_matrix = ddi_regularizer.ddi_matrix.detach().cpu()

        sweep_result = sweep_multilabel_thresholds(
            y_true=targets,
            y_score=probabilities,
            candidates=tuning_cfg["candidates"],
            metric=str(tuning_cfg["metric"]),
            tie_breaker=str(tuning_cfg["tie_breaker"]),
            ddi_matrix=ddi_matrix,
        )
        best_metrics = dict(sweep_result["best_metrics"])
        threshold_selection = {
            "source": "validation_sweep",
            "split": "val",
            "metric": str(tuning_cfg["metric"]),
            "tie_breaker": str(tuning_cfg["tie_breaker"]),
            "candidates": [float(value) for value in tuning_cfg["candidates"]],
            "best_threshold": float(sweep_result["best_threshold"]),
        }
        self.run_context["effective_threshold"] = float(sweep_result["best_threshold"])
        self.run_context["threshold_selection"] = threshold_selection

        threshold_metrics = {
            "val_f1_tuned": float(best_metrics["f1"]),
            "val_jaccard_tuned": float(best_metrics["jaccard"]),
            "val_prauc_tuned": float(sweep_result["prauc"]),
            "val_threshold_best": float(sweep_result["best_threshold"]),
            "val_avg_predicted_drugs_tuned": float(best_metrics["avg_predicted_drugs"]),
            "val_avg_true_drugs": float(best_metrics["avg_true_drugs"]),
        }
        ddi_rate = best_metrics.get("ddi_rate")
        if ddi_rate is not None:
            threshold_metrics["val_ddi_rate_tuned"] = float(ddi_rate)
        if self.detailed_timing_enabled:
            threshold_metrics["val_threshold_tuning_time"] = time.perf_counter() - tuning_start
        return threshold_metrics

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
        self._fit_total_epochs = int(epochs)

        for epoch in range(1, int(epochs) + 1):
            self._current_epoch = int(epoch)
            self._set_dataloader_epoch(train_dataloader, epoch=epoch)
            train_metrics = self.train_one_epoch(train_dataloader)
            val_metrics = self.validate_one_epoch(val_dataloader)
            tuned_threshold_metrics = self._run_threshold_tuning(val_dataloader)
            epoch_metrics = {**train_metrics, **val_metrics, **tuned_threshold_metrics}

            self._step_scheduler(epoch_metrics)

            maybe_best = self.save_best_checkpoint(
                epoch=epoch,
                epoch_metrics=epoch_metrics,
                extra_state=extra_checkpoint_state,
            )
            if maybe_best is not None:
                best_checkpoint_path = maybe_best

            epoch_metrics = {**epoch_metrics, **self._epoch_aux_timing_metrics()}
            metrics_log_write_time = self.log_metrics(epoch=epoch, metrics=epoch_metrics)
            history.append(
                {
                    "epoch": float(epoch),
                    **epoch_metrics,
                    **(
                        {"metrics_log_write_time": float(metrics_log_write_time)}
                        if self.detailed_timing_enabled
                        else {}
                    ),
                }
            )
            if self._maybe_trigger_early_stopping():
                print(f"Early stopping triggered at epoch {epoch}: {self.stop_reason}")
                break

        return {
            "history": history,
            "best_metric": self.best_metric,
            "best_checkpoint_path": None if best_checkpoint_path is None else str(best_checkpoint_path),
            "monitor_metric": self.monitor_metric,
            "epochs_completed": len(history),
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the core ClinRec recommendation model.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to configs/train.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Path to configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml", help="Path to configs/model.yaml")
    parser.add_argument(
        "--profile",
        choices=("safe", "balanced", "fast"),
        default=None,
        help="Optional runtime profile override",
    )
    parser.add_argument("--processed-root", default=None, help="Optional override for processed data root")
    parser.add_argument("--vocab-root", default=None, help="Optional override for vocab directory")
    parser.add_argument("--ddi-matrix-path", default=None, help="Optional override for DDI matrix artifact")
    parser.add_argument("--device", default=None, help="Optional override for runtime device")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for random seed")
    return parser.parse_args()


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_profile_overrides(
    config: Mapping[str, Any],
    *,
    profile_name: str,
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    profile_payload = dict(resolved.pop("profiles", {})).get(profile_name, {})
    if profile_payload:
        resolved = _deep_merge(resolved, profile_payload)
    resolved["_selected_profile"] = profile_name
    return resolved


def resolve_profile_name(train_config: Mapping[str, Any], cli_profile: str | None) -> str:
    if cli_profile is not None:
        return str(cli_profile)
    runtime_cfg = dict(train_config.get("runtime", {}))
    return str(runtime_cfg.get("profile", "balanced"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> torch.device:
    device = torch.device(str(requested_device))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is not available; falling back to CPU.")
        return torch.device("cpu")
    return device


def _first_existing_path(
    candidates: Sequence[tuple[str, Path | None]],
    *,
    kind: str,
) -> tuple[Path, str]:
    checked: list[str] = []
    for label, candidate in candidates:
        if candidate is None:
            continue
        checked.append(f"{label}={candidate}")
        if candidate.exists():
            return candidate, label
    raise FileNotFoundError(
        f"Unable to resolve {kind}. Checked candidates: {checked if checked else ['<none>']}"
    )


def validate_core_runtime_config(
    *,
    runtime_cfg: Mapping[str, Any],
    core_cfg: Mapping[str, Any],
    context_label: str,
) -> None:
    runtime_mode = str(runtime_cfg.get("mode", "core")).strip().lower()
    if runtime_mode != "core":
        raise ValueError(f"{context_label} only supports runtime.mode=core, got {runtime_mode!r}.")
    if bool(core_cfg.get("use_group_encoder", False)):
        raise ValueError(f"{context_label} does not support core.use_group_encoder=true.")


def validate_core_model_config(model_config: Mapping[str, Any]) -> None:
    sequence_cfg = dict(model_config.get("sequence", {}))
    rnn_type = str(sequence_cfg.get("rnn_type", "gru")).strip().lower()
    if rnn_type != "gru":
        raise ValueError(f"The core path only supports sequence.rnn_type=gru, got {rnn_type!r}.")
    if bool(sequence_cfg.get("bidirectional", False)):
        raise ValueError("The core path does not support sequence.bidirectional=true.")


def resolve_runtime_paths(
    *,
    project_root: Path,
    train_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Path]:
    train_paths = dict(train_config.get("paths", {}))
    data_paths = dict(data_config.get("paths", {}))

    processed_root, processed_root_source = _first_existing_path(
        [
            ("arg:processed_root", None if args.processed_root is None else Path(args.processed_root).resolve()),
            (
                "train.paths.processed_root",
                None if train_paths.get("processed_root") is None else resolve_path(project_root, train_paths["processed_root"]).resolve(),
            ),
            (
                "data.paths.processed_root",
                None if data_paths.get("processed_root") is None else resolve_path(project_root, data_paths["processed_root"]).resolve(),
            ),
            ("compat:handover_data/processed", (project_root / "handover_data" / "processed").resolve()),
        ],
        kind="processed_root",
    )
    vocab_root, vocab_root_source = _first_existing_path(
        [
            ("arg:vocab_root", None if args.vocab_root is None else Path(args.vocab_root).resolve()),
            (
                "train.paths.vocab_root",
                None if train_paths.get("vocab_root") is None else resolve_path(project_root, train_paths["vocab_root"]).resolve(),
            ),
            (
                "data.paths.interim_root/vocab",
                None
                if data_paths.get("interim_root") is None
                else (resolve_path(project_root, data_paths["interim_root"]).resolve() / "vocab"),
            ),
            ("compat:handover_data/vocab", (project_root / "handover_data" / "vocab").resolve()),
        ],
        kind="vocab_root",
    )
    ddi_matrix_path, ddi_matrix_path_source = _first_existing_path(
        [
            ("arg:ddi_matrix_path", None if args.ddi_matrix_path is None else Path(args.ddi_matrix_path).resolve()),
            (
                "train.paths.ddi_matrix_path",
                None if train_paths.get("ddi_matrix_path") is None else resolve_path(project_root, train_paths["ddi_matrix_path"]).resolve(),
            ),
            (
                "compat:handover_data/processed/ddi/drug_ddi.pt",
                (project_root / "handover_data" / "processed" / "ddi" / "drug_ddi.pt").resolve(),
            ),
        ],
        kind="ddi_matrix_path",
    )

    checkpoint_dir = ensure_dir(resolve_path(project_root, train_paths.get("checkpoint_dir", "outputs/checkpoints")).resolve())
    log_dir = ensure_dir(resolve_path(project_root, train_paths.get("log_dir", "outputs/logs")).resolve())

    print("Resolved runtime paths:")
    print(f"  processed_root: {processed_root} [{processed_root_source}]")
    print(f"  vocab_root: {vocab_root} [{vocab_root_source}]")
    print(f"  ddi_matrix_path: {ddi_matrix_path} [{ddi_matrix_path_source}]")
    print(f"  checkpoint_dir: {checkpoint_dir}")
    print(f"  log_dir: {log_dir}")
    if processed_root_source.startswith("compat:") or vocab_root_source.startswith("compat:") or ddi_matrix_path_source.startswith("compat:"):
        print("Compatibility fallback is active: runtime is using handover_data artifacts instead of canonical data/... paths.")

    return {
        "processed_root": processed_root,
        "vocab_root": vocab_root,
        "ddi_matrix_path": ddi_matrix_path,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir,
    }


def build_runtime_data_config_file(
    *,
    data_config: Mapping[str, Any],
    processed_root: Path,
    vocab_root: Path,
    temp_dir: Path,
) -> Path:
    runtime_config = copy.deepcopy({key: value for key, value in data_config.items() if not str(key).startswith("_")})
    runtime_config.setdefault("paths", {})
    runtime_config["paths"]["processed_root"] = str(processed_root)
    runtime_config["paths"]["interim_root"] = str(vocab_root.parent)

    runtime_config_path = temp_dir / "runtime_data.yaml"
    runtime_config_path.write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8")
    return runtime_config_path


def build_dataset(
    *,
    split: str,
    runtime_data_config_path: Path,
    processed_root: Path,
    drug_vocab_size: int,
    max_open_shards: int | None = None,
) -> Dataset:
    layout = detect_trajectory_layout(
        split,
        runtime_data_config_path,
        processed_root=processed_root,
    )
    print(
        f"Using dataset layout for split `{split}`: "
        f"{layout['kind']} ({layout['description']}) at {layout['manifest_path']}"
    )
    if layout["kind"] == "direct_split_manifest":
        dataset = DirectParquetTrajectoryDataset(
            split,
            processed_root,
            drug_vocab_size=drug_vocab_size,
            max_open_shards=(
                int(max_open_shards)
                if max_open_shards is not None
                else 2
            ),
        )
        dataset.layout_kind = layout["kind"]
        return dataset

    dataset = MIMICTrajectoryDataset(split, runtime_data_config_path)
    if max_open_shards is not None and hasattr(dataset, "max_open_shards"):
        dataset.max_open_shards = int(max_open_shards)
    dataset.layout_kind = layout["kind"]
    return dataset


def build_dataloaders(
    *,
    runtime_data_config_path: Path,
    processed_root: Path,
    drug_vocab_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    length_bucket_window: int = 256,
    seed: int = 0,
    max_open_shards: int | None = None,
    max_visits: int | None = None,
    max_history: int | None = None,
    include_records: bool = False,
) -> tuple[DataLoader, DataLoader]:
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}")
    if int(num_workers) < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers!r}")

    train_dataset = build_dataset(
        split="train",
        runtime_data_config_path=runtime_data_config_path,
        processed_root=processed_root,
        drug_vocab_size=drug_vocab_size,
        max_open_shards=max_open_shards,
    )
    val_dataset = build_dataset(
        split="val",
        runtime_data_config_path=runtime_data_config_path,
        processed_root=processed_root,
        drug_vocab_size=drug_vocab_size,
        max_open_shards=max_open_shards,
    )
    if len(train_dataset) <= 0:
        raise ValueError("Training dataset is empty")
    if len(val_dataset) <= 0:
        raise ValueError("Validation dataset is empty")

    collate_fn = build_collate_fn(
        include_full_targets=False,
        include_final_target=True,
        max_visits=max_visits,
        max_history=max_history,
        include_records=include_records,
    )
    loader_kwargs: dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": collate_fn,
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    batch_sampler = None
    if getattr(train_dataset, "shard_row_indices", None):
        batch_sampler = ShardLengthBatchSampler(
            train_dataset,
            batch_size=int(batch_size),
            length_bucket_window=int(length_bucket_window),
            shuffle=True,
            seed=int(seed),
        )

    train_loader = (
        DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )
        if batch_sampler is not None
        else DataLoader(
            train_dataset,
            batch_size=int(batch_size),
            shuffle=True,
            **loader_kwargs,
        )
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def load_vocab_sizes(vocab_root: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for name in ("diagnosis", "procedure", "drug"):
        payload = read_json(vocab_root / f"{name}_vocab.json")
        sizes[name] = int(payload["size"])
    return sizes


def build_core_model(
    *,
    train_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    runtime_data_config_path: Path,
    vocab_root: Path,
    ddi_matrix_path: Path,
    pos_weight: torch.Tensor | None = None,
) -> tuple[RetrievalEvidenceFusionModel, MedicationRecommendationLoss]:
    validate_core_runtime_config(
        runtime_cfg=dict(train_config.get("runtime", {"mode": "core"})),
        core_cfg=dict(train_config.get("core", {})),
        context_label="build_core_model",
    )
    validate_core_model_config(model_config)
    vocab_sizes = load_vocab_sizes(vocab_root)
    runtime_data_config = load_yaml_config(runtime_data_config_path)
    feature_cfg = dict(runtime_data_config.get("features", {}))
    spark_cfg = dict(runtime_data_config.get("spark", {}))
    sample_dataset = build_dataset(
        split="train",
        runtime_data_config_path=runtime_data_config_path,
        processed_root=Path(train_config["_resolved_paths"]["processed_root"]),
        drug_vocab_size=vocab_sizes["drug"],
        max_open_shards=spark_cfg.get("max_open_shards_per_dataset"),
    )
    sample_batch = collate_batch(
        [sample_dataset[0]],
        include_full_targets=False,
        include_final_target=True,
        max_visits=feature_cfg.get("max_visits"),
        max_history=feature_cfg.get("max_history"),
    )

    model_cfg = dict(model_config.get("model", {}))
    embedding_cfg = dict(model_config.get("embedding", {}))
    retrieval_cfg = dict(model_config.get("retrieval", {}))
    core_cfg = dict(train_config.get("core", {}))
    core_retrieval_enabled = bool(core_cfg.get("use_retrieval", False))
    history_cfg = dict(model_config.get("history_selector", {}))
    fusion_cfg = dict(model_config.get("fusion", {}))
    decoder_cfg = dict(model_config.get("decoder", {}))
    label_correlation_cfg = dict(decoder_cfg.get("label_correlation", {}))

    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    num_layers = int(model_cfg.get("num_layers", 1))
    model_dropout = float(model_cfg.get("dropout", 0.1))
    code_embedding_dim = int(embedding_cfg.get("diag_dim", hidden_dim))
    proc_dim = int(embedding_cfg.get("proc_dim", code_embedding_dim))
    if proc_dim != code_embedding_dim:
        raise ValueError(
            f"PatientStateEncoder currently expects a shared code embedding dim; got diag_dim={code_embedding_dim}, proc_dim={proc_dim}"
        )
    numeric_projection_dim = int(embedding_cfg.get("lab_dim", 64))
    vital_dim = int(embedding_cfg.get("vital_dim", numeric_projection_dim))
    if vital_dim != numeric_projection_dim:
        raise ValueError(
            "PatientStateEncoder currently expects a shared numeric projection dim; "
            f"got lab_dim={numeric_projection_dim}, vital_dim={vital_dim}"
        )

    encoder = PatientStateEncoder(
        diagnosis_vocab_size=vocab_sizes["diagnosis"],
        procedure_vocab_size=vocab_sizes["procedure"],
        drug_vocab_size=vocab_sizes["drug"],
        num_lab_features=int(sample_batch["lab_values"].shape[-1]),
        num_vital_features=int(sample_batch["vital_values"].shape[-1]),
        code_embedding_dim=code_embedding_dim,
        medication_embedding_dim=int(embedding_cfg.get("drug_dim", hidden_dim)),
        numeric_projection_dim=numeric_projection_dim,
        time_embedding_dim=int(embedding_cfg.get("time_dim", 32)),
        visit_hidden_dim=hidden_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=model_dropout,
    )
    history_selector = HistorySelector(
        hidden_dim=hidden_dim,
        dropout=float(history_cfg.get("dropout", 0.1)),
        score_bias_weight=float(history_cfg.get("score_bias_weight", 0.5)),
        self_top_k=history_cfg.get("self_top_k", 3),
        neighbor_top_k=history_cfg.get("neighbor_top_k", 3),
        use_retrieval_bias=bool(history_cfg.get("use_retrieval_bias", True)),
    )
    fusion_module = FusionModule(
        hidden_dim=hidden_dim,
        dropout=float(fusion_cfg.get("dropout", model_dropout)),
        strategy=str(fusion_cfg.get("strategy", "gated")),
        current_branch_dropout=float(fusion_cfg.get("current_branch_dropout", 0.0)),
    )
    decoder = MedicationDecoder(
        hidden_dim=hidden_dim,
        drug_vocab_size=vocab_sizes["drug"],
        dropout=model_dropout,
        top_k_metadata=int(train_config.get("prediction", {}).get("top_k", 10)),
        decoder_mode=decoder_cfg.get("mode"),
        label_correlation_enabled=bool(label_correlation_cfg.get("enabled", False)),
        correlation_dim=label_correlation_cfg.get("correlation_dim"),
        patient_residual_weight=float(label_correlation_cfg.get("patient_residual_weight", 0.0)),
        coprescription_residual_weight=float(
            label_correlation_cfg.get("coprescription_residual_weight", 0.0)
        ),
        correlation_dropout=float(label_correlation_cfg.get("dropout", model_dropout)),
    )
    ddi_artifact = load_ddi_artifact(ddi_matrix_path, device="cpu")
    ddi_context = normalize_ddi_context({key: value for key, value in ddi_artifact.items() if key != "matrix"})
    loss_ddi_regularizer = None
    if bool(ddi_context["active"]):
        loss_ddi_regularizer = DDIRegularizer(ddi_artifact, reduction="mean")

    model = RetrievalEvidenceFusionModel(
        encoder,
        history_selector,
        fusion_module,
        medication_decoder=decoder,
        ddi_regularizer=None,
        ddi_context=ddi_context,
        mode="core",
        retrieval_top_k=int(retrieval_cfg.get("top_k", 5)),
        temporal_decay_alpha=float(retrieval_cfg.get("temporal_decay_alpha", 0.05)),
        retrieval_backend=str(retrieval_cfg.get("backend", "bruteforce")),
        use_faiss_if_available=bool(retrieval_cfg.get("use_faiss_if_available", True)),
        allow_cross_split=bool(retrieval_cfg.get("allow_cross_split", False)),
        retrieval_scoring_mode=str(retrieval_cfg.get("scoring_mode", "temporal_relevance")),
        cross_split_policy=retrieval_cfg.get("cross_split_policy"),
        core_retrieval_enabled=core_retrieval_enabled,
        retrieval_leakage_safe=bool(retrieval_cfg.get("leakage_safe", True)),
    )
    model.runtime_truth = build_core_runtime_truth(
        fusion_strategy=fusion_module.strategy,
        ddi_context=ddi_context,
        retrieval_active=core_retrieval_enabled,
        retrieval_status="available" if core_retrieval_enabled else "disabled",
        retrieval_top_k=int(retrieval_cfg.get("top_k", 5)) if core_retrieval_enabled else None,
        retrieval_scoring_mode=str(retrieval_cfg.get("scoring_mode", "temporal_relevance")) if core_retrieval_enabled else None,
        retrieval_cross_split_policy=str(retrieval_cfg.get("cross_split_policy") or ("allow_all" if bool(retrieval_cfg.get("allow_cross_split", False)) else "same_split")) if core_retrieval_enabled else None,
        retrieval_leakage_safe=bool(retrieval_cfg.get("leakage_safe", True)) if core_retrieval_enabled else None,
    )
    loss_cfg = dict(train_config.get("loss", {}))
    loss_fn = MedicationRecommendationLoss(
        lambda_ddi=float(loss_cfg.get("ddi_lambda", 0.0)),
        ddi_regularizer=loss_ddi_regularizer,
        ddi_context=ddi_context,
        pos_weight=pos_weight,
        reduction="mean",
        objective=str(loss_cfg.get("objective", "bce")),
        focal_gamma=float(loss_cfg.get("focal_gamma", 1.5)),
        asymmetric_gamma_pos=float(loss_cfg.get("asymmetric_gamma_pos", 0.0)),
        asymmetric_gamma_neg=float(loss_cfg.get("asymmetric_gamma_neg", 4.0)),
        asymmetric_clip=float(loss_cfg.get("asymmetric_clip", 0.05)),
        fusion_entropy_lambda=float(loss_cfg.get("fusion_entropy_lambda", 0.0)),
        fusion_balance_lambda=float(loss_cfg.get("fusion_balance_lambda", 0.0)),
        ranking_lambda=float(loss_cfg.get("ranking_lambda", 0.0)),
        ranking_objective=str(loss_cfg.get("ranking_objective", "bpr")),
        ranking_margin=float(loss_cfg.get("ranking_margin", 1.0)),
        ranking_num_negatives=int(loss_cfg.get("ranking_num_negatives", 32)),
        ranking_hard_negative_fraction=float(loss_cfg.get("ranking_hard_negative_fraction", 0.5)),
    )
    return model, loss_fn


def _record_final_target_ids(record: Mapping[str, Any], *, drug_vocab_size: int) -> list[int]:
    steps = list(record.get("steps", []))
    if not steps:
        return []
    final_step = dict(steps[-1])
    resolved_ids: list[int] = []
    seen: set[int] = set()
    for raw_drug_id in final_step.get("target_drugs", []):
        drug_id = int(raw_drug_id)
        if drug_id < 0 or drug_id >= int(drug_vocab_size):
            continue
        if drug_id in seen:
            continue
        seen.add(drug_id)
        resolved_ids.append(drug_id)
    return resolved_ids


def build_positive_class_weight(
    *,
    dataset: Dataset,
    drug_vocab_size: int,
    mode: str,
    clip: float,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    resolved_mode = str(mode).strip().lower()
    if resolved_mode in {"", "none", "disabled"}:
        return None, {
            "mode": "disabled",
            "clip": float(clip),
            "num_samples": int(len(dataset)),
            "num_labels_with_positive": 0,
        }
    if resolved_mode != "log_balanced":
        raise ValueError(f"Unsupported loss.pos_weight_mode `{mode}`")
    if float(clip) < 1.0:
        raise ValueError(f"loss.pos_weight_clip must be at least 1.0, got {clip!r}")

    positive_counts = torch.zeros(int(drug_vocab_size), dtype=torch.float32)
    num_samples = int(len(dataset))
    for record_index in range(num_samples):
        record = dataset[record_index]
        for drug_id in _record_final_target_ids(record, drug_vocab_size=drug_vocab_size):
            positive_counts[drug_id] += 1.0

    pos_weight = torch.ones(int(drug_vocab_size), dtype=torch.float32)
    positive_mask = positive_counts > 0
    if bool(positive_mask.any().item()):
        positive_values = positive_counts[positive_mask]
        negative_values = float(num_samples) - positive_values
        ratio = negative_values / positive_values.clamp(min=1.0)
        computed = torch.log1p(ratio).clamp(min=1.0, max=float(clip))
        pos_weight[positive_mask] = computed

    return pos_weight, {
        "mode": "log_balanced",
        "clip": float(clip),
        "num_samples": num_samples,
        "num_labels_with_positive": int(positive_mask.sum().item()),
        "mean_weight": float(pos_weight.mean().item()),
        "max_weight": float(pos_weight.max().item()),
        "min_weight": float(pos_weight.min().item()),
    }


def resolve_initialization_config(
    *,
    train_config: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    initialization_cfg = dict(train_config.get("initialization", {}))
    warm_start_mode = str(initialization_cfg.get("warm_start_mode", "disabled")).strip().lower()
    if warm_start_mode in {"", "none", "off"}:
        warm_start_mode = "disabled"
    if warm_start_mode not in {"disabled", "model_only"}:
        raise ValueError(
            f"Unsupported initialization.warm_start_mode `{warm_start_mode}`. "
            "Expected one of ['disabled', 'model_only']."
        )

    warm_start_checkpoint: Path | None = None
    raw_checkpoint = str(initialization_cfg.get("warm_start_checkpoint") or "").strip()
    if raw_checkpoint:
        warm_start_checkpoint = resolve_path(project_root, raw_checkpoint).resolve()
    if warm_start_mode == "disabled" and warm_start_checkpoint is not None:
        raise ValueError(
            "initialization.warm_start_checkpoint was provided while warm_start_mode is disabled. "
            "Set warm_start_mode=model_only to activate warm start."
        )
    if warm_start_mode != "disabled" and warm_start_checkpoint is None:
        raise ValueError(
            "initialization.warm_start_mode is enabled but initialization.warm_start_checkpoint is missing."
        )
    if warm_start_checkpoint is not None and not warm_start_checkpoint.exists():
        raise FileNotFoundError(f"Warm-start checkpoint does not exist: {warm_start_checkpoint}")

    initialization_mode = "scratch" if warm_start_mode == "disabled" else "warm_start_model_only"
    return {
        "warm_start_mode": warm_start_mode,
        "warm_start_checkpoint": warm_start_checkpoint,
        "strict": bool(initialization_cfg.get("strict", True)),
        "initialization_mode": initialization_mode,
    }


def apply_model_initialization(
    *,
    model: torch.nn.Module,
    train_config: Mapping[str, Any],
) -> dict[str, Any]:
    project_root = Path(train_config["_project_root"]).resolve()
    initialization_state = resolve_initialization_config(
        train_config=train_config,
        project_root=project_root,
    )
    warm_start_checkpoint = initialization_state["warm_start_checkpoint"]
    if warm_start_checkpoint is None:
        return {
            "initialization_mode": str(initialization_state["initialization_mode"]),
            "warm_start_mode": str(initialization_state["warm_start_mode"]),
            "warm_start_checkpoint": "",
        }

    checkpoint_payload = torch.load(warm_start_checkpoint, map_location="cpu", weights_only=False)
    model_state_dict = checkpoint_payload.get("model_state_dict")
    if not isinstance(model_state_dict, Mapping):
        raise KeyError(
            f"Warm-start checkpoint at {warm_start_checkpoint} does not contain `model_state_dict`."
        )
    model.load_state_dict(model_state_dict, strict=bool(initialization_state["strict"]))
    print(
        "Warm-start initialization: "
        f"mode={initialization_state['warm_start_mode']} "
        f"checkpoint={warm_start_checkpoint} "
        f"strict={bool(initialization_state['strict'])} "
        f"source_monitor={checkpoint_payload.get('monitor_metric')} "
        f"source_ddi_type={checkpoint_payload.get('ddi_type')} "
        f"source_ddi_research_grade={checkpoint_payload.get('ddi_research_grade')}"
    )
    return {
        "initialization_mode": str(initialization_state["initialization_mode"]),
        "warm_start_mode": str(initialization_state["warm_start_mode"]),
        "warm_start_checkpoint": str(warm_start_checkpoint),
    }


def resolve_train_budget_label(train_config: Mapping[str, Any]) -> str:
    runtime_cfg = dict(train_config.get("runtime", {}))
    explicit_label = str(runtime_cfg.get("train_budget_label") or "").strip()
    if explicit_label:
        return explicit_label

    optimization_cfg = dict(train_config.get("optimization", {}))
    epochs = int(optimization_cfg.get("epochs", 0))
    profile_steps = runtime_cfg.get("profile_steps")
    if profile_steps is None:
        return f"full_data_epochs_{epochs}"
    return f"profile_steps_{int(profile_steps)}_epochs_{epochs}"


def resolve_core_monitor_config(
    train_config: Mapping[str, Any],
    threshold_tuning_cfg: Mapping[str, Any],
) -> tuple[str, str]:
    optimization_cfg = dict(train_config.get("optimization", {}))
    default_monitor_metric = "val_f1_tuned" if bool(threshold_tuning_cfg["enabled"]) else "val_total_loss"
    monitor_metric = str(optimization_cfg.get("monitor_metric") or default_monitor_metric)
    default_monitor_mode = "min" if "total_loss" in monitor_metric else "max"
    monitor_mode = str(optimization_cfg.get("monitor_mode") or default_monitor_mode)
    return monitor_metric, monitor_mode


def build_loss_objective_metadata(loss_fn: MedicationRecommendationLoss) -> dict[str, Any]:
    return {
        "objective": str(getattr(loss_fn, "objective", "bce")),
        "focal_gamma": float(getattr(loss_fn, "focal_gamma", 1.5)),
        "asymmetric_gamma_pos": float(getattr(loss_fn, "asymmetric_gamma_pos", 0.0)),
        "asymmetric_gamma_neg": float(getattr(loss_fn, "asymmetric_gamma_neg", 4.0)),
        "asymmetric_clip": float(getattr(loss_fn, "asymmetric_clip", 0.05)),
    }


def build_optimizer(*, model: torch.nn.Module, train_config: Mapping[str, Any]) -> torch.optim.Optimizer:
    optimization_cfg = dict(train_config.get("optimization", {}))
    optimizer_name = str(optimization_cfg.get("optimizer", "adam")).strip().lower()
    if optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer `{optimizer_name}`. Only `adam` is supported in train_core.py.")
    learning_rate = float(optimization_cfg.get("learning_rate", 1.0e-3))
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    train_config: Mapping[str, Any],
    monitor_mode: str = "min",
) -> Any | None:
    scheduler_name = str(train_config.get("optimization", {}).get("scheduler", "none")).strip().lower()
    if scheduler_name == "none":
        return None
    if scheduler_name != "reduce_on_plateau":
        raise ValueError(
            f"Unsupported scheduler `{scheduler_name}`. "
            "Expected one of ['none', 'reduce_on_plateau']."
        )

    optimization_cfg = dict(train_config.get("optimization", {}))
    factor = float(optimization_cfg.get("scheduler_factor", 0.5))
    patience = int(optimization_cfg.get("scheduler_patience", 1))
    min_lr = float(optimization_cfg.get("scheduler_min_lr", 1.0e-6))
    if not 0.0 < factor < 1.0:
        raise ValueError(f"optimization.scheduler_factor must be within (0, 1), got {factor!r}")
    if patience < 0:
        raise ValueError(f"optimization.scheduler_patience must be non-negative, got {patience!r}")
    if min_lr < 0.0:
        raise ValueError(f"optimization.scheduler_min_lr must be non-negative, got {min_lr!r}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(monitor_mode),
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )


def build_core_memory_bank(
    *,
    model: RetrievalEvidenceFusionModel,
    dataloader: DataLoader,
    device: torch.device,
    split: str,
) -> MemoryBank:
    banks: list[MemoryBank] = []
    was_training = model.training
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            records = batch.get("records")
            if not isinstance(records, Sequence):
                raise RuntimeError("Retrieval memory bank construction requires dataloader batches with records.")
            batch_on_device = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            encoder_outputs = model.encoder(dict(batch_on_device))
            banks.append(MemoryBank.build_from_batch(records, encoder_outputs, split=split))
    model.train(was_training)
    if not banks:
        raise ValueError("Cannot build retrieval memory bank from an empty dataloader")

    payloads = [bank.to_payload() for bank in banks]
    tensor_fields = (
        "visit_states",
        "visit_repr",
        "subject_ids",
        "hadm_ids",
        "stay_ids",
        "visit_index",
        "visit_time_days",
        "num_steps",
    )
    list_fields = (
        "visit_time_text",
        "target_drugs",
        "diag_code_sets",
        "proc_code_sets",
        "lab_feature_sets",
        "vital_feature_sets",
    )
    merged: dict[str, Any] = {
        field: torch.cat([payload[field] for payload in payloads], dim=0)
        for field in tensor_fields
    }
    for field in list_fields:
        values: list[Any] = []
        for payload in payloads:
            values.extend(list(payload[field]))
        merged[field] = values
    merged["split"] = split
    return MemoryBank.from_payload(merged)


def main() -> None:
    args = parse_args()
    raw_train_config = load_yaml_config(args.config)
    raw_data_config = load_yaml_config(args.data_config)
    raw_model_config = load_yaml_config(args.model_config)
    profile_name = resolve_profile_name(raw_train_config, args.profile)
    train_config = apply_profile_overrides(raw_train_config, profile_name=profile_name)
    data_config = apply_profile_overrides(raw_data_config, profile_name=profile_name)
    model_config = apply_profile_overrides(raw_model_config, profile_name=profile_name)
    model_overrides = train_config.get("model_overrides")
    if isinstance(model_overrides, Mapping):
        model_config = _deep_merge(model_config, dict(model_overrides))

    runtime_cfg = dict(train_config.get("runtime", {}))
    validate_core_runtime_config(
        runtime_cfg=runtime_cfg,
        core_cfg=dict(train_config.get("core", {})),
        context_label="train_core.py",
    )
    project_root = Path(train_config["_project_root"]).resolve()
    resolved_paths = resolve_runtime_paths(
        project_root=project_root,
        train_config=train_config,
        data_config=data_config,
        args=args,
    )
    train_config["_resolved_paths"] = {key: str(value) for key, value in resolved_paths.items()}

    device = resolve_device(args.device or runtime_cfg.get("device", "cpu"))
    seed = int(args.seed if args.seed is not None else data_config.get("seed", 17))
    num_workers = int(runtime_cfg.get("num_workers", 0))
    pin_memory = bool(runtime_cfg.get("pin_memory", device.type == "cuda"))
    requested_amp = bool(runtime_cfg.get("amp", False))
    persistent_workers = bool(runtime_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = runtime_cfg.get("prefetch_factor")
    length_bucket_window = int(runtime_cfg.get("length_bucket_window", 256))
    train_decoder_top_k = int(runtime_cfg.get("train_decoder_top_k", 0))
    matmul_precision = runtime_cfg.get("matmul_precision")
    max_grad_norm = float(train_config.get("optimization", {}).get("max_grad_norm", 1.0))
    feature_cfg = dict(data_config.get("features", {}))
    spark_cfg = dict(data_config.get("spark", {}))
    max_open_shards = int(spark_cfg.get("max_open_shards_per_dataset", 2))
    max_visits = feature_cfg.get("max_visits")
    max_history = feature_cfg.get("max_history")
    core_cfg = dict(train_config.get("core", {}))
    retrieval_enabled = bool(core_cfg.get("use_retrieval", False))
    if matmul_precision and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(matmul_precision))
    set_seed(seed)

    print(f"Using device: {device}")
    print(f"Selected profile: {profile_name}")
    print(f"Using seed: {seed}")
    print(
        "DataLoader settings: "
        f"batch_size={int(runtime_cfg.get('batch_size', 16))} "
        f"num_workers={num_workers} "
        f"pin_memory={pin_memory} "
        f"persistent_workers={persistent_workers if num_workers > 0 else False} "
        f"prefetch_factor={prefetch_factor if num_workers > 0 else None} "
        f"length_bucket_window={length_bucket_window}"
    )
    print(
        "Core runtime settings: "
        f"requested_amp={requested_amp} "
        f"grad_accum_steps={int(runtime_cfg.get('grad_accum_steps', 1))} "
        f"non_blocking_transfer={bool(runtime_cfg.get('non_blocking_transfer', False))} "
        f"train_decoder_top_k={train_decoder_top_k} "
        f"profile_steps={runtime_cfg.get('profile_steps')} "
        f"matmul_precision={matmul_precision} "
        f"max_grad_norm={max_grad_norm}"
    )
    print(
        "Core data view: "
        f"max_open_shards={max_open_shards} "
        f"max_visits={max_visits} "
        f"max_history={max_history}"
    )

    with tempfile.TemporaryDirectory(prefix="clinrec_runtime_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        runtime_data_config_path = build_runtime_data_config_file(
            data_config=data_config,
            processed_root=resolved_paths["processed_root"],
            vocab_root=resolved_paths["vocab_root"],
            temp_dir=temp_dir,
        )

        _ = load_vocab_bundle(runtime_data_config_path)
        drug_vocab_size = int(read_json(resolved_paths["vocab_root"] / "drug_vocab.json")["size"])

        train_loader, val_loader = build_dataloaders(
            runtime_data_config_path=runtime_data_config_path,
            processed_root=resolved_paths["processed_root"],
            drug_vocab_size=drug_vocab_size,
            batch_size=int(runtime_cfg.get("batch_size", 16)),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            length_bucket_window=length_bucket_window,
            seed=seed,
            max_open_shards=max_open_shards,
            max_visits=max_visits,
            max_history=max_history,
            include_records=retrieval_enabled,
        )
        loss_cfg = dict(train_config.get("loss", {}))
        pos_weight, pos_weight_stats = build_positive_class_weight(
            dataset=train_loader.dataset,
            drug_vocab_size=drug_vocab_size,
            mode=str(loss_cfg.get("pos_weight_mode", "disabled")),
            clip=float(loss_cfg.get("pos_weight_clip", 1.0)),
        )

        model, loss_fn = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=resolved_paths["vocab_root"],
            ddi_matrix_path=resolved_paths["ddi_matrix_path"],
            pos_weight=pos_weight,
        )
        initialization_context = apply_model_initialization(
            model=model,
            train_config=train_config,
        )
        retrieval_memory_bank = None
        if retrieval_enabled:
            retrieval_memory_bank = build_core_memory_bank(
                model=model,
                dataloader=train_loader,
                device=device,
                split="train",
            )
            print(
                "Core retrieval memory bank: "
                f"split={retrieval_memory_bank.split} "
                f"states={len(retrieval_memory_bank)} "
                f"top_k={model.retrieval_top_k} "
                f"cross_split_policy={model.cross_split_policy or ('allow_all' if model.allow_cross_split else 'same_split')} "
                f"leakage_safe={model.retrieval_leakage_safe}"
            )
        dataset_layouts = {
            "train": getattr(train_loader.dataset, "layout_kind", "unknown"),
            "val": getattr(val_loader.dataset, "layout_kind", "unknown"),
        }

    print(
        "DDI runtime state: "
        f"status={loss_fn.ddi_context.get('status', 'unknown')} "
        f"reason={loss_fn.ddi_context.get('reason', '')} "
        f"source={loss_fn.ddi_context.get('source', '')} "
        f"matched_pairs={loss_fn.ddi_context.get('matched_pairs')} "
        f"nonzero_pairs={loss_fn.ddi_context.get('nonzero_pairs')} "
        f"kind={dict(loss_fn.ddi_context.get('source_metadata') or {}).get('kind', '')} "
        f"research_grade={dict(loss_fn.ddi_context.get('source_metadata') or {}).get('research_grade')} "
        f"purpose={dict(loss_fn.ddi_context.get('source_metadata') or {}).get('purpose', '')} "
        f"configured_ddi_lambda={loss_fn.configured_lambda_ddi:.6f} "
        f"effective_ddi_lambda={loss_fn.effective_lambda_ddi:.6f}"
    )
    if not loss_fn.ddi_active:
        print("DDI regularization is explicitly disabled for this run because the DDI artifact is inactive.")

    runtime_truth = copy.deepcopy(getattr(model, "runtime_truth", {}))
    threshold_tuning_cfg = normalize_threshold_tuning_config(train_config.get("threshold_tuning"))
    default_threshold = float(train_config.get("prediction", {}).get("threshold", 0.5))
    effective_threshold = default_threshold
    train_budget_label = resolve_train_budget_label(train_config)
    monitor_metric, monitor_mode = resolve_core_monitor_config(train_config, threshold_tuning_cfg)
    loss_objective_metadata = build_loss_objective_metadata(loss_fn)
    contrastive_lambda = float(train_config.get("loss", {}).get("contrastive_lambda", 0.0))
    contrastive_temperature = float(train_config.get("loss", {}).get("contrastive_temperature", 0.07))
    print(
        "Core runtime truth: "
        f"pipeline_level={runtime_truth.get('pipeline_level', 'unknown')} "
        f"history_active={runtime_truth.get('history_active')} "
        f"retrieval_active={runtime_truth.get('retrieval_active')} "
        f"fusion_strategy={runtime_truth.get('fusion_strategy', 'unknown')} "
        f"ddi_type={runtime_truth.get('ddi_type', 'unknown')} "
        f"ddi_research_grade={runtime_truth.get('ddi_research_grade')}"
    )
    print(
        "Loss imbalance settings: "
        f"objective={loss_objective_metadata['objective']} "
        f"pos_weight_mode={pos_weight_stats['mode']} "
        f"pos_weight_clip={pos_weight_stats['clip']:.2f} "
        f"labels_with_positive={pos_weight_stats['num_labels_with_positive']} "
        f"mean_weight={pos_weight_stats.get('mean_weight', 1.0):.4f} "
        f"max_weight={pos_weight_stats.get('max_weight', 1.0):.4f} "
        f"asymmetric_gamma_pos={loss_objective_metadata['asymmetric_gamma_pos']:.2f} "
        f"asymmetric_gamma_neg={loss_objective_metadata['asymmetric_gamma_neg']:.2f} "
        f"asymmetric_clip={loss_objective_metadata['asymmetric_clip']:.3f}"
    )
    print(
        "Contrastive settings: "
        f"enabled={contrastive_lambda > 0.0} "
        f"lambda={contrastive_lambda:.6f} "
        f"temperature={contrastive_temperature:.4f}"
    )
    print(
        "Threshold tuning settings: "
        f"enabled={threshold_tuning_cfg['enabled']} "
        f"split={threshold_tuning_cfg['split']} "
        f"metric={threshold_tuning_cfg['metric']} "
        f"tie_breaker={threshold_tuning_cfg['tie_breaker']} "
        f"candidates={threshold_tuning_cfg['candidates']}"
    )
    print(
        "Initialization settings: "
        f"initialization_mode={initialization_context['initialization_mode']} "
        f"warm_start_mode={initialization_context['warm_start_mode']} "
        f"warm_start_checkpoint={initialization_context['warm_start_checkpoint'] or '<none>'} "
        f"train_budget_label={train_budget_label}"
    )
    print(
        "Optimization monitor: "
        f"monitor_metric={monitor_metric} "
        f"monitor_mode={monitor_mode} "
        f"scheduler={str(train_config.get('optimization', {}).get('scheduler', 'none'))} "
        f"early_stopping_patience={train_config.get('optimization', {}).get('early_stopping_patience')}"
    )

    optimizer = build_optimizer(model=model, train_config=train_config)
    scheduler = build_scheduler(
        optimizer=optimizer,
        train_config=train_config,
        monitor_mode=monitor_mode,
    )
    trainer = TqdmCoreTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=resolved_paths["checkpoint_dir"],
        log_dir=resolved_paths["log_dir"],
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
        decoder_top_k=train_decoder_top_k,
        amp=requested_amp,
        grad_accum_steps=int(runtime_cfg.get("grad_accum_steps", 1)),
        max_grad_norm=max_grad_norm,
        non_blocking_transfer=bool(runtime_cfg.get("non_blocking_transfer", False)),
        log_interval=int(runtime_cfg.get("log_interval", 50)),
        profile_steps=runtime_cfg.get("profile_steps"),
        early_stopping_patience=train_config.get("optimization", {}).get("early_stopping_patience"),
        detailed_timing=bool(runtime_cfg.get("detailed_timing", False)),
        retrieval_memory_bank=retrieval_memory_bank,
        retrieval_enabled=retrieval_enabled,
        contrastive_lambda=contrastive_lambda,
        contrastive_temperature=contrastive_temperature,
        run_context={
            **runtime_truth,
            **initialization_context,
            "selected_profile": profile_name,
            "train_budget_label": train_budget_label,
            "ddi_context": copy.deepcopy(loss_fn.ddi_context),
            "configured_ddi_lambda": float(loss_fn.configured_lambda_ddi),
            "effective_ddi_lambda": float(loss_fn.effective_lambda_ddi),
            "effective_threshold": effective_threshold,
            "threshold_selection": {
                "source": "config.prediction.threshold",
                "split": "config",
                "metric": str(threshold_tuning_cfg["metric"]),
                "tie_breaker": str(threshold_tuning_cfg["tie_breaker"]),
                "candidates": [float(value) for value in threshold_tuning_cfg["candidates"]],
                "best_threshold": effective_threshold,
            },
            "threshold_tuning": copy.deepcopy(threshold_tuning_cfg),
            "loss_objective": str(loss_objective_metadata["objective"]),
            "objective_settings": copy.deepcopy(loss_objective_metadata),
            "pos_weight_stats": copy.deepcopy(pos_weight_stats),
            "ranking_loss": {
                "lambda": float(getattr(loss_fn, "ranking_lambda", 0.0)),
                "objective": str(getattr(loss_fn, "ranking_objective", "bpr")),
                "num_negatives": int(getattr(loss_fn, "ranking_num_negatives", 0)),
                "margin": float(getattr(loss_fn, "ranking_margin", 0.0)),
                "hard_negative_fraction": float(
                    getattr(loss_fn, "ranking_hard_negative_fraction", 0.0)
                ),
            },
            "contrastive_loss": {
                "lambda": contrastive_lambda,
                "temperature": contrastive_temperature,
                "enabled": contrastive_lambda > 0.0,
            },
            "dataset_layouts": dataset_layouts,
            "runtime": {
                "batch_size": int(runtime_cfg.get("batch_size", 16)),
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "persistent_workers": persistent_workers if num_workers > 0 else False,
                "prefetch_factor": None if num_workers <= 0 else prefetch_factor,
                "amp": requested_amp,
                "requested_amp": requested_amp,
                "grad_accum_steps": int(runtime_cfg.get("grad_accum_steps", 1)),
                "non_blocking_transfer": bool(runtime_cfg.get("non_blocking_transfer", False)),
                "log_interval": int(runtime_cfg.get("log_interval", 50)),
                "profile_steps": runtime_cfg.get("profile_steps"),
                "train_decoder_top_k": train_decoder_top_k,
                "matmul_precision": matmul_precision,
                "length_bucket_window": length_bucket_window,
                "max_grad_norm": max_grad_norm,
            },
        },
    )

    print(
        "Trainer precision settings: "
        f"requested_amp={trainer.requested_amp} "
        f"resolved_precision={trainer.resolved_precision} "
        f"grad_scaler_enabled={trainer.grad_scaler_enabled} "
        f"max_grad_norm={trainer.max_grad_norm}"
    )

    fit_result = trainer.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=int(train_config.get("optimization", {}).get("epochs", 10)),
        extra_checkpoint_state={
            **runtime_truth,
            **initialization_context,
            "train_config": {key: value for key, value in train_config.items() if not str(key).startswith("_")},
            "data_config": {key: value for key, value in data_config.items() if not str(key).startswith("_")},
            "model_config": {key: value for key, value in model_config.items() if not str(key).startswith("_")},
            "resolved_paths": {key: str(value) for key, value in resolved_paths.items()},
            "selected_profile": profile_name,
            "seed": seed,
            "train_mode": "core",
            "train_budget_label": train_budget_label,
            "ddi_context": copy.deepcopy(loss_fn.ddi_context),
            "configured_ddi_lambda": float(loss_fn.configured_lambda_ddi),
            "effective_ddi_lambda": float(loss_fn.effective_lambda_ddi),
            "threshold_tuning": copy.deepcopy(threshold_tuning_cfg),
            "loss_objective": str(loss_objective_metadata["objective"]),
            "objective_settings": copy.deepcopy(loss_objective_metadata),
            "pos_weight_stats": copy.deepcopy(pos_weight_stats),
            "ranking_loss": {
                "lambda": float(getattr(loss_fn, "ranking_lambda", 0.0)),
                "objective": str(getattr(loss_fn, "ranking_objective", "bpr")),
                "num_negatives": int(getattr(loss_fn, "ranking_num_negatives", 0)),
                "margin": float(getattr(loss_fn, "ranking_margin", 0.0)),
                "hard_negative_fraction": float(
                    getattr(loss_fn, "ranking_hard_negative_fraction", 0.0)
                ),
            },
            "contrastive_loss": {
                "lambda": contrastive_lambda,
                "temperature": contrastive_temperature,
                "enabled": contrastive_lambda > 0.0,
            },
            "dataset_layouts": dataset_layouts,
            "trainer_runtime": {
                "requested_amp": trainer.requested_amp,
                "resolved_precision": trainer.resolved_precision,
                "grad_scaler_enabled": trainer.grad_scaler_enabled,
                "max_grad_norm": trainer.max_grad_norm,
                "monitor_metric": monitor_metric,
                "monitor_mode": monitor_mode,
                "scheduler": str(train_config.get("optimization", {}).get("scheduler", "none")),
                "early_stopping_patience": train_config.get("optimization", {}).get("early_stopping_patience"),
            },
        },
    )

    print(f"Best checkpoint: {fit_result['best_checkpoint_path']}")
    print(f"Monitor metric: {fit_result['monitor_metric']} (best={fit_result['best_metric']:.6f})")
    print(
        "Fit status: "
        f"epochs_completed={fit_result['epochs_completed']} "
        f"stopped_early={fit_result['stopped_early']} "
        f"stop_reason={fit_result['stop_reason'] or '<none>'}"
    )


if __name__ == "__main__":
    main()
