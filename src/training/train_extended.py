from __future__ import annotations

import argparse
import copy
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from src.data.build_vocab import load_vocab_bundle
from src.data.dataset import (
    ShardLengthBatchSampler,
    build_collate_fn,
    collate_batch_with_records as dataset_collate_batch_with_records,
)
from src.evaluation.thresholding import normalize_threshold_tuning_config
from src.models.full_model import RetrievalEvidenceFusionModel
from src.retrieval.memory_bank import MemoryBank
from src.training.trainer import _move_batch_to_device
from src.training.train_core import (
    apply_profile_overrides,
    apply_model_initialization,
    build_positive_class_weight,
    build_core_model,
    build_dataset,
    build_optimizer,
    build_runtime_data_config_file,
    build_scheduler,
    resolve_train_budget_label,
    resolve_profile_name,
    resolve_device,
    resolve_runtime_paths,
    set_seed,
    TqdmCoreTrainer,
)
from src.utils.io import load_yaml_config, read_json
from src.utils.runtime_truth import build_extension_runtime_truth


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the extended ClinRec recommendation model.")
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


def collate_batch_with_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return dataset_collate_batch_with_records(
        records,
        include_full_targets=False,
        include_final_target=True,
    )


def build_extended_dataloaders(
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
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
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
        include_records=True,
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

    train_batch_sampler = None
    if getattr(train_dataset, "shard_row_indices", None):
        train_batch_sampler = ShardLengthBatchSampler(
            train_dataset,
            batch_size=int(batch_size),
            length_bucket_window=int(length_bucket_window),
            shuffle=True,
            seed=int(seed),
        )

    train_loader = (
        DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            **loader_kwargs,
        )
        if train_batch_sampler is not None
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
    train_bank_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        **loader_kwargs,
    )
    val_bank_loader = DataLoader(
        val_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, train_bank_loader, val_bank_loader


def build_optional_group_encoder(
    *,
    train_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    hidden_dim: int,
) -> Any | None:
    extended_cfg = dict(train_config.get("extended", {}))
    if not bool(extended_cfg.get("use_group_encoder", True)):
        print("Extended mode: group encoder disabled by config.")
        return None

    try:
        from src.graph.group_encoder import GroupEncoder
    except Exception as exc:
        print(f"Warning: failed to import GroupEncoder; continuing without it. Reason: {exc}")
        return None

    hypergraph_cfg = dict(model_config.get("hypergraph", {}))
    try:
        return GroupEncoder(
            hidden_dim=int(hidden_dim),
            num_layers=int(hypergraph_cfg.get("num_layers", 2)),
            dropout=float(hypergraph_cfg.get("dropout", 0.1)),
            num_group_prototypes=int(hypergraph_cfg.get("num_group_prototypes", 8)),
            use_semantic_edges=bool(hypergraph_cfg.get("use_semantic_edges", True)),
            use_weighted_edges=bool(hypergraph_cfg.get("use_weighted_edges", True)),
            prototype_top_k=int(hypergraph_cfg.get("prototype_top_k", 2)),
            include_time_edges=bool(hypergraph_cfg.get("include_time_edges", True)),
            include_prototype_edges=bool(hypergraph_cfg.get("include_prototype_edges", True)),
        )
    except Exception as exc:
        print(f"Warning: failed to build GroupEncoder; continuing without it. Reason: {exc}")
        return None


def build_extended_model(
    *,
    train_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    runtime_data_config_path: Path,
    vocab_root: Path,
    ddi_matrix_path: Path,
    pos_weight: torch.Tensor | None = None,
) -> tuple[RetrievalEvidenceFusionModel, Any]:
    core_compatible_train_config = copy.deepcopy(dict(train_config))
    core_compatible_train_config.setdefault("runtime", {})
    core_compatible_train_config["runtime"]["mode"] = "core"
    model, loss_fn = build_core_model(
        train_config=core_compatible_train_config,
        model_config=model_config,
        runtime_data_config_path=runtime_data_config_path,
        vocab_root=vocab_root,
        ddi_matrix_path=ddi_matrix_path,
        pos_weight=pos_weight,
    )

    extended_cfg = dict(train_config.get("extended", {}))
    retrieval_cfg = dict(model_config.get("retrieval", {}))
    hidden_dim = int(model.fusion_module.hidden_dim)

    model.group_encoder = build_optional_group_encoder(
        train_config=train_config,
        model_config=model_config,
        hidden_dim=hidden_dim,
    )
    model.mode = str(extended_cfg.get("mode", "extended"))
    model.retrieval_top_k = int(extended_cfg.get("retrieval_top_k", retrieval_cfg.get("top_k", 5)))
    model.temporal_decay_alpha = float(
        extended_cfg.get("temporal_decay_alpha", retrieval_cfg.get("temporal_decay_alpha", 0.05))
    )
    model.retrieval_backend = str(extended_cfg.get("retrieval_backend", retrieval_cfg.get("backend", "bruteforce")))
    model.use_faiss_if_available = bool(
        extended_cfg.get(
            "use_faiss_if_available",
            retrieval_cfg.get("use_faiss_if_available", True),
        )
    )
    model.allow_cross_split = bool(extended_cfg.get("allow_cross_split", False))
    model.retrieval_scoring_mode = str(
        extended_cfg.get("retrieval_scoring_mode", retrieval_cfg.get("scoring_mode", "temporal_relevance"))
    )
    model.cross_split_policy = str(
        extended_cfg.get(
            "cross_split_policy",
            retrieval_cfg.get("cross_split_policy", "train_bank_only"),
        )
    )
    validation_memory_bank_policy = str(extended_cfg.get("validation_memory_bank_policy", "train_only"))
    model.runtime_truth = build_extension_runtime_truth(
        fusion_strategy=str(model.fusion_module.strategy),
        ddi_context=loss_fn.ddi_context,
        retrieval_active=bool(extended_cfg.get("use_retrieval", True)),
        group_encoder_active=model.group_encoder is not None,
        retrieval_scoring_mode=model.retrieval_scoring_mode,
        retrieval_cross_split_policy=model.cross_split_policy
        or ("allow_all" if model.allow_cross_split else "same_split"),
        retrieval_bank_policy=validation_memory_bank_policy,
    )
    return model, loss_fn


def _merge_memory_banks(banks: Sequence[MemoryBank], *, split: str) -> MemoryBank:
    nonempty_banks = [bank for bank in banks if len(bank) > 0]
    if not nonempty_banks:
        raise ValueError(f"No memory bank rows were collected for split `{split}`")

    return MemoryBank(
        visit_states=torch.cat([bank.visit_states for bank in nonempty_banks], dim=0),
        visit_repr=torch.cat([bank.visit_repr for bank in nonempty_banks], dim=0),
        subject_ids=torch.cat([bank.subject_ids for bank in nonempty_banks], dim=0),
        hadm_ids=torch.cat([bank.hadm_ids for bank in nonempty_banks], dim=0),
        stay_ids=torch.cat([bank.stay_ids for bank in nonempty_banks], dim=0),
        visit_index=torch.cat([bank.visit_index for bank in nonempty_banks], dim=0),
        visit_time_days=torch.cat([bank.visit_time_days for bank in nonempty_banks], dim=0),
        visit_time_text=[item for bank in nonempty_banks for item in bank.visit_time_text],
        target_drugs=[item for bank in nonempty_banks for item in bank.target_drugs],
        num_steps=torch.cat([bank.num_steps for bank in nonempty_banks], dim=0),
        diag_code_sets=[item for bank in nonempty_banks for item in bank.diag_code_sets],
        proc_code_sets=[item for bank in nonempty_banks for item in bank.proc_code_sets],
        lab_feature_sets=[item for bank in nonempty_banks for item in bank.lab_feature_sets],
        vital_feature_sets=[item for bank in nonempty_banks for item in bank.vital_feature_sets],
        split=split,
    )


def build_memory_bank_from_dataloader(
    *,
    model: RetrievalEvidenceFusionModel,
    dataloader: DataLoader,
    device: torch.device,
    split: str,
) -> MemoryBank:
    was_training = model.training
    banks: list[MemoryBank] = []
    model.eval()
    try:
        with torch.no_grad():
            for batch in dataloader:
                records = list(batch.get("records", []))
                if not records:
                    continue
                batch_on_device = _move_batch_to_device(batch, device)
                encoder_outputs = model.encoder(dict(batch_on_device))
                banks.append(MemoryBank.build_from_batch(records, encoder_outputs, split=split))
    finally:
        model.train(was_training)
    return _merge_memory_banks(banks, split=split)


class ExtendedTrainer(TqdmCoreTrainer):
    def __init__(
        self,
        *,
        use_retrieval: bool,
        validation_memory_bank_policy: str = "train_only",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.use_retrieval = bool(use_retrieval)
        self.validation_memory_bank_policy = str(validation_memory_bank_policy).strip().lower()
        if self.validation_memory_bank_policy not in {"train_only", "same_split"}:
            raise ValueError(
                "validation_memory_bank_policy must be one of ['train_only', 'same_split'], "
                f"got {validation_memory_bank_policy!r}"
            )
        self.best_checkpoint_path = self.checkpoint_dir / "train_extended_best.pt"
        self.metrics_log_path = self.log_dir / "train_extended_metrics.jsonl"
        self.train_memory_bank: MemoryBank | None = None
        self.val_memory_bank: MemoryBank | None = None
        self._active_memory_bank: MemoryBank | None = None
        self._last_memory_bank_refresh_time = 0.0

    def _forward_model(self, batch_on_device: Mapping[str, Any]) -> dict[str, Any]:
        return self.model(
            batch_on_device,
            mode="extended",
            memory_bank=self._active_memory_bank if self.use_retrieval else None,
            records=batch_on_device.get("records"),
            decoder_top_k=self.decoder_top_k,
        )

    def _validation_memory_bank(self) -> MemoryBank | None:
        if not self.use_retrieval:
            return None
        if self.validation_memory_bank_policy == "train_only":
            return self.train_memory_bank
        return self.val_memory_bank

    def _run_one_epoch(self, dataloader: DataLoader, *, training: bool) -> dict[str, float]:
        self._active_memory_bank = self.train_memory_bank if training else self._validation_memory_bank()
        return super()._run_one_epoch(dataloader, training=training)

    def _collect_prediction_payload(self, dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        cached_payload = self._cached_validation_prediction_payload_for(dataloader)
        if cached_payload is not None:
            return cached_payload

        previous_memory_bank = self._active_memory_bank
        self._active_memory_bank = self._validation_memory_bank()
        try:
            return super()._collect_prediction_payload(dataloader)
        finally:
            self._active_memory_bank = previous_memory_bank

    def _epoch_aux_timing_metrics(self) -> dict[str, float]:
        metrics = super()._epoch_aux_timing_metrics()
        if self.detailed_timing_enabled:
            metrics["memory_bank_refresh_time"] = float(self._last_memory_bank_refresh_time)
        return metrics

    def refresh_memory_banks(
        self,
        *,
        train_bank_dataloader: DataLoader | None,
        val_bank_dataloader: DataLoader | None,
    ) -> None:
        refresh_start = time.perf_counter()
        if not self.use_retrieval:
            self.train_memory_bank = None
            self.val_memory_bank = None
            self._last_memory_bank_refresh_time = time.perf_counter() - refresh_start
            print("Extended mode retrieval disabled by config; skipping memory bank refresh.")
            return
        if train_bank_dataloader is None:
            raise ValueError("Extended retrieval requires a train memory-bank dataloader.")

        try:
            self.train_memory_bank = build_memory_bank_from_dataloader(
                model=self.model,
                dataloader=train_bank_dataloader,
                device=self.device,
                split="train",
            )
            print(f"Built train memory bank with {len(self.train_memory_bank)} visits.")
        except Exception as exc:
            raise RuntimeError(f"Extended retrieval failed while building the train memory bank: {exc}") from exc

        if self.validation_memory_bank_policy == "train_only":
            self.val_memory_bank = self.train_memory_bank
            self._last_memory_bank_refresh_time = time.perf_counter() - refresh_start
            print("Validation retrieval bank policy: train_only.")
            return

        if val_bank_dataloader is None:
            raise ValueError(
                "validation_memory_bank_policy=same_split requires a validation memory-bank dataloader."
            )
        try:
            self.val_memory_bank = build_memory_bank_from_dataloader(
                model=self.model,
                dataloader=val_bank_dataloader,
                device=self.device,
                split="val",
            )
            print(f"Built val memory bank with {len(self.val_memory_bank)} visits.")
        except Exception as exc:
            raise RuntimeError(f"Extended retrieval failed while building the val memory bank: {exc}") from exc
        self._last_memory_bank_refresh_time = time.perf_counter() - refresh_start

    def fit(
        self,
        *,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int,
        train_bank_dataloader: DataLoader | None = None,
        val_bank_dataloader: DataLoader | None = None,
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
            self.refresh_memory_banks(
                train_bank_dataloader=train_bank_dataloader,
                val_bank_dataloader=val_bank_dataloader,
            )

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

    project_root = Path(train_config["_project_root"]).resolve()
    resolved_paths = resolve_runtime_paths(
        project_root=project_root,
        train_config=train_config,
        data_config=data_config,
        args=args,
    )
    train_config["_resolved_paths"] = {key: str(value) for key, value in resolved_paths.items()}

    runtime_cfg = dict(train_config.get("runtime", {}))
    extended_cfg = dict(train_config.get("extended", {}))
    device = resolve_device(args.device or runtime_cfg.get("device", "cpu"))
    seed = int(args.seed if args.seed is not None else data_config.get("seed", 17))
    requested_amp = bool(runtime_cfg.get("amp", False))
    max_grad_norm = float(train_config.get("optimization", {}).get("max_grad_norm", 1.0))
    set_seed(seed)
    runtime_truth: dict[str, Any] | None = None
    initialization_context: dict[str, Any] | None = None
    loss_cfg = dict(train_config.get("loss", {}))
    threshold_tuning_cfg = normalize_threshold_tuning_config(train_config.get("threshold_tuning"))
    train_budget_label = resolve_train_budget_label(train_config)
    validation_memory_bank_policy = str(extended_cfg.get("validation_memory_bank_policy", "train_only"))
    feature_cfg = dict(data_config.get("features", {}))
    spark_cfg = dict(data_config.get("spark", {}))

    print(f"Using device: {device}")
    print(f"Selected profile: {profile_name}")
    print(f"Using seed: {seed}")
    print(f"Extended mode retrieval enabled: {bool(extended_cfg.get('use_retrieval', True))}")
    print(f"Extended mode group encoder enabled: {bool(extended_cfg.get('use_group_encoder', True))}")

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

        train_loader, val_loader, train_bank_loader, val_bank_loader = build_extended_dataloaders(
            runtime_data_config_path=runtime_data_config_path,
            processed_root=resolved_paths["processed_root"],
            drug_vocab_size=drug_vocab_size,
            batch_size=int(runtime_cfg.get("batch_size", 16)),
            num_workers=int(runtime_cfg.get("num_workers", 0)),
            pin_memory=bool(runtime_cfg.get("pin_memory", False)),
            persistent_workers=bool(runtime_cfg.get("persistent_workers", False)),
            prefetch_factor=runtime_cfg.get("prefetch_factor"),
            length_bucket_window=int(runtime_cfg.get("length_bucket_window", 256)),
            seed=seed,
            max_open_shards=spark_cfg.get("max_open_shards_per_dataset"),
            max_visits=feature_cfg.get("max_visits"),
            max_history=feature_cfg.get("max_history"),
        )
        pos_weight, pos_weight_stats = build_positive_class_weight(
            dataset=train_loader.dataset,
            drug_vocab_size=drug_vocab_size,
            mode=str(loss_cfg.get("pos_weight_mode", "disabled")),
            clip=float(loss_cfg.get("pos_weight_clip", 1.0)),
        )

        model, loss_fn = build_extended_model(
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
        runtime_truth = dict(getattr(model, "runtime_truth", {}))

    print(
        "Extended runtime truth: "
        f"pipeline_level={runtime_truth.get('pipeline_level', 'unknown') if runtime_truth else 'unknown'} "
        f"history_active={runtime_truth.get('history_active') if runtime_truth else None} "
        f"retrieval_active={runtime_truth.get('retrieval_active') if runtime_truth else None} "
        f"retrieval_mode={runtime_truth.get('retrieval_mode', 'unknown') if runtime_truth else 'unknown'} "
        f"retrieval_bank_policy={runtime_truth.get('retrieval_bank_policy', 'unknown') if runtime_truth else 'unknown'} "
        f"fusion_strategy={runtime_truth.get('fusion_strategy', 'unknown') if runtime_truth else 'unknown'} "
        f"extension_status={runtime_truth.get('extension_status', 'unknown') if runtime_truth else 'unknown'}"
    )
    print(
        "Loss imbalance settings: "
        f"objective={getattr(loss_fn, 'objective', 'unknown')} "
        f"pos_weight_mode={pos_weight_stats['mode']} "
        f"pos_weight_clip={pos_weight_stats['clip']:.2f} "
        f"labels_with_positive={pos_weight_stats['num_labels_with_positive']} "
        f"mean_weight={pos_weight_stats.get('mean_weight', 1.0):.4f} "
        f"max_weight={pos_weight_stats.get('max_weight', 1.0):.4f}"
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
        f"initialization_mode={initialization_context['initialization_mode'] if initialization_context else 'unknown'} "
        f"warm_start_mode={initialization_context['warm_start_mode'] if initialization_context else 'unknown'} "
        f"warm_start_checkpoint={initialization_context['warm_start_checkpoint'] if initialization_context else ''} "
        f"train_budget_label={train_budget_label}"
    )

    default_monitor_metric = "val_prauc_tuned" if bool(threshold_tuning_cfg["enabled"]) else "val_total_loss"
    monitor_metric = str(extended_cfg.get("monitor_metric", default_monitor_metric))
    if "total_loss" in monitor_metric:
        monitor_mode = str(extended_cfg.get("monitor_mode", "min"))
    else:
        monitor_mode = str(extended_cfg.get("monitor_mode", "max"))
    optimizer = build_optimizer(model=model, train_config=train_config)
    scheduler = build_scheduler(
        optimizer=optimizer,
        train_config=train_config,
        monitor_mode=monitor_mode,
    )
    trainer = ExtendedTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=resolved_paths["checkpoint_dir"],
        log_dir=resolved_paths["log_dir"],
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
        decoder_top_k=int(train_config.get("prediction", {}).get("top_k", 10)),
        amp=requested_amp,
        grad_accum_steps=int(runtime_cfg.get("grad_accum_steps", 1)),
        max_grad_norm=max_grad_norm,
        non_blocking_transfer=bool(runtime_cfg.get("non_blocking_transfer", False)),
        log_interval=int(runtime_cfg.get("log_interval", 50)),
        profile_steps=runtime_cfg.get("profile_steps"),
        early_stopping_patience=train_config.get("optimization", {}).get("early_stopping_patience"),
        detailed_timing=bool(runtime_cfg.get("detailed_timing", False)),
        run_context={
            **dict(runtime_truth or {}),
            "selected_profile": profile_name,
            "ddi_context": dict(loss_fn.ddi_context),
            "configured_ddi_lambda": float(loss_fn.configured_lambda_ddi),
            "effective_ddi_lambda": float(loss_fn.effective_lambda_ddi),
            "threshold_tuning": dict(threshold_tuning_cfg),
            "loss_objective": str(getattr(loss_fn, "objective", "bce")),
            "fusion_entropy_lambda": float(getattr(loss_fn, "fusion_entropy_lambda", 0.0)),
            "fusion_balance_lambda": float(getattr(loss_fn, "fusion_balance_lambda", 0.0)),
            "ranking_loss": {
                "lambda": float(getattr(loss_fn, "ranking_lambda", 0.0)),
                "objective": str(getattr(loss_fn, "ranking_objective", "bpr")),
                "num_negatives": int(getattr(loss_fn, "ranking_num_negatives", 0)),
                "margin": float(getattr(loss_fn, "ranking_margin", 0.0)),
                "hard_negative_fraction": float(
                    getattr(loss_fn, "ranking_hard_negative_fraction", 0.0)
                ),
            },
            "pos_weight_stats": dict(pos_weight_stats),
            "validation_memory_bank_policy": validation_memory_bank_policy,
            **dict(initialization_context or {}),
            "runtime": {
                "amp": requested_amp,
                "requested_amp": requested_amp,
                "max_grad_norm": max_grad_norm,
            },
        },
        use_retrieval=bool(extended_cfg.get("use_retrieval", True)),
        validation_memory_bank_policy=validation_memory_bank_policy,
    )

    print(
        "Trainer precision settings: "
        f"requested_amp={trainer.requested_amp} "
        f"resolved_precision={trainer.resolved_precision} "
        f"grad_scaler_enabled={trainer.grad_scaler_enabled} "
        f"max_grad_norm={trainer.max_grad_norm}"
    )
    print(
        "Optimization monitor: "
        f"monitor_metric={monitor_metric} "
        f"monitor_mode={monitor_mode} "
        f"scheduler={str(train_config.get('optimization', {}).get('scheduler', 'none'))} "
        f"early_stopping_patience={train_config.get('optimization', {}).get('early_stopping_patience')} "
        f"validation_memory_bank_policy={validation_memory_bank_policy}"
    )

    fit_result = trainer.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        train_bank_dataloader=train_bank_loader if bool(extended_cfg.get("use_retrieval", True)) else None,
        val_bank_dataloader=val_bank_loader
        if bool(extended_cfg.get("use_retrieval", True)) and validation_memory_bank_policy == "same_split"
        else None,
        epochs=int(train_config.get("optimization", {}).get("epochs", 10)),
        extra_checkpoint_state={
            **dict(runtime_truth or {}),
            "train_config": {key: value for key, value in train_config.items() if not str(key).startswith("_")},
            "data_config": {key: value for key, value in data_config.items() if not str(key).startswith("_")},
            "model_config": {key: value for key, value in model_config.items() if not str(key).startswith("_")},
            "resolved_paths": {key: str(value) for key, value in resolved_paths.items()},
            "selected_profile": profile_name,
            "seed": seed,
            "train_mode": "extended",
            "ddi_context": dict(loss_fn.ddi_context),
            "configured_ddi_lambda": float(loss_fn.configured_lambda_ddi),
            "effective_ddi_lambda": float(loss_fn.effective_lambda_ddi),
            "group_encoder_active": model.group_encoder is not None,
            "threshold_tuning": dict(threshold_tuning_cfg),
            "loss_objective": str(getattr(loss_fn, "objective", "bce")),
            "ranking_loss": {
                "lambda": float(getattr(loss_fn, "ranking_lambda", 0.0)),
                "objective": str(getattr(loss_fn, "ranking_objective", "bpr")),
                "num_negatives": int(getattr(loss_fn, "ranking_num_negatives", 0)),
                "margin": float(getattr(loss_fn, "ranking_margin", 0.0)),
                "hard_negative_fraction": float(
                    getattr(loss_fn, "ranking_hard_negative_fraction", 0.0)
                ),
            },
            "pos_weight_stats": dict(pos_weight_stats),
            "validation_memory_bank_policy": validation_memory_bank_policy,
            "train_budget_label": train_budget_label,
            **dict(initialization_context or {}),
        },
    )

    print(f"Best checkpoint: {fit_result['best_checkpoint_path']}")
    print(f"Monitor metric: {fit_result['monitor_metric']} (best={fit_result['best_metric']:.6f})")


if __name__ == "__main__":
    main()
