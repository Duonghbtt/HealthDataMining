from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from src.data.build_vocab import load_vocab_bundle
from src.data.dataset import collate_batch
from src.models.full_model import RetrievalEvidenceFusionModel
from src.retrieval.memory_bank import MemoryBank
from src.training.trainer import Trainer, _LOSS_KEYS, _move_batch_to_device, _to_float
from src.training.train_core import (
    build_core_model,
    build_dataset,
    build_optimizer,
    build_runtime_data_config_file,
    build_scheduler,
    resolve_device,
    resolve_runtime_paths,
    set_seed,
)
from src.utils.io import load_yaml_config, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the extended ClinRec recommendation model.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to configs/train.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Path to configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml", help="Path to configs/model.yaml")
    parser.add_argument("--processed-root", default=None, help="Optional override for processed data root")
    parser.add_argument("--vocab-root", default=None, help="Optional override for vocab directory")
    parser.add_argument("--ddi-matrix-path", default=None, help="Optional override for DDI matrix artifact")
    parser.add_argument("--device", default=None, help="Optional override for runtime device")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for random seed")
    return parser.parse_args()


def collate_batch_with_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate_batch(records)
    batch["records"] = records
    return batch


def build_extended_dataloaders(
    *,
    runtime_data_config_path: Path,
    processed_root: Path,
    drug_vocab_size: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}")

    train_dataset = build_dataset(
        split="train",
        runtime_data_config_path=runtime_data_config_path,
        processed_root=processed_root,
        drug_vocab_size=drug_vocab_size,
    )
    val_dataset = build_dataset(
        split="val",
        runtime_data_config_path=runtime_data_config_path,
        processed_root=processed_root,
        drug_vocab_size=drug_vocab_size,
    )
    if len(train_dataset) <= 0:
        raise ValueError("Training dataset is empty")
    if len(val_dataset) <= 0:
        raise ValueError("Validation dataset is empty")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch_with_records,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch_with_records,
    )
    train_bank_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch_with_records,
    )
    val_bank_loader = DataLoader(
        val_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch_with_records,
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
) -> tuple[RetrievalEvidenceFusionModel, Any]:
    model, loss_fn = build_core_model(
        train_config=train_config,
        model_config=model_config,
        runtime_data_config_path=runtime_data_config_path,
        vocab_root=vocab_root,
        ddi_matrix_path=ddi_matrix_path,
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


class ExtendedTrainer(Trainer):
    def __init__(
        self,
        *,
        use_retrieval: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.use_retrieval = bool(use_retrieval)
        self.best_checkpoint_path = self.checkpoint_dir / "train_extended_best.pt"
        self.metrics_log_path = self.log_dir / "train_extended_metrics.jsonl"
        self.train_memory_bank: MemoryBank | None = None
        self.val_memory_bank: MemoryBank | None = None

    def _should_retry_core(self, exc: Exception) -> bool:
        if isinstance(exc, (ImportError, ModuleNotFoundError, AttributeError)):
            return True
        message = f"{type(exc).__name__}: {exc}".lower()
        keywords = (
            "retrieval",
            "neighbor",
            "group",
            "hypergraph",
            "memory bank",
            "faiss",
            "cross-split",
            "prototype",
            "semantic edge",
        )
        return any(keyword in message for keyword in keywords)

    def _forward_with_fallback(
        self,
        batch_on_device: Mapping[str, Any],
        *,
        memory_bank: MemoryBank | None,
    ) -> dict[str, Any]:
        try:
            return self.model(
                batch_on_device,
                mode="extended",
                memory_bank=memory_bank,
                records=batch_on_device.get("records"),
                decoder_top_k=self.decoder_top_k,
            )
        except Exception as exc:
            if not self._should_retry_core(exc):
                raise
            print(f"Extended forward failed; retrying this batch in core mode. Reason: {exc}")
            return self.model(
                batch_on_device,
                mode="core",
                memory_bank=None,
                records=None,
                decoder_top_k=self.decoder_top_k,
            )

    def _run_one_epoch(self, dataloader: DataLoader, *, training: bool) -> dict[str, float]:
        phase = "train" if training else "val"
        totals = {key: 0.0 for key in _LOSS_KEYS}
        total_examples = 0

        self.model.train(mode=training)
        grad_context = torch.enable_grad if training else torch.no_grad
        phase_memory_bank = self.train_memory_bank if training else self.val_memory_bank

        for step_index, batch in enumerate(dataloader, start=1):
            step_context = f"{phase} step {step_index}"
            batch_on_device = _move_batch_to_device(
                batch,
                self.device,
                non_blocking=self.non_blocking_transfer,
            )
            batch_size = int(batch_on_device["visit_mask"].shape[0])
            if batch_size <= 0:
                continue
            self._validate_batch_inputs_finite(
                batch_on_device,
                context=f"{step_context} before forward",
            )

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with grad_context():
                with self._autocast_context():
                    outputs = self._forward_with_fallback(
                        batch_on_device,
                        memory_bank=phase_memory_bank if self.use_retrieval else None,
                    )
                self._validate_model_outputs_finite(
                    outputs,
                    context=f"{step_context} after forward",
                )
                drug_logits = outputs.get("drug_logits")
                drug_probs = outputs.get("drug_probs")
                if drug_logits is None or drug_probs is None:
                    raise RuntimeError(
                        "Model did not return `drug_logits` and `drug_probs`. "
                        "Ensure a medication decoder is attached in extended training."
                    )

                with self._autocast_context():
                    loss_outputs = self.loss_fn(
                        drug_logits=drug_logits,
                        drug_probs=drug_probs,
                        target_drugs=batch_on_device["target_drugs"],
                        visit_mask=batch_on_device["visit_mask"],
                    )

                if training:
                    if self.grad_scaler_enabled and self.scaler is not None:
                        self.scaler.scale(loss_outputs["total_loss"]).backward()
                    else:
                        loss_outputs["total_loss"].backward()
                    self._optimizer_step(context=f"{step_context} after optimizer step")

            total_examples += batch_size
            for key in _LOSS_KEYS:
                totals[key] += _to_float(loss_outputs[key]) * batch_size

        if total_examples <= 0:
            raise ValueError(f"{phase} dataloader produced zero valid examples")

        return {
            f"{phase}_{key}": totals[key] / float(total_examples)
            for key in _LOSS_KEYS
        }

    def refresh_memory_banks(
        self,
        *,
        train_bank_dataloader: DataLoader | None,
        val_bank_dataloader: DataLoader | None,
    ) -> None:
        if not self.use_retrieval:
            self.train_memory_bank = None
            self.val_memory_bank = None
            print("Extended mode retrieval disabled by config; skipping memory bank refresh.")
            return

        try:
            if train_bank_dataloader is not None:
                self.train_memory_bank = build_memory_bank_from_dataloader(
                    model=self.model,
                    dataloader=train_bank_dataloader,
                    device=self.device,
                    split="train",
                )
                print(f"Built train memory bank with {len(self.train_memory_bank)} visits.")
            else:
                self.train_memory_bank = None
        except Exception as exc:
            self.train_memory_bank = None
            print(f"Warning: failed to build train memory bank; continuing without retrieval. Reason: {exc}")

        try:
            if val_bank_dataloader is not None:
                self.val_memory_bank = build_memory_bank_from_dataloader(
                    model=self.model,
                    dataloader=val_bank_dataloader,
                    device=self.device,
                    split="val",
                )
                print(f"Built val memory bank with {len(self.val_memory_bank)} visits.")
            else:
                self.val_memory_bank = None
        except Exception as exc:
            self.val_memory_bank = None
            print(f"Warning: failed to build val memory bank; continuing without retrieval. Reason: {exc}")

        if self.train_memory_bank is None and self.val_memory_bank is None:
            print("Extended mode could not build any memory bank; batches will run in soft fallback mode.")

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

        for epoch in range(1, int(epochs) + 1):
            self.refresh_memory_banks(
                train_bank_dataloader=train_bank_dataloader,
                val_bank_dataloader=val_bank_dataloader,
            )

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


def main() -> None:
    args = parse_args()
    train_config = load_yaml_config(args.config)
    data_config = load_yaml_config(args.data_config)
    model_config = load_yaml_config(args.model_config)

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

    print(f"Using device: {device}")
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
        )

        model, loss_fn = build_extended_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=resolved_paths["vocab_root"],
            ddi_matrix_path=resolved_paths["ddi_matrix_path"],
        )

    optimizer = build_optimizer(model=model, train_config=train_config)
    scheduler = build_scheduler(optimizer=optimizer, train_config=train_config)
    trainer = ExtendedTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=resolved_paths["checkpoint_dir"],
        log_dir=resolved_paths["log_dir"],
        monitor_metric="val_total_loss",
        monitor_mode="min",
        decoder_top_k=int(train_config.get("prediction", {}).get("top_k", 10)),
        amp=requested_amp,
        max_grad_norm=max_grad_norm,
        non_blocking_transfer=bool(runtime_cfg.get("non_blocking_transfer", False)),
        log_interval=int(runtime_cfg.get("log_interval", 50)),
        profile_steps=runtime_cfg.get("profile_steps"),
        run_context={
            "runtime": {
                "amp": requested_amp,
                "requested_amp": requested_amp,
                "max_grad_norm": max_grad_norm,
            },
        },
        use_retrieval=bool(extended_cfg.get("use_retrieval", True)),
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
        train_bank_dataloader=train_bank_loader if bool(extended_cfg.get("use_retrieval", True)) else None,
        val_bank_dataloader=val_bank_loader if bool(extended_cfg.get("use_retrieval", True)) else None,
        epochs=int(train_config.get("optimization", {}).get("epochs", 10)),
        extra_checkpoint_state={
            "train_config": {key: value for key, value in train_config.items() if not str(key).startswith("_")},
            "data_config": {key: value for key, value in data_config.items() if not str(key).startswith("_")},
            "model_config": {key: value for key, value in model_config.items() if not str(key).startswith("_")},
            "resolved_paths": {key: str(value) for key, value in resolved_paths.items()},
            "seed": seed,
            "train_mode": "extended",
            "group_encoder_enabled": model.group_encoder is not None,
            "retrieval_enabled": bool(extended_cfg.get("use_retrieval", True)),
        },
    )

    print(f"Best checkpoint: {fit_result['best_checkpoint_path']}")
    print(f"Monitor metric: {fit_result['monitor_metric']} (best={fit_result['best_metric']:.6f})")


if __name__ == "__main__":
    main()
