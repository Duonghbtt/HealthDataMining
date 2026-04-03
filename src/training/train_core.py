from __future__ import annotations

import argparse
import copy
import random
import tempfile
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data.build_vocab import load_vocab_bundle
from src.data.dataset import MIMICTrajectoryDataset, collate_batch
from src.models.ddi_regularization import DDIRegularizer
from src.models.full_model import RetrievalEvidenceFusionModel
from src.models.fusion import FusionModule
from src.models.history_selector import HistorySelector
from src.models.medication_decoder import MedicationDecoder
from src.models.patient_state_encoder import PatientStateEncoder
from src.training.losses import MedicationRecommendationLoss
from src.training.trainer import Trainer, _LOSS_KEYS, _move_batch_to_device, _to_float
from src.utils.io import ensure_dir, load_yaml_config, read_json, resolve_path


class DirectParquetTrajectoryDataset(Dataset):
    """Fallback dataset for direct split manifest layout under `processed/<split>`."""

    def __init__(
        self,
        split: str,
        processed_root: str | Path,
        *,
        drug_vocab_size: int,
        max_open_shards: int = 2,
    ) -> None:
        self.split = split
        self.processed_root = Path(processed_root)
        self.drug_vocab_size = int(drug_vocab_size)
        self.max_open_shards = int(max_open_shards)
        self.shards: list[dict[str, Any]] = []
        self.cumulative_rows: list[int] = []
        self._shard_cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()

        manifest_path = self.processed_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing processed manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        split_payload = manifest.get("splits", {}).get(split)
        if split_payload is None:
            raise FileNotFoundError(f"Split `{split}` is missing from manifest {manifest_path}")

        total = 0
        for shard in split_payload.get("shards", []):
            shard_path = self.processed_root / shard["path"]
            rows = int(shard["rows"])
            self.shards.append({"path": shard_path, "rows": rows})
            total += rows
            self.cumulative_rows.append(total)

    def __len__(self) -> int:
        return self.cumulative_rows[-1] if self.cumulative_rows else 0

    def _augment_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        resolved = dict(record)
        steps = list(resolved.get("steps", []))
        resolved["drug_vocab_size"] = int(resolved.get("drug_vocab_size", self.drug_vocab_size))
        resolved["num_steps"] = int(resolved.get("num_steps", len(steps)))
        resolved["lab_feature_size"] = int(
            resolved.get(
                "lab_feature_size",
                max((len(step.get("lab_values", [])) for step in steps), default=0),
            )
        )
        resolved["vital_feature_size"] = int(
            resolved.get(
                "vital_feature_size",
                max((len(step.get("vital_values", [])) for step in steps), default=0),
            )
        )
        return resolved

    def _load_shard(self, shard_index: int) -> list[dict[str, Any]]:
        if shard_index in self._shard_cache:
            rows = self._shard_cache.pop(shard_index)
            self._shard_cache[shard_index] = rows
            return rows

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required to read parquet trajectories. Install requirements.txt first."
            ) from exc

        shard = self.shards[shard_index]
        shard_path = Path(shard["path"])
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing trajectory shard: {shard_path}")

        rows = [self._augment_record(row) for row in pq.read_table(shard_path).to_pylist()]
        if len(rows) != int(shard["rows"]):
            raise RuntimeError(
                f"Shard row count mismatch at {shard_path}: manifest={shard['rows']} actual={len(rows)}"
            )

        self._shard_cache[shard_index] = rows
        while len(self._shard_cache) > self.max_open_shards:
            self._shard_cache.popitem(last=False)
        return rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative_rows, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_rows[shard_index - 1]
        local_index = index - shard_start
        return dict(self._load_shard(shard_index)[local_index])


class TqdmCoreTrainer(Trainer):
    """Core trainer that adds tqdm progress bars for train and validation."""

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
        epoch_index = int(getattr(self, "_current_epoch", 0))
        total_epochs = int(getattr(self, "_fit_total_epochs", 0))
        epoch_label = f"{epoch_index}/{total_epochs}" if total_epochs > 0 else str(epoch_index)
        desc = f"{phase.upper()} {epoch_label}"

        progress = tqdm(
            dataloader,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
        )

        for batch in progress:
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

            running_total_loss = totals["total_loss"] / float(total_examples)
            running_prediction_loss = totals["prediction_loss"] / float(total_examples)
            running_ddi_loss = totals["ddi_loss"] / float(total_examples)
            progress.set_postfix(
                total_loss=f"{running_total_loss:.4f}",
                pred_loss=f"{running_prediction_loss:.4f}",
                ddi_loss=f"{running_ddi_loss:.4f}",
            )

        progress.close()

        if total_examples <= 0:
            raise ValueError(f"{phase} dataloader produced zero valid examples")

        return {
            f"{phase}_{key}": totals[key] / float(total_examples)
            for key in _LOSS_KEYS
        }

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the core ClinRec recommendation model.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to configs/train.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Path to configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml", help="Path to configs/model.yaml")
    parser.add_argument("--processed-root", default=None, help="Optional override for processed data root")
    parser.add_argument("--vocab-root", default=None, help="Optional override for vocab directory")
    parser.add_argument("--ddi-matrix-path", default=None, help="Optional override for DDI matrix artifact")
    parser.add_argument("--device", default=None, help="Optional override for runtime device")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for random seed")
    return parser.parse_args()


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
    candidates: Sequence[Path | None],
    *,
    kind: str,
) -> Path:
    checked: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        checked.append(str(candidate))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Unable to resolve {kind}. Checked candidates: {checked if checked else ['<none>']}"
    )


def resolve_runtime_paths(
    *,
    project_root: Path,
    train_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Path]:
    train_paths = dict(train_config.get("paths", {}))
    data_paths = dict(data_config.get("paths", {}))

    processed_root = _first_existing_path(
        [
            None if args.processed_root is None else Path(args.processed_root).resolve(),
            None if train_paths.get("processed_root") is None else resolve_path(project_root, train_paths["processed_root"]).resolve(),
            None if data_paths.get("processed_root") is None else resolve_path(project_root, data_paths["processed_root"]).resolve(),
            (project_root / "handover_data" / "processed").resolve(),
        ],
        kind="processed_root",
    )
    vocab_root = _first_existing_path(
        [
            None if args.vocab_root is None else Path(args.vocab_root).resolve(),
            None if train_paths.get("vocab_root") is None else resolve_path(project_root, train_paths["vocab_root"]).resolve(),
            None
            if data_paths.get("interim_root") is None
            else (resolve_path(project_root, data_paths["interim_root"]).resolve() / "vocab"),
            (project_root / "handover_data" / "vocab").resolve(),
        ],
        kind="vocab_root",
    )
    ddi_matrix_path = _first_existing_path(
        [
            None if args.ddi_matrix_path is None else Path(args.ddi_matrix_path).resolve(),
            None if train_paths.get("ddi_matrix_path") is None else resolve_path(project_root, train_paths["ddi_matrix_path"]).resolve(),
            (project_root / "handover_data" / "processed" / "ddi" / "drug_ddi.pt").resolve(),
        ],
        kind="ddi_matrix_path",
    )

    checkpoint_dir = ensure_dir(resolve_path(project_root, train_paths.get("checkpoint_dir", "outputs/checkpoints")).resolve())
    log_dir = ensure_dir(resolve_path(project_root, train_paths.get("log_dir", "outputs/logs")).resolve())

    print("Resolved runtime paths:")
    print(f"  processed_root: {processed_root}")
    print(f"  vocab_root: {vocab_root}")
    print(f"  ddi_matrix_path: {ddi_matrix_path}")
    print(f"  checkpoint_dir: {checkpoint_dir}")
    print(f"  log_dir: {log_dir}")

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
) -> Dataset:
    try:
        return MIMICTrajectoryDataset(split, runtime_data_config_path)
    except FileNotFoundError as exc:
        manifest_path = processed_root / "manifest.json"
        if not manifest_path.exists():
            raise
        print(f"Falling back to direct parquet dataset for split `{split}`: {exc}")
        return DirectParquetTrajectoryDataset(
            split,
            processed_root,
            drug_vocab_size=drug_vocab_size,
        )


def build_dataloaders(
    *,
    runtime_data_config_path: Path,
    processed_root: Path,
    drug_vocab_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
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
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collate_batch,
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
) -> tuple[RetrievalEvidenceFusionModel, MedicationRecommendationLoss]:
    vocab_sizes = load_vocab_sizes(vocab_root)
    sample_dataset = build_dataset(
        split="train",
        runtime_data_config_path=runtime_data_config_path,
        processed_root=Path(train_config["_resolved_paths"]["processed_root"]),
        drug_vocab_size=vocab_sizes["drug"],
    )
    sample_batch = collate_batch([sample_dataset[0]])

    model_cfg = dict(model_config.get("model", {}))
    embedding_cfg = dict(model_config.get("embedding", {}))
    history_cfg = dict(model_config.get("history_selector", {}))
    fusion_cfg = dict(model_config.get("fusion", {}))

    hidden_dim = int(model_cfg.get("hidden_dim", 128))
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
    )
    decoder = MedicationDecoder(
        hidden_dim=hidden_dim,
        drug_vocab_size=vocab_sizes["drug"],
        dropout=model_dropout,
        top_k_metadata=int(train_config.get("prediction", {}).get("top_k", 10)),
    )
    model_ddi_regularizer = DDIRegularizer(ddi_matrix_path, reduction="mean")
    loss_ddi_regularizer = DDIRegularizer(ddi_matrix_path, reduction="mean")

    model = RetrievalEvidenceFusionModel(
        encoder,
        history_selector,
        fusion_module,
        medication_decoder=decoder,
        ddi_regularizer=model_ddi_regularizer,
        mode="core",
    )
    loss_fn = MedicationRecommendationLoss(
        lambda_ddi=float(train_config.get("loss", {}).get("ddi_lambda", 0.0)),
        ddi_regularizer=loss_ddi_regularizer,
        reduction="mean",
    )
    return model, loss_fn


def build_optimizer(*, model: torch.nn.Module, train_config: Mapping[str, Any]) -> torch.optim.Optimizer:
    optimization_cfg = dict(train_config.get("optimization", {}))
    optimizer_name = str(optimization_cfg.get("optimizer", "adam")).strip().lower()
    if optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer `{optimizer_name}`. Only `adam` is supported in train_core.py.")
    learning_rate = float(optimization_cfg.get("learning_rate", 1.0e-3))
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def build_scheduler(*, optimizer: torch.optim.Optimizer, train_config: Mapping[str, Any]) -> Any | None:
    scheduler_name = str(train_config.get("optimization", {}).get("scheduler", "none")).strip().lower()
    if scheduler_name != "none":
        raise ValueError(f"Unsupported scheduler `{scheduler_name}`. Only `none` is supported in train_core.py.")
    _ = optimizer
    return None


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
    device = resolve_device(args.device or runtime_cfg.get("device", "cpu"))
    seed = int(args.seed if args.seed is not None else data_config.get("seed", 17))
    num_workers = int(runtime_cfg.get("num_workers", 0))
    pin_memory = bool(runtime_cfg.get("pin_memory", device.type == "cuda"))
    set_seed(seed)

    print(f"Using device: {device}")
    print(f"Using seed: {seed}")
    print(
        "DataLoader settings: "
        f"batch_size={int(runtime_cfg.get('batch_size', 16))} "
        f"num_workers={num_workers} "
        f"pin_memory={pin_memory}"
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
        )

        model, loss_fn = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=resolved_paths["vocab_root"],
            ddi_matrix_path=resolved_paths["ddi_matrix_path"],
        )

    optimizer = build_optimizer(model=model, train_config=train_config)
    scheduler = build_scheduler(optimizer=optimizer, train_config=train_config)
    trainer = TqdmCoreTrainer(
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
    )

    fit_result = trainer.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=int(train_config.get("optimization", {}).get("epochs", 10)),
        extra_checkpoint_state={
            "train_config": {key: value for key, value in train_config.items() if not str(key).startswith("_")},
            "data_config": {key: value for key, value in data_config.items() if not str(key).startswith("_")},
            "model_config": {key: value for key, value in model_config.items() if not str(key).startswith("_")},
            "resolved_paths": {key: str(value) for key, value in resolved_paths.items()},
            "seed": seed,
        },
    )

    print(f"Best checkpoint: {fit_result['best_checkpoint_path']}")
    print(f"Monitor metric: {fit_result['monitor_metric']} (best={fit_result['best_metric']:.6f})")


if __name__ == "__main__":
    main()
