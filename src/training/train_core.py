from __future__ import annotations

import argparse
import copy
import random
import tempfile
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
from src.models.ddi_regularization import DDIRegularizer, load_ddi_artifact
from src.models.full_model import RetrievalEvidenceFusionModel
from src.models.fusion import FusionModule
from src.models.history_selector import HistorySelector
from src.models.medication_decoder import MedicationDecoder
from src.models.patient_state_encoder import PatientStateEncoder
from src.training.losses import MedicationRecommendationLoss
from src.training.trainer import Trainer
from src.utils.io import ensure_dir, load_yaml_config, read_json, resolve_path


class TqdmCoreTrainer(Trainer):
    """Core trainer that adds tqdm progress bars for train and validation."""

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
        progress.set_postfix(**postfix)

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
    if bool(core_cfg.get("use_retrieval", False)):
        raise ValueError(f"{context_label} does not support core.use_retrieval=true.")
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
    history_cfg = dict(model_config.get("history_selector", {}))
    fusion_cfg = dict(model_config.get("fusion", {}))

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
    )
    decoder = MedicationDecoder(
        hidden_dim=hidden_dim,
        drug_vocab_size=vocab_sizes["drug"],
        dropout=model_dropout,
        top_k_metadata=int(train_config.get("prediction", {}).get("top_k", 10)),
    )
    ddi_artifact = load_ddi_artifact(ddi_matrix_path, device="cpu")
    ddi_context = {key: value for key, value in ddi_artifact.items() if key != "matrix"}
    ddi_context["status"] = "active" if ddi_context["active"] else "inactive"
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
    )
    loss_fn = MedicationRecommendationLoss(
        lambda_ddi=float(train_config.get("loss", {}).get("ddi_lambda", 0.0)),
        ddi_regularizer=loss_ddi_regularizer,
        ddi_context=ddi_context,
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
    raw_train_config = load_yaml_config(args.config)
    raw_data_config = load_yaml_config(args.data_config)
    raw_model_config = load_yaml_config(args.model_config)
    profile_name = resolve_profile_name(raw_train_config, args.profile)
    train_config = apply_profile_overrides(raw_train_config, profile_name=profile_name)
    data_config = apply_profile_overrides(raw_data_config, profile_name=profile_name)
    model_config = apply_profile_overrides(raw_model_config, profile_name=profile_name)

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
    persistent_workers = bool(runtime_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = runtime_cfg.get("prefetch_factor")
    length_bucket_window = int(runtime_cfg.get("length_bucket_window", 256))
    train_decoder_top_k = int(runtime_cfg.get("train_decoder_top_k", 0))
    matmul_precision = runtime_cfg.get("matmul_precision")
    feature_cfg = dict(data_config.get("features", {}))
    spark_cfg = dict(data_config.get("spark", {}))
    max_open_shards = int(spark_cfg.get("max_open_shards_per_dataset", 2))
    max_visits = feature_cfg.get("max_visits")
    max_history = feature_cfg.get("max_history")
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
        f"amp={bool(runtime_cfg.get('amp', False))} "
        f"grad_accum_steps={int(runtime_cfg.get('grad_accum_steps', 1))} "
        f"non_blocking_transfer={bool(runtime_cfg.get('non_blocking_transfer', False))} "
        f"train_decoder_top_k={train_decoder_top_k} "
        f"profile_steps={runtime_cfg.get('profile_steps')} "
        f"matmul_precision={matmul_precision}"
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
        )

        model, loss_fn = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=resolved_paths["vocab_root"],
            ddi_matrix_path=resolved_paths["ddi_matrix_path"],
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
        decoder_top_k=train_decoder_top_k,
        amp=bool(runtime_cfg.get("amp", False)),
        grad_accum_steps=int(runtime_cfg.get("grad_accum_steps", 1)),
        non_blocking_transfer=bool(runtime_cfg.get("non_blocking_transfer", False)),
        log_interval=int(runtime_cfg.get("log_interval", 50)),
        profile_steps=runtime_cfg.get("profile_steps"),
        run_context={
            "selected_profile": profile_name,
            "ddi_context": copy.deepcopy(loss_fn.ddi_context),
            "configured_ddi_lambda": float(loss_fn.configured_lambda_ddi),
            "effective_ddi_lambda": float(loss_fn.effective_lambda_ddi),
            "dataset_layouts": dataset_layouts,
            "runtime": {
                "batch_size": int(runtime_cfg.get("batch_size", 16)),
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "persistent_workers": persistent_workers if num_workers > 0 else False,
                "prefetch_factor": None if num_workers <= 0 else prefetch_factor,
                "amp": bool(runtime_cfg.get("amp", False)),
                "grad_accum_steps": int(runtime_cfg.get("grad_accum_steps", 1)),
                "non_blocking_transfer": bool(runtime_cfg.get("non_blocking_transfer", False)),
                "log_interval": int(runtime_cfg.get("log_interval", 50)),
                "profile_steps": runtime_cfg.get("profile_steps"),
                "train_decoder_top_k": train_decoder_top_k,
                "matmul_precision": matmul_precision,
                "length_bucket_window": length_bucket_window,
            },
        },
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
            "selected_profile": profile_name,
            "seed": seed,
            "ddi_context": copy.deepcopy(loss_fn.ddi_context),
            "configured_ddi_lambda": float(loss_fn.configured_lambda_ddi),
            "effective_ddi_lambda": float(loss_fn.effective_lambda_ddi),
            "dataset_layouts": dataset_layouts,
        },
    )

    print(f"Best checkpoint: {fit_result['best_checkpoint_path']}")
    print(f"Monitor metric: {fit_result['monitor_metric']} (best={fit_result['best_metric']:.6f})")


if __name__ == "__main__":
    main()
