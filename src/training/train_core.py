from __future__ import annotations

import argparse
import copy
import os
import random
import tempfile
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import MIMICTrajectoryDataset, collate_batch
from src.data.tensorized_dataset import (
    TensorizedTrajectoryDataset,
    tensorized_collate_batch,
    tensorized_manifest_path_from_config,
)
from src.models.ddi_regularization import load_ddi_matrix
from src.models.full_model import FullMedicationModel
from src.models.fusion import FusionModule
from src.models.history_selector import SelfHistorySelector
from src.models.medication_decoder import MedicationDecoder
from src.models.patient_state_encoder import PatientStateEncoder
from src.retrieval.topk_retriever import TopKVisitRetriever
from src.training import runtime_builder
from src.training.trainer import Trainer
from src.utils.io import ensure_dir, load_yaml_config, read_json, resolve_path, write_json


class DirectParquetTrajectoryDataset(Dataset):
    """Fallback dataset for direct split manifest layout under `processed/<split>`."""

    def __init__(
        self,
        split: str,
        processed_root: str | Path,
        *,
        drug_vocab_size: int,
        max_open_shards: int = 8,
    ) -> None:
        self.split = split
        self.processed_root = Path(processed_root)
        self.drug_vocab_size = int(drug_vocab_size)
        self.max_open_shards = int(max_open_shards)
        self._storage_mode = "direct_parquet"
        self.shards: list[dict[str, Any]] = []
        self.cumulative_rows: list[int] = []
        self._shard_cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self.default_lab_feature_size = 0
        self.default_vital_feature_size = 0

        manifest_path = self.processed_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing processed manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        split_payload = manifest.get("splits", {}).get(split)
        if split_payload is None:
            raise FileNotFoundError(f"Split `{split}` is missing from manifest {manifest_path}")

        metadata_path = self.processed_root / "metadata.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            self.default_lab_feature_size = int(metadata.get("lab_feature_size", 0))
            self.default_vital_feature_size = int(metadata.get("vital_feature_size", 0))

        total = 0
        for shard in split_payload.get("shards", []):
            shard_path = self.processed_root / shard["path"]
            rows = int(shard["rows"])
            self.shards.append({"path": shard_path, "rows": rows})
            total += rows
            self.cumulative_rows.append(total)

    @property
    def storage_mode(self) -> str:
        return self._storage_mode

    @property
    def num_shards(self) -> int:
        return len(self.shards)

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
                max((len(step.get("lab_values", [])) for step in steps), default=self.default_lab_feature_size),
            )
        )
        resolved["vital_feature_size"] = int(
            resolved.get(
                "vital_feature_size",
                max((len(step.get("vital_values", [])) for step in steps), default=self.default_vital_feature_size),
            )
        )
        return resolved

    def _touch_cached_shard(self, shard_index: int) -> list[dict[str, Any]] | None:
        cached_rows = self._shard_cache.pop(shard_index, None)
        if cached_rows is not None:
            self._shard_cache[shard_index] = cached_rows
        return cached_rows

    def _store_cached_shard(self, shard_index: int, rows: list[dict[str, Any]]) -> None:
        self._shard_cache[shard_index] = rows
        while len(self._shard_cache) > self.max_open_shards:
            self._shard_cache.popitem(last=False)

    def _load_shard(self, shard_index: int) -> list[dict[str, Any]]:
        cached_rows = self._touch_cached_shard(shard_index)
        if cached_rows is not None:
            return cached_rows

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

        rows = [self._augment_record(row) for row in pq.read_table(shard_path, use_threads=True).to_pylist()]
        if len(rows) != int(shard["rows"]):
            raise RuntimeError(
                f"Shard row count mismatch at {shard_path}: manifest={shard['rows']} actual={len(rows)}"
            )

        self._store_cached_shard(shard_index, rows)
        return rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative_rows, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_rows[shard_index - 1]
        local_index = index - shard_start
        return dict(self._load_shard(shard_index)[local_index])


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
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for the fixed prediction threshold used in validation/evaluation",
    )
    parser.add_argument(
        "--baseline-mode",
        default=None,
        choices=("current_only", "current_self_history", "current_self_history_ddi"),
        help="Select the baseline protocol without manually editing configs",
    )
    parser.add_argument(
        "--monitor-metric",
        default=None,
        help="Optional override for the checkpoint monitor metric (defaults to current repo behavior)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for number of training epochs")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short end-to-end smoke path with limited train/validation/eval batches",
    )
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional cap for train batches per epoch")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional cap for validation batches per epoch")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Optional cap for post-train evaluation batches")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


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


def _runtime_cache_size(runtime_data_config_path: Path) -> int:
    runtime_data_config = load_yaml_config(runtime_data_config_path)
    spark_cfg = runtime_data_config.get("spark", {})
    if isinstance(spark_cfg, dict) and spark_cfg.get("max_open_shards_per_dataset") is not None:
        return int(spark_cfg["max_open_shards_per_dataset"])
    return 8


def _dataset_storage_mode(dataset: Dataset) -> str:
    return str(
        getattr(
            dataset,
            "storage_mode",
            getattr(dataset, "_storage_mode", "unknown"),
        )
    )


def _dataset_num_shards(dataset: Dataset) -> int:
    num_shards = getattr(dataset, "num_shards", None)
    if num_shards is not None:
        return int(num_shards)
    shards = getattr(dataset, "shards", None)
    if isinstance(shards, Sequence):
        return len(shards)
    return 0


def _print_dataset_details(split: str, dataset: Dataset) -> None:
    print(
        f"Dataset `{split}`: "
        f"class={type(dataset).__name__} "
        f"storage_mode={_dataset_storage_mode(dataset)} "
        f"size={len(dataset)} "
        f"num_shards={_dataset_num_shards(dataset)} "
        f"max_open_shards={getattr(dataset, 'max_open_shards', 'n/a')}"
    )


def select_collate_fn(dataset: Dataset):
    if isinstance(dataset, TensorizedTrajectoryDataset) or _dataset_storage_mode(dataset) == "tensorized_pt":
        return tensorized_collate_batch
    return collate_batch


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
            None if data_paths.get("interim_root") is None else (resolve_path(project_root, data_paths["interim_root"]).resolve() / "vocab"),
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
    report_dir = ensure_dir(resolve_path(project_root, train_paths.get("report_dir", "outputs/reports")).resolve())

    print("Resolved runtime paths:")
    print(f"  processed_root: {processed_root}")
    print(f"  vocab_root: {vocab_root}")
    print(f"  ddi_matrix_path: {ddi_matrix_path}")
    print(f"  checkpoint_dir: {checkpoint_dir}")
    print(f"  log_dir: {log_dir}")
    print(f"  report_dir: {report_dir}")

    return {
        "processed_root": processed_root,
        "vocab_root": vocab_root,
        "ddi_matrix_path": ddi_matrix_path,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir,
        "report_dir": report_dir,
    }


def resolve_baseline_settings(
    *,
    train_config: Mapping[str, Any],
    baseline_mode: str | None,
) -> dict[str, Any]:
    requested_mode = str(
        baseline_mode
        or train_config.get("baseline", {}).get("mode")
        or "current_self_history_ddi"
    ).strip()
    configured_lambda_ddi = float(train_config.get("loss", {}).get("lambda_ddi", 0.0))

    if requested_mode == "current_only":
        return {
            "mode": requested_mode,
            "use_self_history": False,
            "use_ddi": False,
            "lambda_ddi": 0.0,
        }
    if requested_mode == "current_self_history":
        return {
            "mode": requested_mode,
            "use_self_history": True,
            "use_ddi": False,
            "lambda_ddi": 0.0,
        }
    if requested_mode == "current_self_history_ddi":
        return {
            "mode": requested_mode,
            "use_self_history": True,
            "use_ddi": True,
            "lambda_ddi": configured_lambda_ddi,
        }
    raise ValueError(f"Unsupported baseline_mode: {requested_mode!r}")


def apply_baseline_settings(train_config: Mapping[str, Any], baseline_settings: Mapping[str, Any]) -> dict[str, Any]:
    runtime_train_config = copy.deepcopy(dict(train_config))
    runtime_train_config.setdefault("loss", {})
    runtime_train_config["loss"]["lambda_ddi"] = float(baseline_settings["lambda_ddi"])
    runtime_train_config["baseline"] = {
        "mode": str(baseline_settings["mode"]),
        "use_self_history": bool(baseline_settings["use_self_history"]),
        "use_ddi": bool(baseline_settings["use_ddi"]),
        "lambda_ddi": float(baseline_settings["lambda_ddi"]),
    }
    return runtime_train_config


def build_runtime_data_config_file(
    *,
    project_root: Path,
    data_config: Mapping[str, Any],
    processed_root: Path,
    vocab_root: Path,
    temp_dir: Path,
) -> Path:
    runtime_config = copy.deepcopy({key: value for key, value in data_config.items() if not str(key).startswith("_")})
    runtime_config.setdefault("paths", {})
    runtime_paths = runtime_config["paths"]
    runtime_paths["processed_root"] = str(processed_root.resolve())
    runtime_paths["interim_root"] = str(vocab_root.parent.resolve())
    runtime_paths["vocab_root"] = str(vocab_root.resolve())

    for path_key in (
        "raw_root",
        "cohort_root",
        "trajectory_interim_root",
        "ddi_root",
        "rxnorm_root",
        "drugbank_vocab_path",
        "ddinter_root",
        "artifacts_root",
        "tensorized_root",
        "encoder_artifact_root",
    ):
        path_value = runtime_paths.get(path_key)
        if path_value:
            runtime_paths[path_key] = str(resolve_path(project_root, path_value).resolve())

    spark_cfg = runtime_config.get("spark", {})
    if isinstance(spark_cfg, dict):
        spark_cfg = dict(spark_cfg)
        stage_cache_dir = spark_cfg.get("stage_cache_dir")
        if stage_cache_dir:
            spark_cfg["stage_cache_dir"] = str(resolve_path(project_root, stage_cache_dir).resolve())
        runtime_config["spark"] = spark_cfg

    runtime_config_path = temp_dir / "runtime_data.yaml"
    runtime_config_path.write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8")
    return runtime_config_path


def build_dataset(
    *,
    split: str,
    runtime_data_config_path: Path,
    processed_root: Path,
    drug_vocab_size: int,
    dataset_cache_size: int | None = None,
) -> Dataset:
    resolved_dataset_cache_size = int(dataset_cache_size if dataset_cache_size is not None else _runtime_cache_size(runtime_data_config_path))
    runtime_config = load_yaml_config(runtime_data_config_path)
    tensorized_manifest_path = tensorized_manifest_path_from_config(runtime_config)
    if tensorized_manifest_path.exists():
        try:
            dataset = TensorizedTrajectoryDataset(
                split,
                runtime_data_config_path,
                max_open_shards=resolved_dataset_cache_size,
            )
            print(
                f"Using TensorizedTrajectoryDataset for split `{split}` "
                f"(size={len(dataset)}, manifest={tensorized_manifest_path})"
            )
            return dataset
        except FileNotFoundError as exc:
            print(f"Tensorized dataset unavailable for split `{split}`: {exc}")

    try:
        dataset = MIMICTrajectoryDataset(split, runtime_data_config_path)
        print(
            f"Using MIMICTrajectoryDataset for split `{split}` "
            f"(size={len(dataset)}, config={runtime_data_config_path})"
        )
        return dataset
    except FileNotFoundError as exc:
        manifest_path = processed_root / "manifest.json"
        if not manifest_path.exists():
            raise
        print(f"Falling back to direct parquet dataset for split `{split}`: {exc}")
        dataset = DirectParquetTrajectoryDataset(
            split,
            processed_root,
            drug_vocab_size=drug_vocab_size,
            max_open_shards=resolved_dataset_cache_size,
        )
        print(
            f"Using DirectParquetTrajectoryDataset for split `{split}` "
            f"(size={len(dataset)}, processed_root={processed_root})"
        )
        return dataset


def build_dataloaders(
    *,
    runtime_data_config_path: Path,
    processed_root: Path,
    drug_vocab_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, Dataset]:
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}")
    if int(num_workers) < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers!r}")

    dataset_cache_size = _runtime_cache_size(runtime_data_config_path)
    train_dataset = build_dataset(
        split="train",
        runtime_data_config_path=runtime_data_config_path,
        processed_root=processed_root,
        drug_vocab_size=drug_vocab_size,
        dataset_cache_size=dataset_cache_size,
    )
    val_dataset = build_dataset(
        split="val",
        runtime_data_config_path=runtime_data_config_path,
        processed_root=processed_root,
        drug_vocab_size=drug_vocab_size,
        dataset_cache_size=dataset_cache_size,
    )
    if len(train_dataset) <= 0:
        raise ValueError("Training dataset is empty")
    if len(val_dataset) <= 0:
        raise ValueError("Validation dataset is empty")

    _print_dataset_details("train", train_dataset)
    _print_dataset_details("val", val_dataset)

    persistent_workers = int(num_workers) > 0
    prefetch_factor = 2 if persistent_workers else None
    print(
        "DataLoader settings: "
        f"batch_size={int(batch_size)} "
        f"num_workers={int(num_workers)} "
        f"pin_memory={bool(pin_memory)} "
        f"persistent_workers={persistent_workers} "
        f"prefetch_factor={prefetch_factor}"
    )

    loader_kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if persistent_workers:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=select_collate_fn(train_dataset),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=select_collate_fn(val_dataset),
        **loader_kwargs,
    )
    return train_loader, val_loader, train_dataset


def load_vocab_sizes(vocab_root: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    sizes["diagnosis"] = int(read_json(vocab_root / "diagnosis_vocab.json")["size"])
    sizes["procedure"] = int(read_json(vocab_root / "procedure_vocab.json")["size"])

    med_vocab_path = vocab_root / "med_vocab_main.json"
    legacy_drug_vocab_path = vocab_root / "drug_vocab.json"
    resolved_drug_vocab_path = med_vocab_path if med_vocab_path.exists() else legacy_drug_vocab_path
    sizes["drug"] = int(read_json(resolved_drug_vocab_path)["size"])
    return sizes


def build_core_model(
    *,
    train_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    train_dataset: Dataset | None = None,
    runtime_data_config_path: Path | None = None,
    processed_root: Path | None = None,
    vocab_root: Path,
    ddi_matrix_path: Path,
) -> FullMedicationModel:
    vocab_sizes = load_vocab_sizes(vocab_root)
    resolved_train_dataset = train_dataset
    if resolved_train_dataset is None:
        if runtime_data_config_path is None or processed_root is None:
            raise ValueError("Either `train_dataset` or both `runtime_data_config_path` and `processed_root` must be provided.")
        resolved_train_dataset = build_dataset(
            split="train",
            runtime_data_config_path=runtime_data_config_path,
            processed_root=processed_root,
            drug_vocab_size=vocab_sizes["drug"],
        )

    num_lab_features, num_vital_features = runtime_builder.infer_numeric_feature_sizes_from_dataset(
        resolved_train_dataset
    )

    model_cfg = dict(model_config.get("model", {}))
    embedding_cfg = dict(model_config.get("embedding", {}))
    encoder_cfg = dict(model_config.get("encoder", {}))
    history_cfg = dict(model_config.get("history_selector", {}))
    retrieval_cfg = dict(model_config.get("retrieval", {}))
    full_model_cfg = dict(model_config.get("full_model", {}))
    debug_cfg = dict(model_config.get("debug", {}))
    fusion_cfg = dict(model_config.get("fusion", {}))

    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    model_dropout = float(model_cfg.get("dropout", 0.1))
    code_embedding_dim = int(embedding_cfg.get("diag_dim", hidden_dim))
    proc_dim = int(embedding_cfg.get("proc_dim", code_embedding_dim))
    if proc_dim != code_embedding_dim:
        raise ValueError(
            "PatientStateEncoder currently expects a shared code embedding dim; "
            f"got diag_dim={code_embedding_dim}, proc_dim={proc_dim}"
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
        num_lab_features=num_lab_features,
        num_vital_features=num_vital_features,
        code_embedding_dim=code_embedding_dim,
        medication_embedding_dim=int(embedding_cfg.get("drug_dim", hidden_dim)),
        numeric_projection_dim=numeric_projection_dim,
        time_embedding_dim=int(embedding_cfg.get("time_dim", 32)),
        visit_hidden_dim=hidden_dim,
        hidden_dim=hidden_dim,
        dropout=model_dropout,
        encoder_mode=str(encoder_cfg.get("mode", "legacy_gru")),
        modality_hidden_dim=encoder_cfg.get("modality_hidden_dim"),
        fusion_hidden_dim=encoder_cfg.get("fusion_hidden_dim"),
        modality_dropout=encoder_cfg.get("modality_dropout"),
        use_temporal_attention=bool(encoder_cfg.get("use_temporal_attention", True)),
        temporal_attention_heads=int(encoder_cfg.get("temporal_attention_heads", 1)),
        temporal_attention_dropout=encoder_cfg.get("temporal_attention_dropout"),
    )
    history_selector = SelfHistorySelector(
        hidden_dim=hidden_dim,
        dropout=float(history_cfg.get("dropout", 0.1)),
        self_top_k=history_cfg.get("top_k", history_cfg.get("self_top_k", 3)),
        selection_mode=str(
            history_cfg.get(
                "mode",
                "visit_only" if bool(history_cfg.get("enabled", True)) else "none",
            )
        ),
        attention_type=str(history_cfg.get("attention_type", "softmax_topk")),
        return_attention_weights=bool(history_cfg.get("return_attention_weights", True)),
        save_selected_indices=bool(history_cfg.get("save_selected_indices", True)),
    )
    fusion_module = FusionModule(
        hidden_dim=hidden_dim,
        dropout=float(fusion_cfg.get("dropout", model_dropout)),
        strategy=str(fusion_cfg.get("mode", fusion_cfg.get("strategy", "gated"))),
    )
    retriever = None
    if bool(retrieval_cfg.get("enabled", False)):
        retriever = TopKVisitRetriever(
            hidden_dim=hidden_dim,
            drug_vocab_size=vocab_sizes["drug"],
            top_k=int(retrieval_cfg.get("top_k", 5)),
            backend=str(retrieval_cfg.get("mode", retrieval_cfg.get("backend", "bruteforce"))),
            use_faiss_if_available=bool(retrieval_cfg.get("use_faiss_if_available", True)),
            similarity_mode=str(retrieval_cfg.get("similarity_mode", "cosine_decay")),
            temporal_decay_alpha=float(retrieval_cfg.get("temporal_decay_alpha", 0.05)),
            allow_same_patient=bool(retrieval_cfg.get("allow_same_patient", False)),
            exclude_future=bool(retrieval_cfg.get("exclude_future", True)),
            exclude_exact_match=bool(retrieval_cfg.get("exclude_exact_match", True)),
            use_time_gap=bool(retrieval_cfg.get("use_time_gap", True)),
            dropout=float(retrieval_cfg.get("dropout", model_dropout)),
        )
    decoder = MedicationDecoder(
        hidden_dim=hidden_dim,
        drug_vocab_size=vocab_sizes["drug"],
        dropout=model_dropout,
    )
    ddi_matrix = load_ddi_matrix(ddi_matrix_path, device="cpu")
    lambda_ddi = float(train_config.get("loss", {}).get("lambda_ddi", 0.0))
    baseline_cfg = dict(train_config.get("baseline", {}))
    use_self_history = bool(baseline_cfg.get("use_self_history", True))

    return FullMedicationModel(
        encoder,
        history_selector,
        fusion_module,
        medication_decoder=decoder,
        retriever=retriever,
        ddi_matrix=ddi_matrix,
        lambda_ddi=lambda_ddi,
        history_mode=str(
            full_model_cfg.get(
                "history_mode",
                "self_retrieval" if retriever is not None and use_self_history else "retrieval_only" if retriever is not None else "self_only",
            )
        ),
        use_self_history=use_self_history,
        return_retrieval_aux=bool(debug_cfg.get("return_retrieval_aux", True)),
    )


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
    original_train_config = load_yaml_config(args.config)
    data_config = load_yaml_config(args.data_config)
    model_config = load_yaml_config(args.model_config)
    baseline_settings = resolve_baseline_settings(
        train_config=original_train_config,
        baseline_mode=args.baseline_mode,
    )
    train_config = apply_baseline_settings(original_train_config, baseline_settings)

    project_root = Path(train_config["_project_root"]).resolve()
    resolved_paths = resolve_runtime_paths(
        project_root=project_root,
        train_config=train_config,
        data_config=data_config,
        args=args,
    )

    runtime_cfg = dict(train_config.get("runtime", {}))
    run_cfg = dict(train_config.get("run", {}))
    device = runtime_builder.resolve_device(args.device or runtime_cfg.get("device", "cpu"))
    seed = int(args.seed if args.seed is not None else data_config.get("seed", 17))
    num_workers = int(runtime_cfg.get("num_workers", 0))
    pin_memory = bool(runtime_cfg.get("pin_memory", device.type == "cuda"))
    threshold = float(args.threshold if args.threshold is not None else train_config.get("prediction", {}).get("threshold", 0.5))
    training_cfg = dict(train_config.get("training", {}))
    configured_selection_metric = str(training_cfg.get("selection_metric", "")).strip().lower()
    if args.monitor_metric is not None:
        monitor_metric = str(args.monitor_metric)
    elif runtime_cfg.get("monitor_metric") is not None:
        monitor_metric = str(runtime_cfg.get("monitor_metric"))
    elif configured_selection_metric in {"jaccard", "f1", "prauc"}:
        monitor_metric = f"val_{configured_selection_metric}"
    else:
        monitor_metric = "val_total_loss"
    if runtime_cfg.get("monitor_mode") is not None:
        monitor_mode = str(runtime_cfg.get("monitor_mode"))
    elif monitor_metric in {"val_jaccard", "val_f1", "val_prauc"}:
        monitor_mode = "max"
    else:
        monitor_mode = "min"
    set_seed(seed)
    smoke_test = bool(args.smoke_test or run_cfg.get("smoke_test", False))
    max_train_batches = (
        args.max_train_batches
        if args.max_train_batches is not None
        else run_cfg.get("max_train_batches")
    )
    max_val_batches = (
        args.max_val_batches
        if args.max_val_batches is not None
        else run_cfg.get("max_val_batches")
    )
    max_eval_batches = (
        args.max_eval_batches
        if args.max_eval_batches is not None
        else run_cfg.get("max_eval_batches")
    )
    if smoke_test:
        if max_train_batches is None:
            max_train_batches = 2
        if max_val_batches is None:
            max_val_batches = 2
        if max_eval_batches is None:
            max_eval_batches = 2

    print(f"Using device: {device}")
    print(f"Using seed: {seed}")
    print("Run configuration:")
    print(f"  baseline_mode: {baseline_settings['mode']}")
    print(f"  use_self_history: {baseline_settings['use_self_history']}")
    print(f"  use_ddi: {baseline_settings['use_ddi']}")
    print(f"  lambda_ddi: {float(baseline_settings['lambda_ddi']):.6f}")
    print(f"  retrieval_enabled: {bool(model_config.get('retrieval', {}).get('enabled', False))}")
    print(f"  history_mode: {str(model_config.get('full_model', {}).get('history_mode', 'self_only'))}")
    print(f"  stage_training: {bool(train_config.get('loss', {}).get('stage_training', False))}")
    print(f"  stage1_epochs: {int(train_config.get('training', {}).get('stage1_epochs', 0))}")
    print(f"  stage2_epochs: {int(train_config.get('training', {}).get('stage2_epochs', 0))}")
    print(f"  threshold: {threshold:.4f}")
    print(f"  monitor_metric: {monitor_metric}")
    print(f"  monitor_mode: {monitor_mode}")
    print(f"  smoke_test: {smoke_test}")
    print(f"  max_train_batches: {max_train_batches}")
    print(f"  max_val_batches: {max_val_batches}")
    print(f"  max_eval_batches: {max_eval_batches}")

    with tempfile.TemporaryDirectory(prefix="clinrec_runtime_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        runtime_data_config_path = runtime_builder.build_runtime_data_config_file(
            project_root=project_root,
            data_config=data_config,
            processed_root=resolved_paths["processed_root"],
            vocab_root=resolved_paths["vocab_root"],
            temp_dir=temp_dir,
        )
        med_vocab_path = resolved_paths["vocab_root"] / "med_vocab_main.json"
        legacy_drug_vocab_path = resolved_paths["vocab_root"] / "drug_vocab.json"
        resolved_drug_vocab_path = med_vocab_path if med_vocab_path.exists() else legacy_drug_vocab_path
        drug_vocab_size = int(read_json(resolved_drug_vocab_path)["size"])

        train_loader, val_loader, train_dataset = runtime_builder.build_dataloaders(
            runtime_data_config_path=runtime_data_config_path,
            processed_root=resolved_paths["processed_root"],
            drug_vocab_size=drug_vocab_size,
            batch_size=int(runtime_cfg.get("batch_size", 16)),
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )
        model = runtime_builder.build_core_model(
            train_config=train_config,
            model_config=model_config,
            train_dataset=train_dataset,
            vocab_root=resolved_paths["vocab_root"],
            ddi_matrix_path=resolved_paths["ddi_matrix_path"],
        )

    optimizer = build_optimizer(model=model, train_config=train_config)
    scheduler = build_scheduler(optimizer=optimizer, train_config=train_config)
    optimization_config = train_config.get("optimization", {})
    max_grad_norm = optimization_config.get("max_grad_norm", 1.0)
    sanitize_code_embedding_grads = optimization_config.get(
        "sanitize_code_embedding_grads",
        optimization_config.get("sanitize_diagnosis_embedding_grad", True),
    )
    code_embedding_grad_max_norm = optimization_config.get(
        "code_embedding_grad_max_norm",
        optimization_config.get("diagnosis_embedding_grad_max_norm", 0.5),
    )
    freeze_code_embedding_epochs = optimization_config.get(
        "freeze_code_embedding_epochs",
        optimization_config.get("diagnosis_embedding_freeze_epochs", 1),
    )
    debug_checks_config = dict(train_config.get("debug_checks", {}))
    debug_checks_enabled = debug_checks_config.get("enabled", True)
    debug_checks_light_mode = debug_checks_config.get("light_mode", True)
    debug_check_every_n_steps = debug_checks_config.get("check_every_n_steps", 100)
    sync_timing = debug_checks_config.get("sync_timing", False)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=resolved_paths["checkpoint_dir"],
        log_dir=resolved_paths["log_dir"],
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
        validation_threshold=threshold,
        use_self_history=bool(baseline_settings["use_self_history"]),
        use_ddi=bool(baseline_settings["use_ddi"]),
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        max_grad_norm=max_grad_norm,
        sanitize_code_embedding_grads=sanitize_code_embedding_grads,
        code_embedding_grad_max_norm=code_embedding_grad_max_norm,
        freeze_code_embedding_epochs=freeze_code_embedding_epochs,
        debug_checks_enabled=debug_checks_enabled,
        debug_checks_light_mode=debug_checks_light_mode,
        debug_check_every_n_steps=debug_check_every_n_steps,
        sync_timing=sync_timing,
    )

    training_epochs = int(
        args.epochs
        if args.epochs is not None
        else train_config.get("optimization", {}).get("epochs", 10)
    )
    if smoke_test:
        training_epochs = 1
    print(f"  epochs: {training_epochs}")

    fit_result = trainer.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=training_epochs,
        extra_checkpoint_state={
            "train_config": {key: value for key, value in train_config.items() if not str(key).startswith("_")},
            "data_config": {key: value for key, value in data_config.items() if not str(key).startswith("_")},
            "model_config": {key: value for key, value in model_config.items() if not str(key).startswith("_")},
            "resolved_paths": {key: str(value) for key, value in resolved_paths.items()},
            "seed": seed,
            "baseline_mode": str(baseline_settings["mode"]),
            "threshold": threshold,
            "smoke_test": smoke_test,
            "max_train_batches": max_train_batches,
            "max_val_batches": max_val_batches,
            "max_eval_batches": max_eval_batches,
        },
    )

    print(f"Best checkpoint: {fit_result['best_checkpoint_path']}")
    print(f"Monitor metric: {fit_result['monitor_metric']} (best={fit_result['best_metric']:.6f})")

    best_checkpoint_path = fit_result.get("best_checkpoint_path")
    if not best_checkpoint_path:
        raise RuntimeError("Training finished without producing a best checkpoint.")

    best_lambda_config_path = ensure_dir(resolved_paths["report_dir"]) / f"best_lambda_ddi_config_{baseline_settings['mode']}.json"
    write_json(
        best_lambda_config_path,
        {
            "baseline_mode": str(baseline_settings["mode"]),
            "best_checkpoint_path": str(best_checkpoint_path),
            "monitor_metric": fit_result["monitor_metric"],
            "best_metric": float(fit_result["best_metric"]),
            "loss": copy.deepcopy(dict(train_config.get("loss", {}))),
            "training": copy.deepcopy(dict(train_config.get("training", {}))),
        },
    )

    from src.evaluation.evaluate_core import evaluate_checkpoint

    eval_config_path = resolve_path(project_root, "configs/eval.yaml")
    evaluation_report = evaluate_checkpoint(
        checkpoint_path=Path(str(best_checkpoint_path)),
        eval_config_path=eval_config_path,
        split="test",
        threshold=args.threshold,
        device=device,
        max_eval_batches=max_eval_batches,
    )
    if hasattr(model, "get_retrieval_policy"):
        retrieval_policy_path = ensure_dir(resolved_paths["report_dir"]) / "retrieval_policy.json"
        write_json(retrieval_policy_path, model.get_retrieval_policy())
    else:
        retrieval_policy_path = None
    baseline_report_path = ensure_dir(resolved_paths["report_dir"]) / f"baseline_report_{baseline_settings['mode']}.json"
    baseline_report_payload = {
        "baseline_mode": str(baseline_settings["mode"]),
        "seed": seed,
        "threshold": threshold,
        "smoke_test": smoke_test,
        "monitor_metric": fit_result["monitor_metric"],
        "best_metric": float(fit_result["best_metric"]),
        "best_checkpoint_path": str(best_checkpoint_path),
        "training_artifacts": {
            "metrics_per_epoch_json": fit_result["metrics_per_epoch_json"],
            "metrics_per_epoch_csv": fit_result["metrics_per_epoch_csv"],
            "best_metrics_json": fit_result["best_metrics_json"],
            "best_lambda_ddi_config_json": str(best_lambda_config_path),
            "retrieval_policy_json": None if retrieval_policy_path is None else str(retrieval_policy_path),
        },
        "evaluation": evaluation_report,
    }
    write_json(baseline_report_path, baseline_report_payload)
    print(f"Baseline report: {baseline_report_path}")


if __name__ == "__main__":
    main()
