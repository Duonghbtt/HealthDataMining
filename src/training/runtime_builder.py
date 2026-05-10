from __future__ import annotations

import copy
import random
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import MIMICTrajectoryDataset, collate_batch, validate_patient_level_splits
from src.data.retrieval_cache import retrieval_cache_enabled
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
from src.training.losses import build_medication_loss_config
from src.utils.io import ensure_dir, load_yaml_config, read_json, resolve_path


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
        resolved["patient_id"] = int(resolved.get("patient_id", resolved.get("subject_id", -1)))
        visit_index = int(resolved.get("visit_index", max(int(resolved["num_steps"]) - 1, 0)))
        visit_position = int(
            resolved.get(
                "visit_position",
                visit_index + 1 if int(resolved["num_steps"]) > 0 else 0,
            )
        )
        resolved["visit_index"] = visit_index
        resolved["visit_position"] = visit_position
        resolved["history_length"] = int(resolved.get("history_length", visit_position))
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


def resolve_device(requested_device: str) -> torch.device:
    device = torch.device(str(requested_device))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is not available; falling back to CPU.")
        return torch.device("cpu")
    return device


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


def _infer_feature_size_from_sample_value(value: Any) -> tuple[int, bool]:
    if value is None:
        return 0, False
    shape = getattr(value, "shape", None)
    if shape is None:
        return 0, False
    if len(shape) <= 0:
        return 0, True
    return int(shape[-1]), True


def infer_numeric_feature_sizes_from_dataset(dataset: Dataset) -> tuple[int, int]:
    num_lab_features = int(getattr(dataset, "default_lab_feature_size", 0))
    num_vital_features = int(getattr(dataset, "default_vital_feature_size", 0))
    lab_feature_resolved = num_lab_features > 0
    vital_feature_resolved = num_vital_features > 0

    if lab_feature_resolved and vital_feature_resolved:
        return num_lab_features, num_vital_features

    sample_record = dataset[0]
    sample_record_keys = sorted(str(key) for key in sample_record.keys())

    if not lab_feature_resolved and "lab_feature_size" in sample_record:
        num_lab_features = int(sample_record["lab_feature_size"])
        lab_feature_resolved = True
    if not vital_feature_resolved and "vital_feature_size" in sample_record:
        num_vital_features = int(sample_record["vital_feature_size"])
        vital_feature_resolved = True

    if not lab_feature_resolved:
        num_lab_features, lab_feature_resolved = _infer_feature_size_from_sample_value(
            sample_record.get("lab_values")
        )
    if not vital_feature_resolved:
        num_vital_features, vital_feature_resolved = _infer_feature_size_from_sample_value(
            sample_record.get("vital_values")
        )

    if not lab_feature_resolved or not vital_feature_resolved:
        sample_batch = select_collate_fn(dataset)([sample_record])
        if not lab_feature_resolved:
            num_lab_features = int(sample_batch["lab_values"].shape[-1])
            lab_feature_resolved = True
        if not vital_feature_resolved:
            num_vital_features = int(sample_batch["vital_values"].shape[-1])
            vital_feature_resolved = True

    if not lab_feature_resolved or not vital_feature_resolved:
        raise ValueError(
            "Unable to infer lab/vital feature sizes from the training dataset sample. "
            f"dataset_class={type(dataset).__name__} storage_mode={_dataset_storage_mode(dataset)} "
            f"sample_record_keys={sample_record_keys}"
        )
    return num_lab_features, num_vital_features


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python/NumPy/Torch in a pickle-safe way for Windows DataLoader workers.

    PyTorch assigns each worker a deterministic torch seed before calling
    ``worker_init_fn``. On Windows, the worker function must live at module
    scope so it can be pickled under the ``spawn`` start method.
    """

    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    try:
        import numpy as np  # type: ignore[import-not-found]

        np.random.seed(worker_seed)
    except ImportError:
        pass
    torch.manual_seed(worker_seed)


def build_runtime_data_config_file(
    *,
    project_root: Path,
    data_config: Mapping[str, Any],
    processed_root: Path,
    vocab_root: Path,
    temp_dir: Path,
    retrieval_cache_config: Mapping[str, Any] | None = None,
) -> Path:
    runtime_config = copy.deepcopy({key: value for key, value in data_config.items() if not str(key).startswith("_")})
    if retrieval_cache_config is not None:
        runtime_config["retrieval_cache"] = copy.deepcopy(dict(retrieval_cache_config))
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

    retrieval_cache_cfg = runtime_config.get("retrieval_cache", {})
    if isinstance(retrieval_cache_cfg, dict):
        retrieval_cache_cfg = dict(retrieval_cache_cfg)
        cache_root = retrieval_cache_cfg.get("cache_root")
        if cache_root:
            retrieval_cache_cfg["cache_root"] = str(resolve_path(project_root, cache_root).resolve())
        runtime_config["retrieval_cache"] = retrieval_cache_cfg

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
    resolved_dataset_cache_size = int(
        dataset_cache_size if dataset_cache_size is not None else _runtime_cache_size(runtime_data_config_path)
    )
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
    if retrieval_cache_enabled(runtime_config):
        raise FileNotFoundError(
            "retrieval_cache.enabled=true requires TensorizedTrajectoryDataset so cache rows align "
            f"with tensorized sample order; missing tensorized manifest at {tensorized_manifest_path}"
        )

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
    seed: int | None = None,
    train_num_workers: int | None = None,
    val_num_workers: int | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
) -> tuple[DataLoader, DataLoader, Dataset]:
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}")
    resolved_train_num_workers = int(num_workers if train_num_workers is None else train_num_workers)
    resolved_val_num_workers = int(num_workers if val_num_workers is None else val_num_workers)
    if resolved_train_num_workers < 0:
        raise ValueError(f"train_num_workers must be non-negative, got {resolved_train_num_workers!r}")
    if resolved_val_num_workers < 0:
        raise ValueError(f"val_num_workers must be non-negative, got {resolved_val_num_workers!r}")

    split_validation = validate_patient_level_splits(runtime_data_config_path)
    if bool(split_validation.get("validated")):
        print(f"Validated patient-level split manifests: {split_validation['counts']}")

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

    def _resolve_worker_settings(loader_name: str, worker_count: int) -> tuple[bool, int | None]:
        resolved_persistent_workers = (
            worker_count > 0 if persistent_workers is None else bool(persistent_workers)
        )
        if worker_count <= 0:
            if resolved_persistent_workers:
                raise ValueError(f"{loader_name} persistent_workers=True requires num_workers > 0")
            return False, None
        resolved_prefetch_factor = 2 if prefetch_factor is None else int(prefetch_factor)
        if resolved_prefetch_factor <= 0:
            raise ValueError(f"prefetch_factor must be positive when num_workers > 0, got {prefetch_factor!r}")
        return resolved_persistent_workers, resolved_prefetch_factor

    train_persistent_workers, train_prefetch_factor = _resolve_worker_settings(
        "train",
        resolved_train_num_workers,
    )
    val_persistent_workers, val_prefetch_factor = _resolve_worker_settings(
        "val",
        resolved_val_num_workers,
    )
    print(
        "DataLoader settings: "
        f"batch_size={int(batch_size)} "
        f"pin_memory={bool(pin_memory)} "
        f"train_num_workers={resolved_train_num_workers} "
        f"train_persistent_workers={train_persistent_workers} "
        f"train_prefetch_factor={train_prefetch_factor} "
        f"val_num_workers={resolved_val_num_workers} "
        f"val_persistent_workers={val_persistent_workers} "
        f"val_prefetch_factor={val_prefetch_factor}"
    )

    def _build_loader_kwargs(
        *,
        worker_count: int,
        resolved_persistent_workers: bool,
        resolved_prefetch_factor: int | None,
    ) -> dict[str, Any]:
        loader_kwargs: dict[str, Any] = {
            "batch_size": int(batch_size),
            "num_workers": int(worker_count),
            "pin_memory": bool(pin_memory),
        }
        if seed is not None and worker_count > 0:
            loader_kwargs["worker_init_fn"] = seed_dataloader_worker
        if worker_count > 0:
            loader_kwargs["persistent_workers"] = bool(resolved_persistent_workers)
            loader_kwargs["prefetch_factor"] = int(resolved_prefetch_factor)
        return loader_kwargs

    train_generator = None
    if seed is not None:
        train_generator = torch.Generator()
        train_generator.manual_seed(int(seed))

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=select_collate_fn(train_dataset),
        generator=train_generator,
        **_build_loader_kwargs(
            worker_count=resolved_train_num_workers,
            resolved_persistent_workers=train_persistent_workers,
            resolved_prefetch_factor=train_prefetch_factor,
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=select_collate_fn(val_dataset),
        **_build_loader_kwargs(
            worker_count=resolved_val_num_workers,
            resolved_persistent_workers=val_persistent_workers,
            resolved_prefetch_factor=val_prefetch_factor,
        ),
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

    num_lab_features, num_vital_features = infer_numeric_feature_sizes_from_dataset(
        resolved_train_dataset
    )

    model_cfg = dict(model_config.get("model", {}))
    embedding_cfg = dict(model_config.get("embedding", {}))
    encoder_cfg = dict(model_config.get("encoder", {}))
    history_cfg = dict(model_config.get("history_selector", {}))
    retrieval_cfg = dict(model_config.get("retrieval", {}))
    train_core_cfg = dict(train_config.get("core", {}))
    train_extended_cfg = dict(train_config.get("extended", {}))
    retrieval_cache_cfg = dict(train_config.get("retrieval_cache", {}))
    if train_core_cfg.get("use_retrieval") is not None:
        retrieval_cfg["enabled"] = bool(train_core_cfg.get("use_retrieval"))
    elif train_extended_cfg.get("use_retrieval") is not None:
        retrieval_cfg["enabled"] = bool(train_extended_cfg.get("use_retrieval"))
    if train_extended_cfg.get("retrieval_top_k") is not None:
        retrieval_cfg["top_k"] = int(train_extended_cfg["retrieval_top_k"])
    if train_extended_cfg.get("retrieval_backend") is not None:
        retrieval_cfg["backend"] = train_extended_cfg["retrieval_backend"]
        retrieval_cfg["mode"] = train_extended_cfg["retrieval_backend"]
    if train_extended_cfg.get("use_faiss_if_available") is not None:
        retrieval_cfg["use_faiss_if_available"] = bool(train_extended_cfg["use_faiss_if_available"])
    if train_extended_cfg.get("temporal_decay_alpha") is not None:
        retrieval_cfg["temporal_decay_alpha"] = float(train_extended_cfg["temporal_decay_alpha"])
    use_precomputed_retrieval_cache = bool(
        retrieval_cache_cfg.get("enabled", False)
        and retrieval_cache_cfg.get("use_precomputed", True)
    )
    if use_precomputed_retrieval_cache:
        retrieval_cfg["enabled"] = True
    full_model_cfg = dict(model_config.get("full_model", {}))
    if retrieval_cfg.get("enabled") and str(full_model_cfg.get("history_mode", "self_only")) == "self_only":
        full_model_cfg["history_mode"] = "self_retrieval"
    debug_cfg = dict(model_config.get("debug", {}))
    fusion_cfg = dict(model_config.get("fusion", {}))
    decoder_cfg = dict(train_config.get("decoder", {}))
    loss_cfg = build_medication_loss_config(
        loss_config=train_config.get("loss", {}),
        training_config=train_config.get("training", {}),
    )

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
            exclude_future_same_patient=retrieval_cfg.get("exclude_future_same_patient"),
            exclude_future_all_patients_if_absolute_time=bool(
                retrieval_cfg.get("exclude_future_all_patients_if_absolute_time", True)
            ),
            require_absolute_time_for_cross_patient_temporal_filter=bool(
                retrieval_cfg.get("require_absolute_time_for_cross_patient_temporal_filter", False)
            ),
            use_time_gap=bool(retrieval_cfg.get("use_time_gap", True)),
            dropout=float(retrieval_cfg.get("dropout", model_dropout)),
        )
    decoder = MedicationDecoder(
        hidden_dim=hidden_dim,
        drug_vocab_size=vocab_sizes["drug"],
        dropout=float(decoder_cfg.get("dropout", model_dropout)),
        hidden_multiplier=int(decoder_cfg.get("hidden_multiplier", 2)),
        activation=str(decoder_cfg.get("activation", "relu")),
        layer_norm=bool(decoder_cfg.get("layer_norm", True)),
        decoder_type=str(decoder_cfg.get("type", "residual_mlp")),
        decoder_mode=decoder_cfg.get("mode", "legacy"),
        gate_type=str(decoder_cfg.get("gate_type", "scalar")),
        use_history_copy=bool(decoder_cfg.get("use_history_copy", True)),
        use_retrieval_copy=bool(decoder_cfg.get("use_retrieval_copy", True)),
        use_memory_copy=bool(decoder_cfg.get("use_memory_copy", False)),
        copy_projection=str(decoder_cfg.get("copy_projection", "none")),
        gate_hidden_dim=decoder_cfg.get("gate_hidden_dim"),
    )
    ddi_matrix = load_ddi_matrix(ddi_matrix_path, device="cpu")
    baseline_cfg = dict(train_config.get("baseline", {}))
    use_self_history = bool(baseline_cfg.get("use_self_history", True))
    use_ddi = bool(baseline_cfg.get("use_ddi", True))
    lambda_ddi = float(baseline_cfg.get("lambda_ddi", train_config.get("loss", {}).get("lambda_ddi", 0.0)))
    if not use_ddi:
        lambda_ddi = 0.0
        loss_cfg["use_ddi"] = False
        loss_cfg["lambda_ddi"] = 0.0
    else:
        loss_cfg["use_ddi"] = True
        loss_cfg["lambda_ddi"] = lambda_ddi

    return FullMedicationModel(
        encoder,
        history_selector,
        fusion_module,
        medication_decoder=decoder,
        retriever=retriever,
        ddi_matrix=ddi_matrix,
        lambda_ddi=lambda_ddi,
        use_self_history=use_self_history,
        history_mode=str(
            full_model_cfg.get(
                "history_mode",
                "self_retrieval" if retriever is not None and use_self_history else "retrieval_only" if retriever is not None else "self_only",
            )
        ),
        return_retrieval_aux=bool(debug_cfg.get("return_retrieval_aux", True)),
        loss_config=loss_cfg,
        use_precomputed_retrieval_cache=use_precomputed_retrieval_cache,
        retrieval_cache_config=retrieval_cache_cfg,
    )


__all__ = [
    "DirectParquetTrajectoryDataset",
    "build_core_model",
    "build_dataloaders",
    "build_dataset",
    "build_runtime_data_config_file",
    "resolve_device",
    "select_collate_fn",
]
