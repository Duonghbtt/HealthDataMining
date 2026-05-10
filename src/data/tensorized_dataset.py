from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from src.data.build_vocab import load_vocab_bundle
from src.data.retrieval_cache import (
    attach_retrieval_batch_fields,
    load_retrieval_cache_for_split,
    retrieval_cache_enabled,
    retrieval_record_from_cache,
)
from src.data.tensorization_utils import load_optional_ddi_tensors, validate_collated_batch
from src.utils.io import load_pt, load_yaml_config, read_json, resolve_path


_TENSOR_BATCH_KEYS = (
    "diag_codes",
    "diag_mask",
    "proc_codes",
    "proc_mask",
    "lab_values",
    "lab_mask",
    "vital_values",
    "vital_mask",
    "med_history",
    "med_history_mask",
    "time_delta_hours",
    "visit_mask",
    "target_drugs",
)
_ID_KEYS = ("subject_ids", "hadm_ids", "stay_ids")
_ALL_KEYS = (*_TENSOR_BATCH_KEYS, *_ID_KEYS)


def _resolve_max_open_shards(config: Mapping[str, Any] | None) -> int:
    if config is None:
        return 8
    spark_cfg = config.get("spark", {})
    if isinstance(spark_cfg, dict) and spark_cfg.get("max_open_shards_per_dataset") is not None:
        return int(spark_cfg["max_open_shards_per_dataset"])
    return 8


def tensorized_root_from_config(config: Mapping[str, Any]) -> Path:
    paths_cfg = dict(config.get("paths", {}))
    tensorized_root = paths_cfg.get("tensorized_root")
    if tensorized_root:
        return Path(resolve_path(config["_project_root"], tensorized_root))
    artifacts_root = resolve_path(config["_project_root"], paths_cfg.get("artifacts_root", "data/artifacts"))
    return Path(artifacts_root) / "tensorized"


def tensorized_manifest_path_from_config(config: Mapping[str, Any]) -> Path:
    return tensorized_root_from_config(config) / "manifest.json"


def resolve_tensorized_manifest_path(config_or_manifest_path: str | Path | Mapping[str, Any]) -> tuple[Path, dict[str, Any] | None]:
    if isinstance(config_or_manifest_path, Mapping):
        config = dict(config_or_manifest_path)
        return tensorized_manifest_path_from_config(config), config

    source_path = Path(config_or_manifest_path).resolve()
    suffix = source_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        config = load_yaml_config(source_path)
        return tensorized_manifest_path_from_config(config), config
    if source_path.is_dir():
        return source_path / "manifest.json", None
    return source_path, None


def _touch_cached_shard(
    cache: OrderedDict[int, dict[str, torch.Tensor]],
    shard_index: int,
) -> dict[str, torch.Tensor] | None:
    cached_payload = cache.pop(shard_index, None)
    if cached_payload is not None:
        cache[shard_index] = cached_payload
    return cached_payload


def _store_cached_shard(
    cache: OrderedDict[int, dict[str, torch.Tensor]],
    *,
    shard_index: int,
    payload: dict[str, torch.Tensor],
    max_open_shards: int,
) -> None:
    cache[shard_index] = payload
    while len(cache) > max_open_shards:
        cache.popitem(last=False)


def _all_same_shapes(records: list[dict[str, torch.Tensor]]) -> bool:
    if not records:
        return True
    reference_shapes = {key: tuple(records[0][key].shape) for key in _TENSOR_BATCH_KEYS}
    for record in records[1:]:
        for key in _TENSOR_BATCH_KEYS:
            if tuple(record[key].shape) != reference_shapes[key]:
                return False
    return True


def _copy_tensor_into_batch(destination: torch.Tensor, source: torch.Tensor, batch_index: int) -> None:
    slices = (batch_index, *[slice(0, dim_size) for dim_size in source.shape])
    destination[slices] = source


def _cast_tensorized_value(key: str, value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if key in {"diag_codes", "proc_codes", "med_history", "subject_ids", "hadm_ids", "stay_ids"}:
        return tensor.to(dtype=torch.long)
    if key in {"diag_mask", "proc_mask", "lab_mask", "vital_mask", "med_history_mask", "visit_mask"}:
        return tensor.to(dtype=torch.bool)
    if key in {"lab_values", "vital_values", "time_delta_hours", "target_drugs"}:
        return tensor.to(dtype=torch.float32)
    return tensor


def _resolve_shared_static_tensor(records: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor | None:
    resolved: torch.Tensor | None = None
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        if resolved is None:
            resolved = value
            continue
        if tuple(value.shape) != tuple(resolved.shape):
            raise ValueError(
                f"Static tensor `{key}` must have one shared shape across records; "
                f"got {tuple(resolved.shape)} and {tuple(value.shape)}"
            )
    return resolved


def _resolve_manifest_drug_vocab_size(
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None,
) -> int:
    manifest_value = int(manifest.get("drug_vocab_size", 0))
    if manifest_value > 0:
        return manifest_value
    if config is None:
        return 0
    vocab_bundle = load_vocab_bundle(config)
    return int(len(vocab_bundle["med_main"]["idx_to_token"]))


def tensorized_collate_batch(records: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("tensorized_collate_batch requires at least one record")

    ddi_adj = _resolve_shared_static_tensor(records, "ddi_adj")
    ddi_severity_adj = _resolve_shared_static_tensor(records, "ddi_severity_adj")

    if _all_same_shapes(records):
        batch = {key: torch.stack([record[key] for record in records], dim=0) for key in _ALL_KEYS}
        visit_position = batch["visit_mask"].sum(dim=1, dtype=torch.long)
        batch["patient_ids"] = batch["subject_ids"]
        batch["visit_index"] = torch.clamp(visit_position - 1, min=0)
        batch["visit_position"] = visit_position
        batch["history_length"] = visit_position
        if ddi_adj is not None:
            batch["ddi_adj"] = ddi_adj
        if ddi_severity_adj is not None:
            batch["ddi_severity_adj"] = ddi_severity_adj
        attach_retrieval_batch_fields(
            batch,
            records,
            drug_vocab_size=int(batch["target_drugs"].shape[-1]),
        )
        validate_collated_batch(batch)
        return batch

    batch_size = len(records)
    max_steps = max(int(record["visit_mask"].shape[0]) for record in records)
    max_diag_codes = max(int(record["diag_codes"].shape[1]) for record in records)
    max_proc_codes = max(int(record["proc_codes"].shape[1]) for record in records)
    max_history = max(int(record["med_history"].shape[1]) for record in records)
    lab_feature_size = max(int(record["lab_values"].shape[1]) for record in records)
    vital_feature_size = max(int(record["vital_values"].shape[1]) for record in records)
    drug_vocab_size = max(int(record["target_drugs"].shape[1]) for record in records)

    diag_codes = torch.zeros(batch_size, max_steps, max_diag_codes, dtype=records[0]["diag_codes"].dtype)
    diag_mask = torch.zeros(batch_size, max_steps, max_diag_codes, dtype=records[0]["diag_mask"].dtype)
    proc_codes = torch.zeros(batch_size, max_steps, max_proc_codes, dtype=records[0]["proc_codes"].dtype)
    proc_mask = torch.zeros(batch_size, max_steps, max_proc_codes, dtype=records[0]["proc_mask"].dtype)
    med_history = torch.zeros(batch_size, max_steps, max_history, dtype=records[0]["med_history"].dtype)
    med_history_mask = torch.zeros(batch_size, max_steps, max_history, dtype=records[0]["med_history_mask"].dtype)
    lab_values = torch.zeros(batch_size, max_steps, lab_feature_size, dtype=records[0]["lab_values"].dtype)
    lab_mask = torch.zeros(batch_size, max_steps, lab_feature_size, dtype=records[0]["lab_mask"].dtype)
    vital_values = torch.zeros(batch_size, max_steps, vital_feature_size, dtype=records[0]["vital_values"].dtype)
    vital_mask = torch.zeros(batch_size, max_steps, vital_feature_size, dtype=records[0]["vital_mask"].dtype)
    time_delta_hours = torch.zeros(batch_size, max_steps, dtype=records[0]["time_delta_hours"].dtype)
    visit_mask = torch.zeros(batch_size, max_steps, dtype=records[0]["visit_mask"].dtype)
    target_drugs = torch.zeros(batch_size, max_steps, drug_vocab_size, dtype=records[0]["target_drugs"].dtype)

    for batch_index, record in enumerate(records):
        _copy_tensor_into_batch(diag_codes, record["diag_codes"], batch_index)
        _copy_tensor_into_batch(diag_mask, record["diag_mask"], batch_index)
        _copy_tensor_into_batch(proc_codes, record["proc_codes"], batch_index)
        _copy_tensor_into_batch(proc_mask, record["proc_mask"], batch_index)
        _copy_tensor_into_batch(med_history, record["med_history"], batch_index)
        _copy_tensor_into_batch(med_history_mask, record["med_history_mask"], batch_index)
        _copy_tensor_into_batch(lab_values, record["lab_values"], batch_index)
        _copy_tensor_into_batch(lab_mask, record["lab_mask"], batch_index)
        _copy_tensor_into_batch(vital_values, record["vital_values"], batch_index)
        _copy_tensor_into_batch(vital_mask, record["vital_mask"], batch_index)
        _copy_tensor_into_batch(time_delta_hours, record["time_delta_hours"], batch_index)
        _copy_tensor_into_batch(visit_mask, record["visit_mask"], batch_index)
        _copy_tensor_into_batch(target_drugs, record["target_drugs"], batch_index)

    visit_position = visit_mask.sum(dim=1, dtype=torch.long)
    batch = {
        "diag_codes": diag_codes,
        "diag_mask": diag_mask,
        "proc_codes": proc_codes,
        "proc_mask": proc_mask,
        "lab_values": lab_values,
        "lab_mask": lab_mask,
        "vital_values": vital_values,
        "vital_mask": vital_mask,
        "med_history": med_history,
        "med_history_mask": med_history_mask,
        "time_delta_hours": time_delta_hours,
        "visit_mask": visit_mask,
        "target_drugs": target_drugs,
        "patient_ids": torch.stack([record["subject_ids"] for record in records], dim=0),
        "subject_ids": torch.stack([record["subject_ids"] for record in records], dim=0),
        "hadm_ids": torch.stack([record["hadm_ids"] for record in records], dim=0),
        "stay_ids": torch.stack([record["stay_ids"] for record in records], dim=0),
        "visit_index": torch.clamp(visit_position - 1, min=0),
        "visit_position": visit_position,
        "history_length": visit_position,
    }
    if ddi_adj is not None:
        batch["ddi_adj"] = ddi_adj
    if ddi_severity_adj is not None:
        batch["ddi_severity_adj"] = ddi_severity_adj
    attach_retrieval_batch_fields(
        batch,
        records,
        drug_vocab_size=int(target_drugs.shape[-1]),
    )
    validate_collated_batch(batch)
    return batch


class TensorizedTrajectoryDataset(Dataset):
    def __init__(
        self,
        split: str,
        config_or_manifest_path: str | Path | Mapping[str, Any],
        *,
        max_open_shards: int | None = None,
    ) -> None:
        manifest_path, resolved_config = resolve_tensorized_manifest_path(config_or_manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Tensorized manifest does not exist: {manifest_path}")

        manifest = read_json(manifest_path)
        split_payload = manifest.get("splits", {}).get(split)
        if split_payload is None:
            raise FileNotFoundError(f"Split `{split}` is missing from tensorized manifest {manifest_path}")

        self.split = str(split)
        self.manifest_path = manifest_path
        self.max_open_shards = int(max_open_shards if max_open_shards is not None else _resolve_max_open_shards(resolved_config))
        self._storage_mode = "tensorized_pt"
        self.default_lab_feature_size = int(manifest.get("default_lab_feature_size", 0))
        self.default_vital_feature_size = int(manifest.get("default_vital_feature_size", 0))
        self.drug_vocab_size = _resolve_manifest_drug_vocab_size(manifest, config=resolved_config)
        self.ddi_tensors = (
            load_optional_ddi_tensors(
                resolved_config,
                expected_drug_vocab_size=self.drug_vocab_size,
            )
            if resolved_config is not None
            else {}
        )

        self.shards: list[dict[str, Any]] = []
        self.cumulative_rows: list[int] = []
        self._shard_cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()

        total = 0
        for shard in split_payload:
            shard_path = manifest_path.parent / shard["path"]
            rows = int(shard["rows"])
            self.shards.append({"path": shard_path, "rows": rows})
            total += rows
            self.cumulative_rows.append(total)
        self._retrieval_cache_config = (
            dict(resolved_config)
            if resolved_config is not None and retrieval_cache_enabled(resolved_config)
            else None
        )
        self._retrieval_cache: dict[str, Any] | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_retrieval_cache"] = None
        state["_shard_cache"] = OrderedDict()
        return state

    @property
    def storage_mode(self) -> str:
        return self._storage_mode

    @property
    def num_shards(self) -> int:
        return len(self.shards)

    def __len__(self) -> int:
        return self.cumulative_rows[-1] if self.cumulative_rows else 0

    def _load_shard(self, shard_index: int) -> dict[str, torch.Tensor]:
        cached_payload = _touch_cached_shard(self._shard_cache, shard_index)
        if cached_payload is not None:
            return cached_payload

        shard = self.shards[shard_index]
        shard_path = Path(shard["path"])
        if not shard_path.exists():
            raise FileNotFoundError(f"Tensorized shard is missing: {shard_path}")

        raw_payload = load_pt(shard_path)
        if not isinstance(raw_payload, Mapping):
            raise RuntimeError(f"Tensorized shard must contain a dict payload: {shard_path}")

        visit_mask_value = raw_payload.get("visit_mask")
        if visit_mask_value is None:
            raise KeyError(f"Tensorized shard is missing required key `visit_mask`: {shard_path}")
        visit_mask = _cast_tensorized_value("visit_mask", visit_mask_value)
        if visit_mask.ndim != 2:
            raise RuntimeError(
                f"Tensorized shard visit_mask must have shape (B, T), got {tuple(visit_mask.shape)}: {shard_path}"
            )
        row_count = int(visit_mask.shape[0])
        max_steps = int(visit_mask.shape[1])

        payload: dict[str, torch.Tensor] = {}
        for key in _ALL_KEYS:
            value = raw_payload.get(key)
            if value is None:
                if key == "diag_mask":
                    payload[key] = payload["diag_codes"].ne(0)
                    continue
                if key == "proc_mask":
                    payload[key] = payload["proc_codes"].ne(0)
                    continue
                if key == "med_history_mask":
                    payload[key] = payload["med_history"].ne(0)
                    continue
                if key == "time_delta_hours":
                    payload[key] = torch.zeros(row_count, max_steps, dtype=torch.float32)
                    continue
                if key == "lab_values":
                    payload[key] = torch.zeros(
                        row_count,
                        max_steps,
                        self.default_lab_feature_size,
                        dtype=torch.float32,
                    )
                    continue
                if key == "lab_mask":
                    payload[key] = torch.zeros(
                        row_count,
                        max_steps,
                        self.default_lab_feature_size,
                        dtype=torch.bool,
                    )
                    continue
                if key == "vital_values":
                    payload[key] = torch.zeros(
                        row_count,
                        max_steps,
                        self.default_vital_feature_size,
                        dtype=torch.float32,
                    )
                    continue
                if key == "vital_mask":
                    payload[key] = torch.zeros(
                        row_count,
                        max_steps,
                        self.default_vital_feature_size,
                        dtype=torch.bool,
                    )
                    continue
                raise KeyError(f"Tensorized shard is missing key `{key}`: {shard_path}")
            payload[key] = _cast_tensorized_value(key, value)

        if row_count != int(shard["rows"]):
            raise RuntimeError(
                f"Tensorized shard row count mismatch at {shard_path}: "
                f"manifest={shard['rows']} actual={row_count}"
            )
        for key, value in payload.items():
            if key in _ID_KEYS:
                if value.ndim != 1 or int(value.shape[0]) != row_count:
                    raise RuntimeError(
                        f"Tensorized shard id field `{key}` must have shape ({row_count},), "
                        f"got {tuple(value.shape)} at {shard_path}"
                    )
                continue
            if key in {"time_delta_hours", "visit_mask"}:
                if value.ndim != 2 or tuple(value.shape) != (row_count, max_steps):
                    raise RuntimeError(
                        f"Tensorized shard field `{key}` must have shape ({row_count}, {max_steps}), "
                        f"got {tuple(value.shape)} at {shard_path}"
                    )
            elif value.ndim != 3 or tuple(value.shape[:2]) != (row_count, max_steps):
                raise RuntimeError(
                    f"Tensorized shard field `{key}` must align to batch/time shape ({row_count}, {max_steps}, *), "
                    f"got {tuple(value.shape)} at {shard_path}"
                )
            if value.is_floating_point() or value.is_complex():
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"Tensorized shard field `{key}` contains non-finite values at {shard_path}")
        if int(payload["target_drugs"].shape[-1]) != int(self.drug_vocab_size):
            raise RuntimeError(
                f"Tensorized shard target width must match med vocab size {self.drug_vocab_size}, "
                f"got {tuple(payload['target_drugs'].shape)} at {shard_path}"
            )

        _store_cached_shard(
            self._shard_cache,
            shard_index=shard_index,
            payload=payload,
            max_open_shards=self.max_open_shards,
        )
        return payload

    def _load_retrieval_cache(self) -> dict[str, Any] | None:
        if self._retrieval_cache is not None:
            return self._retrieval_cache
        if self._retrieval_cache_config is None:
            return None
        self._retrieval_cache = load_retrieval_cache_for_split(
            config=self._retrieval_cache_config,
            split=self.split,
            expected_rows=len(self),
            drug_vocab_size=self.drug_vocab_size,
        )
        return self._retrieval_cache

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        shard_index = bisect_right(self.cumulative_rows, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_rows[shard_index - 1]
        local_index = index - shard_start
        shard_payload = self._load_shard(shard_index)
        record = {key: shard_payload[key][local_index] for key in _ALL_KEYS}
        visit_position = record["visit_mask"].sum(dtype=torch.long)
        record["patient_ids"] = record["subject_ids"]
        record["visit_index"] = torch.clamp(visit_position - 1, min=0)
        record["visit_position"] = visit_position
        record["history_length"] = visit_position
        if self.ddi_tensors:
            record.update(self.ddi_tensors)
        retrieval_cache = self._load_retrieval_cache()
        if retrieval_cache is not None:
            record.update(retrieval_record_from_cache(retrieval_cache, index))
        return record


__all__ = [
    "TensorizedTrajectoryDataset",
    "resolve_tensorized_manifest_path",
    "tensorized_collate_batch",
    "tensorized_manifest_path_from_config",
    "tensorized_root_from_config",
]
