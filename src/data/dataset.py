from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from functools import partial
from pathlib import Path
from random import Random
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from src.data.build_vocab import load_vocab_bundle
from src.data.load_mimic import spark_config
from src.utils.io import iter_jsonl_gz, load_yaml_config, read_json, resolve_path


def _trajectory_root(config: dict) -> Path:
    processed_root = resolve_path(config["_project_root"], config["paths"]["processed_root"])
    return Path(processed_root) / "trajectories"


def _legacy_trajectory_file(config: dict, split: str) -> Path:
    return _trajectory_root(config) / split / "trajectories.jsonl.gz"


def _manifest_path(config: dict) -> Path:
    return _trajectory_root(config) / "manifest.json"


def _augment_record(record: dict[str, Any], *, drug_vocab_size: int) -> dict[str, Any]:
    resolved = dict(record)
    steps = list(resolved.get("steps", []))
    resolved["drug_vocab_size"] = int(resolved.get("drug_vocab_size", drug_vocab_size))
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


def _import_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for parquet trajectory loading. Install requirements.txt first."
        ) from exc
    return pq


def _materialize_table_row(table: Any, local_index: int) -> dict[str, Any]:
    return {
        column_name: table.column(column_name)[local_index].as_py()
        for column_name in table.column_names
    }


def _read_num_steps_column(shard_path: Path, *, expected_rows: int) -> list[int]:
    pq = _import_pyarrow_parquet()
    table = pq.read_table(shard_path, columns=["num_steps"])
    if table.num_rows != int(expected_rows):
        raise RuntimeError(
            f"Trajectory shard row count mismatch at {shard_path}: "
            f"manifest={expected_rows} actual={table.num_rows}"
        )
    if "num_steps" not in table.column_names:
        return [0] * int(expected_rows)
    return [int(value) for value in table["num_steps"].to_pylist()]


def detect_trajectory_layout(
    split: str,
    config_path: str | Path | dict,
    *,
    processed_root: str | Path | None = None,
) -> dict[str, Any]:
    config = config_path if isinstance(config_path, dict) else load_yaml_config(config_path)
    resolved_processed_root = (
        Path(processed_root).resolve()
        if processed_root is not None
        else resolve_path(config["_project_root"], config["paths"]["processed_root"]).resolve()
    )

    trajectory_root = resolved_processed_root / "trajectories"
    canonical_manifest = trajectory_root / "manifest.json"
    canonical_legacy = trajectory_root / split / "trajectories.jsonl.gz"
    direct_manifest = resolved_processed_root / "manifest.json"

    if canonical_manifest.exists():
        return {
            "kind": "canonical_parquet",
            "description": "canonical trajectories manifest under processed_root/trajectories",
            "processed_root": resolved_processed_root,
            "manifest_path": canonical_manifest,
        }
    if canonical_legacy.exists():
        return {
            "kind": "canonical_legacy_jsonl",
            "description": "legacy canonical jsonl trajectories under processed_root/trajectories",
            "processed_root": resolved_processed_root,
            "manifest_path": canonical_legacy,
        }
    if direct_manifest.exists():
        return {
            "kind": "direct_split_manifest",
            "description": "direct split manifest under processed_root",
            "processed_root": resolved_processed_root,
            "manifest_path": direct_manifest,
        }

    raise FileNotFoundError(
        "Unable to detect trajectory layout for "
        f"split `{split}`. Checked: {canonical_manifest}, {canonical_legacy}, {direct_manifest}"
    )


class _ParquetTrajectoryDatasetBase(Dataset):
    def __init__(
        self,
        *,
        split: str,
        drug_vocab_size: int,
        max_open_shards: int,
    ) -> None:
        self.split = split
        self.drug_vocab_size = int(drug_vocab_size)
        self.max_open_shards = int(max_open_shards)
        self.shards: list[dict[str, Any]] = []
        self.cumulative_rows: list[int] = []
        self.shard_row_indices: list[list[int]] = []
        self.row_num_steps: list[int] = []
        self._shard_cache: OrderedDict[int, Any] = OrderedDict()

    def _initialize_parquet_index(self) -> None:
        total = 0
        self.cumulative_rows = []
        self.shard_row_indices = []
        self.row_num_steps = []
        for shard in self.shards:
            rows = int(shard["rows"])
            shard_indices = list(range(total, total + rows))
            self.shard_row_indices.append(shard_indices)
            total += rows
            self.cumulative_rows.append(total)
            self.row_num_steps.extend(
                _read_num_steps_column(Path(shard["path"]), expected_rows=rows)
            )

    def __len__(self) -> int:
        return self.cumulative_rows[-1] if self.cumulative_rows else 0

    def _load_shard(self, shard_index: int) -> Any:
        if shard_index in self._shard_cache:
            table = self._shard_cache.pop(shard_index)
            self._shard_cache[shard_index] = table
            return table

        pq = _import_pyarrow_parquet()
        shard = self.shards[shard_index]
        shard_path = Path(shard["path"])
        if not shard_path.exists():
            raise FileNotFoundError(
                f"Trajectory shard for split `{self.split}` is missing: {shard_path}"
            )

        table = pq.ParquetFile(shard_path).read()
        if table.num_rows != int(shard["rows"]):
            raise RuntimeError(
                f"Trajectory shard row count mismatch for split `{self.split}` at {shard_path}: "
                f"manifest={shard['rows']} actual={table.num_rows}"
            )
        self._shard_cache[shard_index] = table
        while len(self._shard_cache) > self.max_open_shards:
            self._shard_cache.popitem(last=False)
        return table

    def _parquet_record(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative_rows, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_rows[shard_index - 1]
        local_index = index - shard_start
        table = self._load_shard(shard_index)
        return _augment_record(
            _materialize_table_row(table, local_index),
            drug_vocab_size=self.drug_vocab_size,
        )


class MIMICTrajectoryDataset(_ParquetTrajectoryDatasetBase):
    def __init__(self, split: str, config_path: str | Path) -> None:
        self.config = load_yaml_config(config_path)
        self.vocab_bundle = load_vocab_bundle(self.config)
        drug_vocab_size = len(self.vocab_bundle["drug"]["idx_to_token"])
        max_open_shards = int(spark_config(self.config).get("max_open_shards_per_dataset", 2))
        super().__init__(
            split=split,
            drug_vocab_size=drug_vocab_size,
            max_open_shards=max_open_shards,
        )
        self._storage_mode = "legacy"
        self.records: list[dict[str, Any]] = []

        manifest_path = _manifest_path(self.config)
        legacy_path = _legacy_trajectory_file(self.config, split)
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            split_payload = manifest.get("splits", {}).get(split)
            if split_payload is None:
                raise FileNotFoundError(
                    f"Split `{split}` is missing from trajectory manifest {manifest_path}."
                )
            self._storage_mode = "parquet"
            for shard in split_payload.get("shards", []):
                shard_path = _trajectory_root(self.config) / shard["path"]
                self.shards.append({"path": shard_path, "rows": int(shard["rows"])})
            self._initialize_parquet_index()
        elif legacy_path.exists():
            self.records = list(iter_jsonl_gz(legacy_path))
        else:
            raise FileNotFoundError(
                f"Neither parquet manifest {manifest_path} nor legacy trajectory file {legacy_path} exists."
            )

    def __len__(self) -> int:
        if self._storage_mode == "parquet":
            return super().__len__()
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._storage_mode == "legacy":
            return _augment_record(dict(self.records[index]), drug_vocab_size=self.drug_vocab_size)
        return self._parquet_record(index)


class DirectParquetTrajectoryDataset(_ParquetTrajectoryDatasetBase):
    """Dataset for direct split manifest layout under `processed/<split>`."""

    def __init__(
        self,
        split: str,
        processed_root: str | Path,
        *,
        drug_vocab_size: int,
        max_open_shards: int = 2,
    ) -> None:
        super().__init__(
            split=split,
            drug_vocab_size=drug_vocab_size,
            max_open_shards=max_open_shards,
        )
        self.processed_root = Path(processed_root)
        self.layout_kind = "direct_split_manifest"

        manifest_path = self.processed_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing processed manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        split_payload = manifest.get("splits", {}).get(split)
        if split_payload is None:
            raise FileNotFoundError(f"Split `{split}` is missing from manifest {manifest_path}")

        for shard in split_payload.get("shards", []):
            shard_path = self.processed_root / shard["path"]
            self.shards.append({"path": shard_path, "rows": int(shard["rows"])})
        self._initialize_parquet_index()

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._parquet_record(index)


class ShardLengthBatchSampler(Sampler[list[int]]):
    """Batch sampler that keeps training batches within shard and near sequence length."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        length_bucket_window: int = 256,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")
        if int(length_bucket_window) <= 0:
            raise ValueError(
                f"length_bucket_window must be positive, got {length_bucket_window!r}"
            )
        shard_row_indices = getattr(dataset, "shard_row_indices", None)
        row_num_steps = getattr(dataset, "row_num_steps", None)
        if shard_row_indices is None or row_num_steps is None:
            raise TypeError("ShardLengthBatchSampler requires a dataset with shard_row_indices and row_num_steps")

        self.shard_row_indices = [list(indices) for indices in shard_row_indices]
        self.row_num_steps = list(row_num_steps)
        self.batch_size = int(batch_size)
        self.length_bucket_window = int(length_bucket_window)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        total_batches = 0
        for shard_indices in self.shard_row_indices:
            shard_size = len(shard_indices)
            if self.drop_last:
                total_batches += shard_size // self.batch_size
            else:
                total_batches += (shard_size + self.batch_size - 1) // self.batch_size
        return total_batches

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _sortish_indices(self, indices: Sequence[int], *, rng: Random) -> list[int]:
        windows = [
            list(indices[start : start + self.length_bucket_window])
            for start in range(0, len(indices), self.length_bucket_window)
        ]
        if self.shuffle:
            rng.shuffle(windows)

        ordered: list[int] = []
        reverse = False
        for window in windows:
            window.sort(key=lambda index: self.row_num_steps[index], reverse=reverse)
            ordered.extend(window)
            reverse = not reverse
        return ordered

    def __iter__(self):
        rng = Random(self.seed + self._epoch)
        shard_order = list(range(len(self.shard_row_indices)))
        if self.shuffle:
            rng.shuffle(shard_order)

        for shard_index in shard_order:
            shard_indices = list(self.shard_row_indices[shard_index])
            ordered_indices = self._sortish_indices(shard_indices, rng=rng)
            for start in range(0, len(ordered_indices), self.batch_size):
                batch_indices = ordered_indices[start : start + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                if batch_indices:
                    yield batch_indices


def _max_length(records: list[dict[str, Any]], field_name: str) -> int:
    return max(
        (
            len(step.get(field_name, []))
            for record in records
            for step in record.get("steps", [])
        ),
        default=0,
    )


def build_collate_fn(
    *,
    include_full_targets: bool = True,
    include_final_target: bool = True,
    max_visits: int | None = None,
    max_history: int | None = None,
):
    return partial(
        collate_batch,
        include_full_targets=include_full_targets,
        include_final_target=include_final_target,
        max_visits=max_visits,
        max_history=max_history,
    )


def collate_batch(
    records: list[dict[str, Any]],
    *,
    include_full_targets: bool = True,
    include_final_target: bool = True,
    max_visits: int | None = None,
    max_history: int | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("collate_batch requires at least one record")
    if max_visits is not None and int(max_visits) <= 0:
        raise ValueError(f"max_visits must be positive when provided, got {max_visits!r}")
    if max_history is not None and int(max_history) <= 0:
        raise ValueError(f"max_history must be positive when provided, got {max_history!r}")

    resolved_max_visits = None if max_visits is None else int(max_visits)
    resolved_max_history = None if max_history is None else int(max_history)
    prepared_records: list[dict[str, Any]] = []
    for record in records:
        steps = list(record.get("steps", []))
        if resolved_max_visits is not None:
            steps = steps[-resolved_max_visits:]
        prepared_records.append(
            {
                **dict(record),
                "steps": steps,
                "num_steps": len(steps),
            }
        )

    batch_size = len(prepared_records)
    max_steps = max(int(record["num_steps"]) for record in prepared_records)
    max_diag_codes = _max_length(prepared_records, "diagnosis_ids")
    max_proc_codes = _max_length(prepared_records, "procedure_ids")
    max_history_length = _max_length(prepared_records, "med_history_ids")
    if resolved_max_history is not None:
        max_history_length = min(max_history_length, resolved_max_history)
    drug_vocab_size = max(int(record.get("drug_vocab_size", 0)) for record in prepared_records)
    lab_feature_size = max(int(record.get("lab_feature_size", 0)) for record in prepared_records)
    vital_feature_size = max(int(record.get("vital_feature_size", 0)) for record in prepared_records)

    diag_codes = torch.zeros(batch_size, max_steps, max_diag_codes, dtype=torch.long)
    diag_mask = torch.zeros(batch_size, max_steps, max_diag_codes, dtype=torch.bool)
    proc_codes = torch.zeros(batch_size, max_steps, max_proc_codes, dtype=torch.long)
    proc_mask = torch.zeros(batch_size, max_steps, max_proc_codes, dtype=torch.bool)
    med_history = torch.zeros(batch_size, max_steps, max_history_length, dtype=torch.long)
    med_history_mask = torch.zeros(batch_size, max_steps, max_history_length, dtype=torch.bool)
    lab_values = torch.zeros(batch_size, max_steps, lab_feature_size, dtype=torch.float32)
    lab_mask = torch.zeros(batch_size, max_steps, lab_feature_size, dtype=torch.bool)
    vital_values = torch.zeros(batch_size, max_steps, vital_feature_size, dtype=torch.float32)
    vital_mask = torch.zeros(batch_size, max_steps, vital_feature_size, dtype=torch.bool)
    time_delta_hours = torch.zeros(batch_size, max_steps, dtype=torch.float32)
    visit_mask = torch.zeros(batch_size, max_steps, dtype=torch.bool)
    visit_lengths = torch.zeros(batch_size, dtype=torch.long)
    final_target_drugs = torch.zeros(batch_size, drug_vocab_size, dtype=torch.float32)
    target_drugs = (
        torch.zeros(batch_size, max_steps, drug_vocab_size, dtype=torch.float32)
        if include_full_targets
        else None
    )

    subject_ids: list[int] = []
    hadm_ids: list[int] = []
    stay_ids: list[int] = []

    for batch_index, record in enumerate(prepared_records):
        subject_ids.append(int(record["subject_id"]))
        hadm_ids.append(int(record["hadm_id"]))
        stay_ids.append(int(record["stay_id"]))
        visit_lengths[batch_index] = int(record["num_steps"])
        steps = list(record["steps"])
        for step_index, step in enumerate(record["steps"]):
            visit_mask[batch_index, step_index] = True
            diagnosis_ids = list(step.get("diagnosis_ids", []))
            procedure_ids = list(step.get("procedure_ids", []))
            history_ids = list(step.get("med_history_ids", []))
            if resolved_max_history is not None:
                history_ids = history_ids[-resolved_max_history:]
            target_ids = list(step.get("target_drugs", []))

            if diagnosis_ids:
                diag_codes[batch_index, step_index, : len(diagnosis_ids)] = torch.tensor(diagnosis_ids, dtype=torch.long)
                diag_mask[batch_index, step_index, : len(diagnosis_ids)] = True
            if procedure_ids:
                proc_codes[batch_index, step_index, : len(procedure_ids)] = torch.tensor(procedure_ids, dtype=torch.long)
                proc_mask[batch_index, step_index, : len(procedure_ids)] = True
            if history_ids:
                med_history[batch_index, step_index, : len(history_ids)] = torch.tensor(history_ids, dtype=torch.long)
                med_history_mask[batch_index, step_index, : len(history_ids)] = True

            if lab_feature_size:
                lab_values[batch_index, step_index] = torch.tensor(step.get("lab_values", []), dtype=torch.float32)
                lab_mask[batch_index, step_index] = torch.tensor(step.get("lab_mask", []), dtype=torch.bool)
            if vital_feature_size:
                vital_values[batch_index, step_index] = torch.tensor(step.get("vital_values", []), dtype=torch.float32)
                vital_mask[batch_index, step_index] = torch.tensor(step.get("vital_mask", []), dtype=torch.bool)

            if target_drugs is not None:
                for drug_id in target_ids:
                    if 0 <= int(drug_id) < drug_vocab_size:
                        target_drugs[batch_index, step_index, int(drug_id)] = 1.0
            time_delta_hours[batch_index, step_index] = float(step.get("delta_hours", 0.0))

        if include_final_target and steps:
            for drug_id in steps[-1].get("target_drugs", []):
                if 0 <= int(drug_id) < drug_vocab_size:
                    final_target_drugs[batch_index, int(drug_id)] = 1.0

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
        "visit_lengths": visit_lengths,
        "subject_ids": subject_ids,
        "hadm_ids": hadm_ids,
        "stay_ids": stay_ids,
    }
    if include_full_targets and target_drugs is not None:
        batch["target_drugs"] = target_drugs
    if include_final_target:
        batch["final_target_drugs"] = final_target_drugs
    return batch


__all__ = [
    "DirectParquetTrajectoryDataset",
    "MIMICTrajectoryDataset",
    "ShardLengthBatchSampler",
    "build_collate_fn",
    "collate_batch",
    "detect_trajectory_layout",
]
