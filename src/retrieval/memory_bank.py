from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from src.utils.io import ensure_dir, load_pt, parse_datetime, resolve_path, save_pt


def _coerce_long_tensor(values: Sequence[int] | torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.long).flatten().cpu()


def _coerce_float_tensor(values: Sequence[float] | torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32).flatten().cpu()


def _coerce_2d_float_tensor(values: Sequence[Sequence[float]] | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).cpu()
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D float tensor, got shape {tuple(tensor.shape)}")
    return tensor


def _coerce_tuple_list(values: Sequence[Sequence[int]] | None) -> list[tuple[int, ...]]:
    if values is None:
        return []
    return [tuple(int(item) for item in row) for row in values]


def _visit_timestamp(record: Mapping[str, Any], step_index: int) -> str:
    base = parse_datetime(record.get("intime"))
    if base is None:
        return str(record.get("outtime", ""))
    cumulative_hours = 0.0
    for local_index in range(step_index + 1):
        cumulative_hours += float(record["steps"][local_index].get("delta_hours", 0.0))
    return (base + timedelta(hours=cumulative_hours)).strftime("%Y-%m-%d %H:%M:%S")


def _feature_set_from_mask(values: Sequence[int | bool] | None) -> tuple[int, ...]:
    if not values:
        return ()
    return tuple(index for index, flag in enumerate(values) if int(flag) != 0)


@dataclass
class MemoryBank:
    visit_states: torch.Tensor
    visit_repr: torch.Tensor
    subject_ids: torch.Tensor
    hadm_ids: torch.Tensor
    stay_ids: torch.Tensor
    visit_index: torch.Tensor
    visit_time_days: torch.Tensor
    visit_time_text: list[str]
    target_drugs: list[tuple[int, ...]]
    num_steps: torch.Tensor
    diag_code_sets: list[tuple[int, ...]]
    proc_code_sets: list[tuple[int, ...]]
    lab_feature_sets: list[tuple[int, ...]]
    vital_feature_sets: list[tuple[int, ...]]
    split: str
    _rows_by_stay_cache: list[torch.Tensor] | None = field(init=False, default=None, repr=False)
    _stay_group_ids_cache: torch.Tensor | None = field(init=False, default=None, repr=False)
    _unique_stay_ids_cache: torch.Tensor | None = field(init=False, default=None, repr=False)
    _stay_to_position_cache: dict[int, int] | None = field(init=False, default=None, repr=False)
    _normalized_visit_states_cache: torch.Tensor | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.visit_states = _coerce_2d_float_tensor(self.visit_states)
        self.visit_repr = _coerce_2d_float_tensor(self.visit_repr)
        self.subject_ids = _coerce_long_tensor(self.subject_ids)
        self.hadm_ids = _coerce_long_tensor(self.hadm_ids)
        self.stay_ids = _coerce_long_tensor(self.stay_ids)
        self.visit_index = _coerce_long_tensor(self.visit_index)
        self.visit_time_days = _coerce_float_tensor(self.visit_time_days)
        self.num_steps = _coerce_long_tensor(self.num_steps)
        self.visit_time_text = [str(value) for value in self.visit_time_text]
        self.target_drugs = _coerce_tuple_list(self.target_drugs)
        self.diag_code_sets = _coerce_tuple_list(self.diag_code_sets)
        self.proc_code_sets = _coerce_tuple_list(self.proc_code_sets)
        self.lab_feature_sets = _coerce_tuple_list(self.lab_feature_sets)
        self.vital_feature_sets = _coerce_tuple_list(self.vital_feature_sets)

        size = self.visit_states.shape[0]
        if self.visit_repr.shape[0] != size:
            raise ValueError("visit_repr length must match visit_states length")
        for name, value in (
            ("subject_ids", self.subject_ids),
            ("hadm_ids", self.hadm_ids),
            ("stay_ids", self.stay_ids),
            ("visit_index", self.visit_index),
            ("visit_time_days", self.visit_time_days),
            ("num_steps", self.num_steps),
        ):
            if value.shape[0] != size:
                raise ValueError(f"{name} length {value.shape[0]} does not match visit_states length {size}")
        for name, value in (
            ("visit_time_text", self.visit_time_text),
            ("target_drugs", self.target_drugs),
            ("diag_code_sets", self.diag_code_sets),
            ("proc_code_sets", self.proc_code_sets),
            ("lab_feature_sets", self.lab_feature_sets),
            ("vital_feature_sets", self.vital_feature_sets),
        ):
            if len(value) != size:
                raise ValueError(f"{name} length {len(value)} does not match visit_states length {size}")

    def __len__(self) -> int:
        return int(self.visit_states.shape[0])

    def _build_stay_cache(self) -> None:
        if self._rows_by_stay_cache is not None:
            return

        rows_by_stay: dict[int, list[int]] = {}
        stay_order: list[int] = []
        for row_index, stay_id in enumerate(self.stay_ids.tolist()):
            resolved_stay_id = int(stay_id)
            if resolved_stay_id not in rows_by_stay:
                rows_by_stay[resolved_stay_id] = []
                stay_order.append(resolved_stay_id)
            rows_by_stay[resolved_stay_id].append(int(row_index))

        self._rows_by_stay_cache = [
            torch.tensor(rows_by_stay[stay_id], dtype=torch.long)
            for stay_id in stay_order
        ]
        self._unique_stay_ids_cache = torch.tensor(stay_order, dtype=torch.long)
        self._stay_to_position_cache = {
            int(stay_id): int(position)
            for position, stay_id in enumerate(stay_order)
        }
        self._stay_group_ids_cache = torch.empty(len(self), dtype=torch.long)
        for stay_position, row_indices in enumerate(self._rows_by_stay_cache):
            self._stay_group_ids_cache[row_indices] = int(stay_position)

    @property
    def rows_by_stay(self) -> list[torch.Tensor]:
        self._build_stay_cache()
        if self._rows_by_stay_cache is None:
            raise RuntimeError("rows_by_stay cache was not initialized")
        return self._rows_by_stay_cache

    @property
    def stay_group_ids(self) -> torch.Tensor:
        self._build_stay_cache()
        if self._stay_group_ids_cache is None:
            raise RuntimeError("stay_group_ids cache was not initialized")
        return self._stay_group_ids_cache

    @property
    def unique_stay_ids(self) -> torch.Tensor:
        self._build_stay_cache()
        if self._unique_stay_ids_cache is None:
            raise RuntimeError("unique_stay_ids cache was not initialized")
        return self._unique_stay_ids_cache

    def stay_position(self, stay_id: int) -> int | None:
        self._build_stay_cache()
        if self._stay_to_position_cache is None:
            raise RuntimeError("stay_to_position cache was not initialized")
        return self._stay_to_position_cache.get(int(stay_id))

    @property
    def normalized_visit_states(self) -> torch.Tensor:
        if self._normalized_visit_states_cache is None:
            self._normalized_visit_states_cache = F.normalize(
                self.visit_states.to(dtype=torch.float32),
                p=2,
                dim=-1,
                eps=1.0e-12,
            )
        return self._normalized_visit_states_cache

    @classmethod
    def build_from_batch(
        cls,
        records: Sequence[Mapping[str, Any]],
        encoder_outputs: Mapping[str, torch.Tensor],
        *,
        split: str,
    ) -> "MemoryBank":
        state_sequence = torch.as_tensor(encoder_outputs["state_sequence"], dtype=torch.float32).cpu()
        visit_repr = torch.as_tensor(encoder_outputs["visit_repr"], dtype=torch.float32).cpu()
        visit_mask = torch.as_tensor(encoder_outputs["visit_mask"], dtype=torch.bool).cpu()
        if state_sequence.ndim != 3 or visit_repr.ndim != 3 or visit_mask.ndim != 2:
            raise ValueError("encoder_outputs must contain visit-level tensors with shapes (B,T,H), (B,T,H), (B,T)")
        if state_sequence.shape[:2] != visit_mask.shape or visit_repr.shape[:2] != visit_mask.shape:
            raise ValueError("state_sequence, visit_repr, and visit_mask must align on batch and time dimensions")
        if len(records) != int(state_sequence.shape[0]):
            raise ValueError("Record count must match encoder output batch size")

        state_rows: list[torch.Tensor] = []
        repr_rows: list[torch.Tensor] = []
        payload: dict[str, list[Any]] = {
            "subject_ids": [],
            "hadm_ids": [],
            "stay_ids": [],
            "visit_index": [],
            "visit_time_days": [],
            "visit_time_text": [],
            "target_drugs": [],
            "num_steps": [],
            "diag_code_sets": [],
            "proc_code_sets": [],
            "lab_feature_sets": [],
            "vital_feature_sets": [],
        }
        for batch_index, record in enumerate(records):
            for step_index, step in enumerate(record["steps"]):
                if not bool(visit_mask[batch_index, step_index].item()):
                    continue
                state_rows.append(state_sequence[batch_index, step_index])
                repr_rows.append(visit_repr[batch_index, step_index])
                payload["subject_ids"].append(int(record["subject_id"]))
                payload["hadm_ids"].append(int(record["hadm_id"]))
                payload["stay_ids"].append(int(record["stay_id"]))
                payload["visit_index"].append(int(step_index))
                payload["visit_time_text"].append(_visit_timestamp(record, step_index))
                dt = parse_datetime(payload["visit_time_text"][-1])
                payload["visit_time_days"].append(
                    0.0 if dt is None else float(dt.toordinal()) + (dt.hour / 24.0) + (dt.minute / 1440.0) + (dt.second / 86400.0)
                )
                payload["target_drugs"].append(tuple(int(drug_id) for drug_id in step.get("target_drugs", [])))
                payload["num_steps"].append(int(record.get("num_steps", 0)))
                payload["diag_code_sets"].append(tuple(int(code_id) for code_id in step.get("diagnosis_ids", [])))
                payload["proc_code_sets"].append(tuple(int(code_id) for code_id in step.get("procedure_ids", [])))
                payload["lab_feature_sets"].append(_feature_set_from_mask(step.get("lab_mask")))
                payload["vital_feature_sets"].append(_feature_set_from_mask(step.get("vital_mask")))

        hidden_dim = int(state_sequence.shape[-1])
        return cls(
            visit_states=(
                torch.stack(state_rows, dim=0)
                if state_rows
                else torch.empty((0, hidden_dim), dtype=torch.float32)
            ),
            visit_repr=(
                torch.stack(repr_rows, dim=0)
                if repr_rows
                else torch.empty((0, hidden_dim), dtype=torch.float32)
            ),
            split=split,
            **payload,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "visit_states": self.visit_states,
            "visit_repr": self.visit_repr,
            "subject_ids": self.subject_ids,
            "hadm_ids": self.hadm_ids,
            "stay_ids": self.stay_ids,
            "visit_index": self.visit_index,
            "visit_time_days": self.visit_time_days,
            "visit_time_text": list(self.visit_time_text),
            "target_drugs": list(self.target_drugs),
            "num_steps": self.num_steps,
            "diag_code_sets": list(self.diag_code_sets),
            "proc_code_sets": list(self.proc_code_sets),
            "lab_feature_sets": list(self.lab_feature_sets),
            "vital_feature_sets": list(self.vital_feature_sets),
            "split": self.split,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MemoryBank":
        return cls(
            visit_states=payload["visit_states"],
            visit_repr=payload["visit_repr"],
            subject_ids=payload["subject_ids"],
            hadm_ids=payload["hadm_ids"],
            stay_ids=payload["stay_ids"],
            visit_index=payload["visit_index"],
            visit_time_days=payload["visit_time_days"],
            visit_time_text=payload["visit_time_text"],
            target_drugs=payload["target_drugs"],
            num_steps=payload["num_steps"],
            diag_code_sets=payload["diag_code_sets"],
            proc_code_sets=payload["proc_code_sets"],
            lab_feature_sets=payload["lab_feature_sets"],
            vital_feature_sets=payload["vital_feature_sets"],
            split=str(payload["split"]),
        )

    @staticmethod
    def artifact_path(project_root: str | Path, split: str) -> Path:
        return ensure_dir(resolve_path(project_root, "data/artifacts/memory_bank")) / f"{split}.pt"

    def save(self, project_root: str | Path, *, split: str | None = None) -> Path:
        target_split = split or self.split
        payload = self.to_payload()
        payload["split"] = target_split
        return save_pt(self.artifact_path(project_root, target_split), payload)

    @classmethod
    def load(cls, project_root: str | Path, split: str) -> "MemoryBank":
        return cls.from_payload(load_pt(cls.artifact_path(project_root, split)))

    def slice_metadata(self, indices: torch.Tensor | Sequence[int]) -> dict[str, Any]:
        idx = _coerce_long_tensor(indices)
        return {
            "subject_ids": self.subject_ids[idx],
            "hadm_ids": self.hadm_ids[idx],
            "stay_ids": self.stay_ids[idx],
            "visit_index": self.visit_index[idx],
            "visit_time_days": self.visit_time_days[idx],
            "visit_time_text": [self.visit_time_text[int(i)] for i in idx.tolist()],
            "target_drugs": [self.target_drugs[int(i)] for i in idx.tolist()],
            "diag_code_sets": [self.diag_code_sets[int(i)] for i in idx.tolist()],
            "proc_code_sets": [self.proc_code_sets[int(i)] for i in idx.tolist()],
            "lab_feature_sets": [self.lab_feature_sets[int(i)] for i in idx.tolist()],
            "vital_feature_sets": [self.vital_feature_sets[int(i)] for i in idx.tolist()],
            "num_steps": self.num_steps[idx],
            "split": self.split,
        }


def build_last_visit_queries(
    records: Sequence[Mapping[str, Any]],
    encoder_outputs: Mapping[str, torch.Tensor],
    *,
    split: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    state_sequence = torch.as_tensor(encoder_outputs["state_sequence"], dtype=torch.float32).cpu()
    visit_mask = torch.as_tensor(encoder_outputs["visit_mask"], dtype=torch.bool).cpu()
    if len(records) != int(state_sequence.shape[0]):
        raise ValueError("Record count must match encoder output batch size")

    query_states: list[torch.Tensor] = []
    metadata: dict[str, list[Any]] = {
        "stay_ids": [],
        "subject_ids": [],
        "hadm_ids": [],
        "visit_indices": [],
        "visit_times": [],
        "visit_time_days": [],
        "diag_code_sets": [],
        "proc_code_sets": [],
        "lab_feature_sets": [],
        "vital_feature_sets": [],
        "split": [],
    }
    for batch_index, record in enumerate(records):
        valid_steps = int(visit_mask[batch_index].sum().item())
        if valid_steps <= 0:
            raise ValueError("Each query record must contain at least one valid visit")
        visit_index = valid_steps - 1
        step = record["steps"][visit_index]
        visit_time_text = _visit_timestamp(record, visit_index)
        dt = parse_datetime(visit_time_text)
        query_states.append(state_sequence[batch_index, visit_index])
        metadata["stay_ids"].append(int(record["stay_id"]))
        metadata["subject_ids"].append(int(record["subject_id"]))
        metadata["hadm_ids"].append(int(record["hadm_id"]))
        metadata["visit_indices"].append(int(visit_index))
        metadata["visit_times"].append(visit_time_text)
        metadata["visit_time_days"].append(
            0.0 if dt is None else float(dt.toordinal()) + (dt.hour / 24.0) + (dt.minute / 1440.0) + (dt.second / 86400.0)
        )
        metadata["diag_code_sets"].append(tuple(int(code_id) for code_id in step.get("diagnosis_ids", [])))
        metadata["proc_code_sets"].append(tuple(int(code_id) for code_id in step.get("procedure_ids", [])))
        metadata["lab_feature_sets"].append(_feature_set_from_mask(step.get("lab_mask")))
        metadata["vital_feature_sets"].append(_feature_set_from_mask(step.get("vital_mask")))
        record_split = record.get("split")
        resolved_split = split if split is not None else record_split
        metadata["split"].append(None if resolved_split is None else str(resolved_split))
    hidden_dim = int(state_sequence.shape[-1])
    query_state_tensor = (
        torch.stack(query_states, dim=0)
        if query_states
        else torch.empty((0, hidden_dim), dtype=torch.float32)
    )
    return query_state_tensor, metadata
