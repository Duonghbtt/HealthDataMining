from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

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

        payload: dict[str, list[Any]] = {
            "visit_states": [],
            "visit_repr": [],
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
                payload["visit_states"].append(state_sequence[batch_index, step_index].tolist())
                payload["visit_repr"].append(visit_repr[batch_index, step_index].tolist())
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

        return cls(split=split, **payload)

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
    split: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    state_sequence = torch.as_tensor(encoder_outputs["state_sequence"], dtype=torch.float32).cpu()
    visit_mask = torch.as_tensor(encoder_outputs["visit_mask"], dtype=torch.bool).cpu()
    if len(records) != int(state_sequence.shape[0]):
        raise ValueError("Record count must match encoder output batch size")

    query_states = []
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
        query_states.append(state_sequence[batch_index, visit_index].tolist())
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
        metadata["split"].append(split)
    return torch.tensor(query_states, dtype=torch.float32), metadata
