from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class VisitMemoryRecord:
    patient_id: int
    visit_index: int
    visit_time: float
    has_absolute_time: bool
    visit_embedding: torch.Tensor
    medication_evidence: torch.Tensor
    metadata: dict[str, Any]


def _as_tensor_1d(name: str, value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must have shape (N,), got {tuple(tensor.shape)}")
    return tensor


def _infer_visit_times(
    *,
    visit_mask: torch.Tensor,
    time_delta_hours: torch.Tensor | None,
    visit_time_absolute_hours: torch.Tensor | None,
    visit_time_absolute_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, time_steps = visit_mask.shape
    if visit_time_absolute_hours is not None:
        time_tensor = torch.as_tensor(visit_time_absolute_hours, dtype=torch.float32)
        if tuple(time_tensor.shape) != (batch_size, time_steps):
            raise ValueError(
                "visit_time_absolute_hours must align with visit_mask: "
                f"got {tuple(time_tensor.shape)} and {tuple(visit_mask.shape)}"
            )
        if visit_time_absolute_mask is None:
            absolute_mask = visit_mask.clone()
        else:
            absolute_mask = torch.as_tensor(visit_time_absolute_mask, dtype=torch.bool)
            if tuple(absolute_mask.shape) != (batch_size, time_steps):
                raise ValueError(
                    "visit_time_absolute_mask must align with visit_mask: "
                    f"got {tuple(absolute_mask.shape)} and {tuple(visit_mask.shape)}"
                )
        return time_tensor, absolute_mask
    if time_delta_hours is not None:
        time_tensor = torch.as_tensor(time_delta_hours, dtype=torch.float32)
        if tuple(time_tensor.shape) != (batch_size, time_steps):
            raise ValueError(
                "time_delta_hours must align with visit_mask: "
                f"got {tuple(time_tensor.shape)} and {tuple(visit_mask.shape)}"
            )
        return torch.cumsum(time_tensor, dim=1), torch.zeros_like(visit_mask, dtype=torch.bool)
    return (
        torch.arange(time_steps, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1),
        torch.zeros_like(visit_mask, dtype=torch.bool),
    )


class VisitMemoryBank:
    """Visit-level retrieval bank with explicit temporal-leakage metadata.

    Exact same visits are always identifiable by `(patient_id, visit_index)`.
    Same-patient future visits can always be blocked by visit order.
    Cross-patient future filtering is only claimed when absolute visit time is
    available for both query and candidate visits.
    """

    def __init__(
        self,
        *,
        split_name: str | None = None,
        time_is_absolute: bool = False,
    ) -> None:
        self.split_name = None if split_name is None else str(split_name)
        self.time_is_absolute = bool(time_is_absolute)
        self.embedding_dim: int | None = None
        self.medication_dim: int | None = None
        self._records: list[VisitMemoryRecord] = []
        self._record_keys: dict[tuple[int, int], int] = {}
        self._absolute_time_records = 0

    def __len__(self) -> int:
        return len(self._records)

    @property
    def num_visits(self) -> int:
        return len(self)

    @property
    def has_absolute_time(self) -> bool:
        return self._absolute_time_records > 0

    @property
    def all_visits_have_absolute_time(self) -> bool:
        return bool(self._records) and self._absolute_time_records == len(self._records)

    def describe_temporal_policy(
        self,
        *,
        exact_match_blocked: bool = True,
        same_patient_future_blocked: bool = True,
        cross_patient_absolute_temporal_filter: bool = False,
        require_absolute_time_for_cross_patient_temporal_filter: bool = False,
    ) -> dict[str, Any]:
        if self.all_visits_have_absolute_time:
            notes = (
                "Absolute visit times are available for all records, so retrieval can "
                "block future visits across patients as well as within patients."
            )
        elif self.has_absolute_time:
            notes = (
                "Absolute visit times are only partially available in the memory bank. "
                "Same-patient future filtering remains safe by visit index, but "
                "cross-patient temporal filtering is only partially enforceable."
            )
        else:
            notes = (
                "Absolute visit times are unavailable. Retrieval remains exact-match "
                "safe and same-patient future-safe by visit index; cross-patient "
                "chronology is guaranteed only by the retrieval-pool split boundary."
            )
        if require_absolute_time_for_cross_patient_temporal_filter:
            notes += " Cross-patient candidates without absolute time are excluded."
        return {
            "memory_bank_split": self.split_name,
            "has_absolute_time": bool(self.has_absolute_time),
            "all_visits_have_absolute_time": bool(self.all_visits_have_absolute_time),
            "exact_match_blocked": bool(exact_match_blocked),
            "same_patient_future_blocked": bool(same_patient_future_blocked),
            "cross_patient_absolute_temporal_filter": bool(cross_patient_absolute_temporal_filter),
            "num_visits": int(self.num_visits),
            "notes": notes,
        }

    def validate(self) -> None:
        if not self._records:
            return
        if self.embedding_dim is None or self.medication_dim is None:
            raise ValueError("Memory bank dimension metadata is missing.")
        for record in self._records:
            if record.visit_embedding.ndim != 1 or record.visit_embedding.shape[0] != self.embedding_dim:
                raise ValueError("Inconsistent visit embedding shape in memory bank.")
            if record.medication_evidence.ndim != 1 or record.medication_evidence.shape[0] != self.medication_dim:
                raise ValueError("Inconsistent medication evidence shape in memory bank.")

    def _validate_row_tensors(
        self,
        *,
        visit_embedding: torch.Tensor,
        medication_evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = torch.as_tensor(visit_embedding, dtype=torch.float32).detach().cpu().reshape(-1)
        medication = torch.as_tensor(medication_evidence, dtype=torch.float32).detach().cpu().reshape(-1)
        if self.embedding_dim is None:
            self.embedding_dim = int(embedding.shape[0])
        if self.medication_dim is None:
            self.medication_dim = int(medication.shape[0])
        if int(embedding.shape[0]) != self.embedding_dim:
            raise ValueError(
                f"visit_embedding width must stay constant at {self.embedding_dim}, got {int(embedding.shape[0])}"
            )
        if int(medication.shape[0]) != self.medication_dim:
            raise ValueError(
                f"medication_evidence width must stay constant at {self.medication_dim}, got {int(medication.shape[0])}"
            )
        return embedding, medication

    def add(
        self,
        *,
        patient_id: int,
        visit_index: int,
        visit_time: float,
        has_absolute_time: bool = False,
        visit_embedding: torch.Tensor,
        medication_evidence: torch.Tensor,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        embedding, medication = self._validate_row_tensors(
            visit_embedding=visit_embedding,
            medication_evidence=medication_evidence,
        )
        resolved_metadata = dict(metadata or {})
        record = VisitMemoryRecord(
            patient_id=int(patient_id),
            visit_index=int(visit_index),
            visit_time=float(visit_time),
            has_absolute_time=bool(has_absolute_time),
            visit_embedding=embedding,
            medication_evidence=medication,
            metadata=resolved_metadata,
        )
        record_key = (record.patient_id, record.visit_index)
        existing_index = self._record_keys.get(record_key)
        if existing_index is None:
            self._record_keys[record_key] = len(self._records)
            self._records.append(record)
            if record.has_absolute_time:
                self._absolute_time_records += 1
            return
        previous_record = self._records[existing_index]
        if previous_record.has_absolute_time and not record.has_absolute_time:
            self._absolute_time_records -= 1
        elif not previous_record.has_absolute_time and record.has_absolute_time:
            self._absolute_time_records += 1
        self._records[existing_index] = record

    def add_batch(
        self,
        *,
        patient_ids: torch.Tensor | list[int],
        visit_embeddings: torch.Tensor,
        medication_evidence: torch.Tensor,
        visit_mask: torch.Tensor,
        time_delta_hours: torch.Tensor | None = None,
        visit_time_absolute_hours: torch.Tensor | None = None,
        visit_time_absolute_mask: torch.Tensor | None = None,
        batch_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        visit_embeddings = torch.as_tensor(visit_embeddings, dtype=torch.float32)
        medication_evidence = torch.as_tensor(medication_evidence, dtype=torch.float32)
        visit_mask = torch.as_tensor(visit_mask, dtype=torch.bool)
        if visit_embeddings.ndim != 3:
            raise ValueError(f"visit_embeddings must have shape (B, T, H), got {tuple(visit_embeddings.shape)}")
        if medication_evidence.ndim != 3:
            raise ValueError(
                f"medication_evidence must have shape (B, T, D), got {tuple(medication_evidence.shape)}"
            )
        if visit_mask.ndim != 2:
            raise ValueError(f"visit_mask must have shape (B, T), got {tuple(visit_mask.shape)}")
        if tuple(visit_embeddings.shape[:2]) != tuple(visit_mask.shape):
            raise ValueError(
                "visit_embeddings and visit_mask must align on batch/time dimensions: "
                f"got {tuple(visit_embeddings.shape[:2])} and {tuple(visit_mask.shape)}"
            )
        if tuple(medication_evidence.shape[:2]) != tuple(visit_mask.shape):
            raise ValueError(
                "medication_evidence and visit_mask must align on batch/time dimensions: "
                f"got {tuple(medication_evidence.shape[:2])} and {tuple(visit_mask.shape)}"
            )

        patient_id_tensor = _as_tensor_1d("patient_ids", patient_ids, dtype=torch.long)
        if patient_id_tensor.shape[0] != visit_embeddings.shape[0]:
            raise ValueError(
                f"patient_ids length must equal batch size {visit_embeddings.shape[0]}, got {patient_id_tensor.shape[0]}"
            )
        visit_times, absolute_time_mask = _infer_visit_times(
            visit_mask=visit_mask,
            time_delta_hours=time_delta_hours,
            visit_time_absolute_hours=visit_time_absolute_hours,
            visit_time_absolute_mask=visit_time_absolute_mask,
        )
        self.time_is_absolute = bool(self.time_is_absolute or bool(absolute_time_mask.any().item()))
        metadata_payload = dict(batch_metadata or {})
        for batch_index in range(visit_embeddings.shape[0]):
            per_row_metadata = {
                key: value[batch_index] if isinstance(value, (list, tuple)) and len(value) == visit_embeddings.shape[0] else value
                for key, value in metadata_payload.items()
            }
            for step_index in range(visit_embeddings.shape[1]):
                if not bool(visit_mask[batch_index, step_index].item()):
                    continue
                self.add(
                    patient_id=int(patient_id_tensor[batch_index].item()),
                    visit_index=int(step_index),
                    visit_time=float(visit_times[batch_index, step_index].item()),
                    has_absolute_time=bool(absolute_time_mask[batch_index, step_index].item()),
                    visit_embedding=visit_embeddings[batch_index, step_index],
                    medication_evidence=medication_evidence[batch_index, step_index],
                    metadata={
                        **per_row_metadata,
                        "step_index": int(step_index),
                        "split_name": self.split_name,
                    },
                )

    def build(self, *, records: list[VisitMemoryRecord]) -> "VisitMemoryBank":
        self._records = []
        self._record_keys = {}
        self.embedding_dim = None
        self.medication_dim = None
        self._absolute_time_records = 0
        for record in records:
            self.add(
                patient_id=record.patient_id,
                visit_index=record.visit_index,
                visit_time=record.visit_time,
                has_absolute_time=record.has_absolute_time,
                visit_embedding=record.visit_embedding,
                medication_evidence=record.medication_evidence,
                metadata=record.metadata,
            )
        return self

    def export_embeddings(self) -> torch.Tensor:
        if not self._records:
            embedding_dim = int(self.embedding_dim or 0)
            return torch.empty(0, embedding_dim, dtype=torch.float32)
        return torch.stack([record.visit_embedding for record in self._records], dim=0)

    def export_medication_evidence(self) -> torch.Tensor:
        if not self._records:
            medication_dim = int(self.medication_dim or 0)
            return torch.empty(0, medication_dim, dtype=torch.float32)
        return torch.stack([record.medication_evidence for record in self._records], dim=0)

    def export_metadata(self) -> dict[str, Any]:
        return {
            "patient_ids": torch.tensor([record.patient_id for record in self._records], dtype=torch.long),
            "visit_indices": torch.tensor([record.visit_index for record in self._records], dtype=torch.long),
            "visit_times": torch.tensor([record.visit_time for record in self._records], dtype=torch.float32),
            "has_absolute_time": torch.tensor([record.has_absolute_time for record in self._records], dtype=torch.bool),
            "metadata": [dict(record.metadata) for record in self._records],
            "time_is_absolute": self.time_is_absolute,
            "split_name": self.split_name,
        }

    def get_candidate_pool(
        self,
        *,
        patient_id: int | None,
        visit_index: int | None,
        visit_time: float | None,
        query_has_absolute_time: bool = False,
        allow_same_patient: bool,
        exclude_future: bool,
        exclude_exact_match: bool,
        exclude_future_all_patients_if_absolute_time: bool = True,
        require_absolute_time_for_cross_patient_temporal_filter: bool = False,
    ) -> dict[str, Any]:
        if not self._records:
            return {
                "indices": torch.empty(0, dtype=torch.long),
                "patient_ids": torch.empty(0, dtype=torch.long),
                "visit_indices": torch.empty(0, dtype=torch.long),
                "visit_times": torch.empty(0, dtype=torch.float32),
                "has_absolute_time": torch.empty(0, dtype=torch.bool),
                "visit_embeddings": self.export_embeddings(),
                "medication_evidence": self.export_medication_evidence(),
                "metadata": [],
            }

        metadata = self.export_metadata()
        mask = torch.ones(len(self._records), dtype=torch.bool)
        patient_tensor = metadata["patient_ids"]
        visit_index_tensor = metadata["visit_indices"]
        visit_time_tensor = metadata["visit_times"]
        absolute_time_tensor = metadata["has_absolute_time"]

        if patient_id is not None and not allow_same_patient:
            mask &= patient_tensor != int(patient_id)
        if patient_id is not None and visit_index is not None and exclude_exact_match:
            mask &= ~(
                (patient_tensor == int(patient_id))
                & (visit_index_tensor == int(visit_index))
            )
        if exclude_future:
            if patient_id is not None and visit_index is not None:
                same_patient_mask = patient_tensor == int(patient_id)
                mask &= ~(same_patient_mask & (visit_index_tensor > int(visit_index)))
            if (
                query_has_absolute_time
                and exclude_future_all_patients_if_absolute_time
                and visit_time is not None
            ):
                mask &= (~absolute_time_tensor) | (visit_time_tensor <= float(visit_time))
                if require_absolute_time_for_cross_patient_temporal_filter and patient_id is not None:
                    cross_patient_mask = patient_tensor != int(patient_id)
                    mask &= ~(cross_patient_mask & ~absolute_time_tensor)

        candidate_indices = torch.nonzero(mask, as_tuple=False).flatten()
        return {
            "indices": candidate_indices,
            "patient_ids": patient_tensor[candidate_indices],
            "visit_indices": visit_index_tensor[candidate_indices],
            "visit_times": visit_time_tensor[candidate_indices],
            "has_absolute_time": absolute_time_tensor[candidate_indices],
            "visit_embeddings": self.export_embeddings()[candidate_indices],
            "medication_evidence": self.export_medication_evidence()[candidate_indices],
            "metadata": [metadata["metadata"][int(index)] for index in candidate_indices.tolist()],
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "split_name": self.split_name,
            "time_is_absolute": self.time_is_absolute,
            "embedding_dim": self.embedding_dim,
            "medication_dim": self.medication_dim,
            "absolute_time_records": self._absolute_time_records,
            "records": [
                {
                    "patient_id": record.patient_id,
                    "visit_index": record.visit_index,
                    "visit_time": record.visit_time,
                    "has_absolute_time": record.has_absolute_time,
                    "visit_embedding": record.visit_embedding,
                    "medication_evidence": record.medication_evidence,
                    "metadata": record.metadata,
                }
                for record in self._records
            ],
        }
        torch.save(payload, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "VisitMemoryBank":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        bank = cls(
            split_name=payload.get("split_name"),
            time_is_absolute=bool(payload.get("time_is_absolute", False)),
        )
        bank.embedding_dim = payload.get("embedding_dim")
        bank.medication_dim = payload.get("medication_dim")
        bank._absolute_time_records = 0
        for item in payload.get("records", []):
            bank.add(
                patient_id=int(item["patient_id"]),
                visit_index=int(item["visit_index"]),
                visit_time=float(item["visit_time"]),
                has_absolute_time=bool(item.get("has_absolute_time", payload.get("time_is_absolute", False))),
                visit_embedding=torch.as_tensor(item["visit_embedding"], dtype=torch.float32),
                medication_evidence=torch.as_tensor(item["medication_evidence"], dtype=torch.float32),
                metadata=dict(item.get("metadata", {})),
            )
        return bank


__all__ = ["VisitMemoryBank", "VisitMemoryRecord"]
