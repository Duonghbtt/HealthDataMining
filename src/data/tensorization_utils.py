from __future__ import annotations

"""Shared tensorization helpers for the canonical patient-visit-prefix benchmark."""

from pathlib import Path
from typing import Any, Mapping

import torch

from src.utils.io import load_pt, parse_datetime, resolve_path

_STATIC_BATCH_TENSOR_KEYS = ("ddi_adj", "ddi_severity_adj")


def _ddi_payload_path_from_config(config: Mapping[str, Any]) -> Path | None:
    paths_cfg = dict(config.get("paths", {}))
    ddi_matrix_path = paths_cfg.get("ddi_matrix_path")
    if ddi_matrix_path:
        return Path(resolve_path(config["_project_root"], ddi_matrix_path))
    ddi_root = paths_cfg.get("ddi_root")
    if not ddi_root:
        return None
    return Path(resolve_path(config["_project_root"], ddi_root)) / "drug_ddi.pt"


def _normalize_ddi_tensor(
    value: Any,
    *,
    expected_drug_vocab_size: int,
    tensor_name: str,
) -> torch.Tensor:
    resolved = torch.as_tensor(value, dtype=torch.float32)
    if resolved.ndim != 2:
        raise ValueError(f"{tensor_name} must have shape (D, D), got {tuple(resolved.shape)}")
    if resolved.shape[0] != resolved.shape[1]:
        raise ValueError(f"{tensor_name} must be square, got {tuple(resolved.shape)}")
    if not torch.isfinite(resolved).all():
        raise ValueError(f"{tensor_name} must contain only finite values")
    if expected_drug_vocab_size > 0 and int(resolved.shape[0]) != int(expected_drug_vocab_size):
        raise ValueError(
            f"{tensor_name} width must match med vocab size {expected_drug_vocab_size}, "
            f"got {tuple(resolved.shape)}"
        )
    return resolved


def load_optional_ddi_tensors(
    config: Mapping[str, Any],
    *,
    expected_drug_vocab_size: int,
) -> dict[str, torch.Tensor]:
    """Load the optional benchmark DDI tensors and validate their shapes."""

    ddi_payload_path = _ddi_payload_path_from_config(config)
    if ddi_payload_path is None or not ddi_payload_path.exists():
        return {}

    payload = load_pt(ddi_payload_path)
    matrix_source = payload.get("matrix", payload) if isinstance(payload, Mapping) else payload
    ddi_adj = _normalize_ddi_tensor(
        matrix_source,
        expected_drug_vocab_size=expected_drug_vocab_size,
        tensor_name="ddi_adj",
    )
    ddi_adj = (ddi_adj > 0).to(dtype=torch.float32)
    ddi_adj = torch.maximum(ddi_adj, ddi_adj.transpose(0, 1))
    ddi_adj.fill_diagonal_(0.0)

    tensors: dict[str, torch.Tensor] = {"ddi_adj": ddi_adj}
    if isinstance(payload, Mapping) and payload.get("severity_matrix") is not None:
        severity_tensor = _normalize_ddi_tensor(
            payload["severity_matrix"],
            expected_drug_vocab_size=expected_drug_vocab_size,
            tensor_name="ddi_severity_adj",
        )
        severity_tensor = torch.maximum(severity_tensor, severity_tensor.transpose(0, 1))
        severity_tensor.fill_diagonal_(0.0)
        tensors["ddi_severity_adj"] = severity_tensor
    return tensors


def _infer_feature_size(
    record: Mapping[str, Any],
    *,
    steps: list[Mapping[str, Any]],
    field_name: str,
    default: int,
) -> int:
    if field_name in record:
        return int(record[field_name])
    value_field_name = field_name.replace("_feature_size", "_values")
    return int(
        max(
            (len(step.get(value_field_name, [])) for step in steps),
            default=default,
        )
    )


def augment_record(
    record: Mapping[str, Any],
    *,
    drug_vocab_size: int,
    default_lab_feature_size: int = 0,
    default_vital_feature_size: int = 0,
) -> dict[str, Any]:
    """Normalize one trajectory record before collation or tensor export."""

    resolved = dict(record)
    steps = [dict(step) for step in resolved.get("steps", [])]
    resolved["steps"] = steps
    resolved["drug_vocab_size"] = int(resolved.get("drug_vocab_size", drug_vocab_size))
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
    resolved["lab_feature_size"] = _infer_feature_size(
        resolved,
        steps=steps,
        field_name="lab_feature_size",
        default=default_lab_feature_size,
    )
    resolved["vital_feature_size"] = _infer_feature_size(
        resolved,
        steps=steps,
        field_name="vital_feature_size",
        default=default_vital_feature_size,
    )
    return resolved


def _resolve_shared_static_tensor(
    records: list[Mapping[str, Any]],
    key: str,
) -> torch.Tensor | None:
    resolved: torch.Tensor | None = None
    saw_value = False
    saw_missing = False
    for record in records:
        value = record.get(key)
        if value is None:
            saw_missing = True
            continue
        saw_value = True
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if resolved is None:
            resolved = tensor
            continue
        if tuple(tensor.shape) != tuple(resolved.shape):
            raise ValueError(
                f"Static batch tensor `{key}` must have one shared shape across records; "
                f"got {tuple(resolved.shape)} and {tuple(tensor.shape)}"
            )
    if saw_value and saw_missing:
        raise ValueError(f"Static batch tensor `{key}` must be present for either all records or none.")
    return resolved


def _assert_finite_tensor(name: str, value: torch.Tensor) -> None:
    if value.is_floating_point() or value.is_complex():
        if not torch.isfinite(value).all():
            raise ValueError(f"Batch tensor `{name}` contains non-finite values")


def _record_keys_for_error(record: Mapping[str, Any]) -> list[str]:
    return sorted(str(key) for key in record.keys())


def _resolve_record_int(
    record: Mapping[str, Any],
    *,
    candidate_keys: tuple[str, ...],
    field_label: str,
) -> int:
    for candidate_key in candidate_keys:
        if candidate_key not in record:
            continue
        value = record.get(candidate_key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Record field `{candidate_key}` for {field_label} must be int-like, "
                f"got {value!r}; keys={_record_keys_for_error(record)}"
            ) from exc
    raise KeyError(
        f"Missing {field_label} in record; checked {list(candidate_keys)}; "
        f"keys={_record_keys_for_error(record)}"
    )


def validate_collated_batch(batch: Mapping[str, Any]) -> None:
    """Check tensorized or freshly-collated batches for shape/value issues."""

    visit_mask = batch.get("visit_mask")
    if not isinstance(visit_mask, torch.Tensor):
        raise TypeError("Collated batch is missing tensor field `visit_mask`.")
    if visit_mask.ndim != 2:
        raise ValueError(f"visit_mask must have shape (B, T), got {tuple(visit_mask.shape)}")
    batch_size = int(visit_mask.shape[0])
    if bool((visit_mask.sum(dim=1) <= 0).any().item()):
        raise ValueError("Each batch element must contain at least one valid visit.")

    aligned_3d_keys = (
        "diag_codes",
        "diag_mask",
        "proc_codes",
        "proc_mask",
        "med_history",
        "med_history_mask",
        "lab_values",
        "lab_mask",
        "vital_values",
        "vital_mask",
        "target_drugs",
    )
    for key in aligned_3d_keys:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Collated batch is missing tensor field `{key}`.")
        if value.ndim != 3 or tuple(value.shape[:2]) != tuple(visit_mask.shape):
            raise ValueError(
                f"{key} must have shape (B, T, C/F) aligned with visit_mask, "
                f"got {tuple(value.shape)} and {tuple(visit_mask.shape)}"
            )
        _assert_finite_tensor(key, value)

    for ids_key in ("diag_codes", "proc_codes", "med_history"):
        ids_value = batch[ids_key]
        if bool((ids_value < 0).any().item()):
            raise ValueError(f"{ids_key} must contain only non-negative ids.")

    target_drugs = batch["target_drugs"]
    if target_drugs.numel() > 0:
        is_binary = bool(torch.logical_or(target_drugs == 0.0, target_drugs == 1.0).all().item())
        if not is_binary:
            raise ValueError("target_drugs must be a binary multi-hot tensor.")

    for key in ("time_delta_hours", "visit_time_absolute_hours"):
        value = batch.get(key)
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Collated batch field `{key}` must be a tensor when provided.")
        if value.ndim != 2 or tuple(value.shape) != tuple(visit_mask.shape):
            raise ValueError(
                f"{key} must have shape (B, T) aligned with visit_mask, "
                f"got {tuple(value.shape)} and {tuple(visit_mask.shape)}"
            )
        _assert_finite_tensor(key, value)

    for key in ("visit_time_absolute_mask",):
        value = batch.get(key)
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Collated batch field `{key}` must be a tensor when provided.")
        if value.ndim != 2 or tuple(value.shape) != tuple(visit_mask.shape):
            raise ValueError(
                f"{key} must have shape (B, T) aligned with visit_mask, "
                f"got {tuple(value.shape)} and {tuple(visit_mask.shape)}"
            )

    for key in _STATIC_BATCH_TENSOR_KEYS:
        value = batch.get(key)
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Static batch tensor `{key}` must be a torch.Tensor when provided.")
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError(f"{key} must have shape (D, D), got {tuple(value.shape)}")
        _assert_finite_tensor(key, value)
        if int(value.shape[0]) != int(target_drugs.shape[-1]):
            raise ValueError(
                f"{key} width must match target_drugs width {int(target_drugs.shape[-1])}, "
                f"got {tuple(value.shape)}"
            )

    for key in ("patient_ids", "subject_ids", "hadm_ids", "stay_ids", "visit_index", "visit_position", "history_length"):
        value = batch.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.ndim == 0 or int(value.shape[0]) != batch_size:
                raise ValueError(f"{key} must have leading batch dimension {batch_size}, got {tuple(value.shape)}")
            continue
        if len(value) != batch_size:
            raise ValueError(f"{key} must contain {batch_size} entries, got {len(value)}")


def _write_id_sequence(
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    batch_index: int,
    step_index: int,
    values: list[int] | tuple[int, ...],
) -> None:
    if not values:
        return
    value_tensor = torch.as_tensor(values, dtype=torch.long)
    sequence_length = int(value_tensor.numel())
    target[batch_index, step_index, :sequence_length] = value_tensor
    mask[batch_index, step_index, :sequence_length] = True


def _stack_dense_steps(
    steps: list[Mapping[str, Any]],
    *,
    field_name: str,
    expected_length: int,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if expected_length <= 0 or not steps:
        return None

    rows: list[Any] = []
    for step in steps:
        values = step.get(field_name)
        if values is None or len(values) != expected_length:
            return None
        rows.append(values)
    return torch.as_tensor(rows, dtype=dtype)


def _build_absolute_visit_times(
    *,
    record: Mapping[str, Any],
    steps: list[Mapping[str, Any]],
    max_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    absolute_hours = torch.zeros(max_steps, dtype=torch.float32)
    absolute_mask = torch.zeros(max_steps, dtype=torch.bool)
    if not steps:
        return absolute_hours, absolute_mask

    intime_value = record.get("intime")
    try:
        intime_dt = parse_datetime(None if intime_value is None else str(intime_value))
    except ValueError:
        intime_dt = None
    if intime_dt is None:
        return absolute_hours, absolute_mask

    # Absolute visit timestamps remain optional auxiliary chronology signals.
    base_hours = float(intime_dt.timestamp() / 3600.0)
    cumulative_hours = 0.0
    for step_index, step in enumerate(steps):
        if step_index > 0:
            cumulative_hours += float(step.get("delta_hours", 0.0))
        absolute_hours[step_index] = base_hours + cumulative_hours
        absolute_mask[step_index] = True
    return absolute_hours, absolute_mask


def collate_batch(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate canonical trajectory records into the dense batch API used by the model."""

    if not records:
        raise ValueError("collate_batch requires at least one record")

    batch_size = len(records)
    max_steps = 0
    max_diag_codes = 0
    max_proc_codes = 0
    max_history = 0
    drug_vocab_size = 0
    lab_feature_size = 0
    vital_feature_size = 0

    record_steps: list[list[Mapping[str, Any]]] = []
    patient_ids: list[int] = []
    subject_ids: list[int] = []
    hadm_ids: list[int] = []
    stay_ids: list[int] = []
    visit_indices: list[int] = []
    visit_positions: list[int] = []
    history_lengths: list[int] = []

    for record in records:
        steps = list(record.get("steps", []))
        num_steps = int(record.get("num_steps", len(steps)))
        if num_steps != len(steps):
            raise ValueError(
                "Trajectory record num_steps must equal len(steps); "
                f"got num_steps={record.get('num_steps')} len(steps)={len(steps)}"
            )
        record_steps.append(steps)
        patient_ids.append(
            _resolve_record_int(
                record,
                candidate_keys=("patient_id", "subject_id"),
                field_label="patient identifier (`patient_id` or `subject_id`)",
            )
        )
        subject_ids.append(
            _resolve_record_int(
                record,
                candidate_keys=("subject_id", "patient_id"),
                field_label="subject identifier (`subject_id` or `patient_id`)",
            )
        )
        hadm_ids.append(
            _resolve_record_int(
                record,
                candidate_keys=("hadm_id",),
                field_label="hospital admission identifier `hadm_id`",
            )
        )
        stay_ids.append(
            _resolve_record_int(
                record,
                candidate_keys=("stay_id",),
                field_label="ICU stay identifier `stay_id`",
            )
        )
        visit_indices.append(int(record["visit_index"]) if "visit_index" in record else max(num_steps - 1, 0))
        visit_positions.append(int(record["visit_position"]) if "visit_position" in record else num_steps)
        history_lengths.append(int(record["history_length"]) if "history_length" in record else num_steps)

        max_steps = max(max_steps, num_steps)
        drug_vocab_size = max(drug_vocab_size, int(record.get("drug_vocab_size", 0)))
        lab_feature_size = max(lab_feature_size, int(record.get("lab_feature_size", 0)))
        vital_feature_size = max(vital_feature_size, int(record.get("vital_feature_size", 0)))

        for step in steps:
            max_diag_codes = max(max_diag_codes, len(step.get("diagnosis_ids", ())))
            max_proc_codes = max(max_proc_codes, len(step.get("procedure_ids", ())))
            max_history = max(max_history, len(step.get("med_history_ids", ())))

    diag_codes = torch.zeros(batch_size, max_steps, max_diag_codes, dtype=torch.long)
    diag_mask = torch.zeros(batch_size, max_steps, max_diag_codes, dtype=torch.bool)
    proc_codes = torch.zeros(batch_size, max_steps, max_proc_codes, dtype=torch.long)
    proc_mask = torch.zeros(batch_size, max_steps, max_proc_codes, dtype=torch.bool)
    med_history = torch.zeros(batch_size, max_steps, max_history, dtype=torch.long)
    med_history_mask = torch.zeros(batch_size, max_steps, max_history, dtype=torch.bool)
    lab_values = torch.zeros(batch_size, max_steps, lab_feature_size, dtype=torch.float32)
    lab_mask = torch.zeros(batch_size, max_steps, lab_feature_size, dtype=torch.bool)
    vital_values = torch.zeros(batch_size, max_steps, vital_feature_size, dtype=torch.float32)
    vital_mask = torch.zeros(batch_size, max_steps, vital_feature_size, dtype=torch.bool)
    time_delta_hours = torch.zeros(batch_size, max_steps, dtype=torch.float32)
    visit_time_absolute_hours = torch.zeros(batch_size, max_steps, dtype=torch.float32)
    visit_time_absolute_mask = torch.zeros(batch_size, max_steps, dtype=torch.bool)
    visit_mask = torch.zeros(batch_size, max_steps, dtype=torch.bool)
    target_drugs = torch.zeros(batch_size, max_steps, drug_vocab_size, dtype=torch.float32)

    for batch_index, steps in enumerate(record_steps):
        num_steps = len(steps)
        if num_steps <= 0:
            continue

        visit_mask[batch_index, :num_steps] = True
        time_delta_hours[batch_index, :num_steps] = torch.as_tensor(
            [float(step.get("delta_hours", 0.0)) for step in steps],
            dtype=torch.float32,
        )
        absolute_hours, absolute_mask = _build_absolute_visit_times(
            record=records[batch_index],
            steps=steps,
            max_steps=max_steps,
        )
        visit_time_absolute_hours[batch_index] = absolute_hours
        visit_time_absolute_mask[batch_index] = absolute_mask

        lab_value_tensor = _stack_dense_steps(
            steps,
            field_name="lab_values",
            expected_length=lab_feature_size,
            dtype=torch.float32,
        )
        lab_mask_tensor = _stack_dense_steps(
            steps,
            field_name="lab_mask",
            expected_length=lab_feature_size,
            dtype=torch.bool,
        )
        vital_value_tensor = _stack_dense_steps(
            steps,
            field_name="vital_values",
            expected_length=vital_feature_size,
            dtype=torch.float32,
        )
        vital_mask_tensor = _stack_dense_steps(
            steps,
            field_name="vital_mask",
            expected_length=vital_feature_size,
            dtype=torch.bool,
        )

        if lab_value_tensor is not None:
            lab_values[batch_index, :num_steps, :lab_feature_size] = lab_value_tensor
        if lab_mask_tensor is not None:
            lab_mask[batch_index, :num_steps, :lab_feature_size] = lab_mask_tensor
        if vital_value_tensor is not None:
            vital_values[batch_index, :num_steps, :vital_feature_size] = vital_value_tensor
        if vital_mask_tensor is not None:
            vital_mask[batch_index, :num_steps, :vital_feature_size] = vital_mask_tensor

        target_step_indices: list[torch.Tensor] = []
        target_drug_indices: list[torch.Tensor] = []
        for step_index, step in enumerate(steps):
            _write_id_sequence(
                diag_codes,
                diag_mask,
                batch_index=batch_index,
                step_index=step_index,
                values=step.get("diagnosis_ids", ()),
            )
            _write_id_sequence(
                proc_codes,
                proc_mask,
                batch_index=batch_index,
                step_index=step_index,
                values=step.get("procedure_ids", ()),
            )
            _write_id_sequence(
                med_history,
                med_history_mask,
                batch_index=batch_index,
                step_index=step_index,
                values=step.get("med_history_ids", ()),
            )

            if lab_value_tensor is None and lab_feature_size:
                current_lab_values = step.get("lab_values", ())
                current_lab_mask = step.get("lab_mask", ())
                if current_lab_values:
                    current_lab_tensor = torch.as_tensor(current_lab_values, dtype=torch.float32)
                    lab_values[batch_index, step_index, : current_lab_tensor.numel()] = current_lab_tensor
                if current_lab_mask:
                    current_lab_mask_tensor = torch.as_tensor(current_lab_mask, dtype=torch.bool)
                    lab_mask[batch_index, step_index, : current_lab_mask_tensor.numel()] = current_lab_mask_tensor

            if vital_value_tensor is None and vital_feature_size:
                current_vital_values = step.get("vital_values", ())
                current_vital_mask = step.get("vital_mask", ())
                if current_vital_values:
                    current_vital_tensor = torch.as_tensor(current_vital_values, dtype=torch.float32)
                    vital_values[batch_index, step_index, : current_vital_tensor.numel()] = current_vital_tensor
                if current_vital_mask:
                    current_vital_mask_tensor = torch.as_tensor(current_vital_mask, dtype=torch.bool)
                    vital_mask[batch_index, step_index, : current_vital_mask_tensor.numel()] = current_vital_mask_tensor

            target_ids = step.get("target_drugs", ())
            if target_ids:
                target_tensor = torch.as_tensor(target_ids, dtype=torch.long)
                valid_mask = (target_tensor >= 0) & (target_tensor < drug_vocab_size)
                if torch.any(valid_mask):
                    valid_target_tensor = target_tensor[valid_mask]
                    target_step_indices.append(
                        torch.full(
                            (int(valid_target_tensor.numel()),),
                            step_index,
                            dtype=torch.long,
                        )
                    )
                    target_drug_indices.append(valid_target_tensor)

        if target_drug_indices:
            target_drugs[batch_index, torch.cat(target_step_indices), torch.cat(target_drug_indices)] = 1.0

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
        "visit_time_absolute_hours": visit_time_absolute_hours,
        "visit_time_absolute_mask": visit_time_absolute_mask,
        "visit_mask": visit_mask,
        "target_drugs": target_drugs,
        "patient_ids": patient_ids,
        "subject_ids": subject_ids,
        "hadm_ids": hadm_ids,
        "stay_ids": stay_ids,
        "visit_index": torch.as_tensor(visit_indices, dtype=torch.long),
        "visit_position": torch.as_tensor(visit_positions, dtype=torch.long),
        "history_length": torch.as_tensor(history_lengths, dtype=torch.long),
        "has_absolute_visit_time": visit_time_absolute_mask.any(dim=1),
    }
    for key in _STATIC_BATCH_TENSOR_KEYS:
        value = _resolve_shared_static_tensor(records, key)
        if value is not None:
            batch[key] = value
    validate_collated_batch(batch)
    return batch


_augment_record = augment_record
_load_optional_ddi_tensors = load_optional_ddi_tensors
_validate_collated_batch = validate_collated_batch


__all__ = [
    "augment_record",
    "collate_batch",
    "load_optional_ddi_tensors",
    "validate_collated_batch",
    "_augment_record",
    "_load_optional_ddi_tensors",
    "_validate_collated_batch",
]
