from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.features.diagnosis_encoder import DiagnosisEncoder
from src.features.lab_processor import LabFeatureEncoder
from src.features.medication_history import MedicationHistoryEncoder
from src.features.procedure_encoder import ProcedureEncoder
from src.features.vital_processor import VitalFeatureEncoder


def _tensor_debug_summary(name: str, value: torch.Tensor) -> str:
    resolved = value.detach()
    shape = tuple(resolved.shape)
    dtype = resolved.dtype
    if resolved.numel() == 0:
        return f"{name}: shape={shape} dtype={dtype} numel=0"
    if not (resolved.is_floating_point() or resolved.is_complex()):
        return (
            f"{name}: shape={shape} dtype={dtype} "
            f"min={resolved.min().item()} max={resolved.max().item()}"
        )
    nan_count = int(torch.isnan(resolved).sum().item())
    inf_count = int(torch.isinf(resolved).sum().item())
    finite = resolved[torch.isfinite(resolved)]
    if finite.numel() == 0:
        return (
            f"{name}: shape={shape} dtype={dtype} nan_count={nan_count} "
            f"inf_count={inf_count} finite_values=0"
        )
    return (
        f"{name}: shape={shape} dtype={dtype} nan_count={nan_count} "
        f"inf_count={inf_count} min={finite.min().item():.6g} "
        f"max={finite.max().item():.6g} mean={finite.mean().item():.6g}"
    )


def _assert_finite(name: str, value: torch.Tensor | None) -> None:
    if value is None or not isinstance(value, torch.Tensor):
        return
    if value.is_floating_point() or value.is_complex():
        if not torch.isfinite(value).all():
            raise ValueError(
                f"Non-finite tensor detected in patient_state_encoder at `{name}`; "
                f"{_tensor_debug_summary(name, value)}"
            )


def _should_debug_check(batch: Mapping[str, Any] | None) -> bool:
    if batch is None:
        return True
    return bool(batch.get("_debug_check_now", True))


def _optional_tensor(batch: Mapping[str, Any], key: str) -> torch.Tensor | None:
    value = batch.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Optional batch field `{key}` must be a torch.Tensor when provided.")
    return value


def _extract_last_valid_state(state_sequence: torch.Tensor, visit_mask: torch.Tensor) -> torch.Tensor:
    if state_sequence.ndim != 3:
        raise ValueError(f"state_sequence must have shape (B, T, H), got {tuple(state_sequence.shape)}")
    if visit_mask.ndim != 2:
        raise ValueError(f"visit_mask must have shape (B, T), got {tuple(visit_mask.shape)}")
    if tuple(state_sequence.shape[:2]) != tuple(visit_mask.shape):
        raise ValueError(
            "state_sequence and visit_mask must align on batch/time dimensions: "
            f"got {tuple(state_sequence.shape[:2])} and {tuple(visit_mask.shape)}"
        )

    valid_counts = visit_mask.to(dtype=torch.long).sum(dim=1).clamp(min=1)
    batch_indices = torch.arange(state_sequence.shape[0], device=state_sequence.device)
    return state_sequence[batch_indices, valid_counts - 1]


def _masked_sequence_average(
    values: torch.Tensor,
    visit_mask: torch.Tensor,
    *,
    debug_checks_enabled: bool = True,
) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError(f"Expected values with shape (B, T, H), got {tuple(values.shape)}")
    if tuple(values.shape[:2]) != tuple(visit_mask.shape):
        raise ValueError(
            "values and visit_mask must align on batch/time dimensions: "
            f"got {tuple(values.shape[:2])} and {tuple(visit_mask.shape)}"
        )
    weights = visit_mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    sanitized_values = torch.where(
        visit_mask.unsqueeze(-1),
        torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0),
        torch.zeros_like(values),
    )
    pooled = sanitized_values.sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    if debug_checks_enabled:
        _assert_finite("masked_sequence_average", pooled)
    return pooled


def _ensure_visit_mask(batch: Mapping[str, Any]) -> torch.Tensor:
    value = batch.get("visit_mask")
    if not isinstance(value, torch.Tensor):
        raise KeyError("Batch is missing tensor field `visit_mask`.")
    if value.ndim != 2:
        raise ValueError(f"visit_mask must have shape (B, T), got {tuple(value.shape)}")
    return value.to(dtype=torch.bool)


def _resolve_code_like_tensor(
    batch: Mapping[str, Any],
    key: str,
    *,
    visit_mask: torch.Tensor,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    value = batch.get(key)
    if value is None:
        return torch.zeros(
            visit_mask.shape[0],
            visit_mask.shape[1],
            1,
            device=visit_mask.device,
            dtype=dtype,
        )
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Batch field `{key}` must be a torch.Tensor when provided.")
    if value.ndim == 2 and int(visit_mask.shape[1]) == 1:
        value = value.unsqueeze(1)
    if value.ndim != 3 or tuple(value.shape[:2]) != tuple(visit_mask.shape):
        raise ValueError(
            f"{key} must have shape (B, T, C) aligned with visit_mask, got {tuple(value.shape)} and {tuple(visit_mask.shape)}"
        )
    return value.to(device=visit_mask.device, dtype=dtype if not value.dtype.is_floating_point else value.dtype)


def _resolve_numeric_tensor(
    batch: Mapping[str, Any],
    key: str,
    *,
    visit_mask: torch.Tensor,
    feature_size: int,
) -> torch.Tensor:
    value = batch.get(key)
    if value is None:
        return torch.zeros(
            visit_mask.shape[0],
            visit_mask.shape[1],
            int(feature_size),
            device=visit_mask.device,
            dtype=torch.float32,
        )
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Batch field `{key}` must be a torch.Tensor when provided.")
    if value.ndim == 2 and int(visit_mask.shape[1]) == 1:
        value = value.unsqueeze(1)
    if value.ndim != 3 or tuple(value.shape[:2]) != tuple(visit_mask.shape):
        raise ValueError(
            f"{key} must have shape (B, T, F) aligned with visit_mask, got {tuple(value.shape)} and {tuple(visit_mask.shape)}"
        )
    if int(value.shape[-1]) != int(feature_size):
        raise ValueError(
            f"{key} must have feature width {int(feature_size)}, got {int(value.shape[-1])}"
        )
    return value.to(device=visit_mask.device, dtype=torch.float32)


def _resolve_mask_tensor(
    batch: Mapping[str, Any],
    key: str,
    *,
    like: torch.Tensor,
    default_bool: torch.Tensor,
) -> torch.Tensor:
    value = batch.get(key)
    if value is None:
        return default_bool.to(device=like.device, dtype=torch.bool)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Mask field `{key}` must be a torch.Tensor when provided.")
    if tuple(value.shape) != tuple(like.shape):
        raise ValueError(f"{key} must match shape {tuple(like.shape)}, got {tuple(value.shape)}")
    return value.to(device=like.device, dtype=torch.bool)


class _ScalarGateFusion(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.gate = nn.Linear(self.input_dim, 1)
        self.output_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        modality_inputs: Mapping[str, torch.Tensor],
        *,
        debug_checks_enabled: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not modality_inputs:
            raise ValueError("modality_inputs must not be empty")
        modality_names = list(modality_inputs.keys())
        reference = modality_inputs[modality_names[0]]
        if reference.ndim != 3:
            raise ValueError(f"Expected modality inputs with shape (B, T, H), got {tuple(reference.shape)}")
        for name in modality_names[1:]:
            if tuple(modality_inputs[name].shape) != tuple(reference.shape):
                raise ValueError(
                    f"All modality inputs must share the same shape, got {tuple(reference.shape)} and {tuple(modality_inputs[name].shape)}"
                )
        stacked = torch.stack([modality_inputs[name] for name in modality_names], dim=2)
        if debug_checks_enabled:
            _assert_finite("scalar_gate.stacked_inputs", stacked)
        gate_logits = self.gate(stacked).squeeze(-1)
        if debug_checks_enabled:
            _assert_finite("scalar_gate.gate_logits_preclamp", gate_logits)
        gate_logits = gate_logits.clamp(min=-30.0, max=30.0)
        gate_weights = torch.softmax(gate_logits, dim=2)
        if debug_checks_enabled:
            _assert_finite("scalar_gate.gate_weights", gate_weights)
        fused_pre_projection = (gate_weights.unsqueeze(-1) * stacked).sum(dim=2)
        if debug_checks_enabled:
            _assert_finite("scalar_gate.fused_pre_projection", fused_pre_projection)
        fused = self.output_projection(fused_pre_projection)
        if debug_checks_enabled:
            _assert_finite("scalar_gate.fused_output", fused)
        return fused, {
            name: gate_weights[:, :, index]
            for index, name in enumerate(modality_names)
        }


class _LightTemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, *, num_heads: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=int(hidden_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(
        self,
        state_sequence: torch.Tensor,
        visit_mask: torch.Tensor,
        *,
        debug_checks_enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if debug_checks_enabled:
            _assert_finite("temporal_attention.input_state_sequence", state_sequence)
        resolved_mask = visit_mask.to(device=state_sequence.device, dtype=torch.bool)
        valid_counts = resolved_mask.sum(dim=1)
        safe_mask = resolved_mask.clone()
        empty_rows = valid_counts <= 0
        single_visit_rows = valid_counts == 1
        multi_visit_rows = valid_counts > 1
        if bool(empty_rows.any().item()):
            # MultiheadAttention can emit NaNs when every position is masked.
            # Keep one dummy slot open, then zero the outputs back out afterward.
            safe_mask[empty_rows, 0] = True
        # Returning attention weights during training can force a slower and less
        # stable MultiheadAttention backward path. Keep weights only for eval/debug.
        request_attention_weights = not self.training
        attn_output, attn_weights = self.attention(
            state_sequence,
            state_sequence,
            state_sequence,
            key_padding_mask=~safe_mask,
            need_weights=request_attention_weights,
        )
        if debug_checks_enabled:
            _assert_finite("temporal_attention.raw_output", attn_output)
        if attn_weights is None:
            attn_weights = torch.zeros(
                state_sequence.shape[0],
                state_sequence.shape[1],
                state_sequence.shape[1],
                device=state_sequence.device,
                dtype=state_sequence.dtype,
            )
        if debug_checks_enabled:
            _assert_finite("temporal_attention.raw_weights", attn_weights)
        attn_output = torch.where(
            multi_visit_rows.view(-1, 1, 1),
            attn_output,
            torch.zeros_like(attn_output),
        )
        attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=1.0e4, neginf=-1.0e4).clamp(
            min=-1.0e4,
            max=1.0e4,
        )
        if debug_checks_enabled:
            _assert_finite("temporal_attention.stabilized_output", attn_output)
        residual = state_sequence + self.dropout(attn_output)
        residual = torch.nan_to_num(residual, nan=0.0, posinf=1.0e4, neginf=-1.0e4).clamp(
            min=-1.0e4,
            max=1.0e4,
        )
        if debug_checks_enabled:
            _assert_finite("temporal_attention.residual_pre_norm", residual)
        attended = self.norm(residual)
        if debug_checks_enabled:
            _assert_finite("temporal_attention.post_norm", attended)
        attended = torch.where(single_visit_rows.view(-1, 1, 1), state_sequence, attended)
        attended = attended * resolved_mask.unsqueeze(-1).to(dtype=attended.dtype)
        attn_weights = (
            torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
            * resolved_mask.unsqueeze(-1).to(dtype=attn_weights.dtype)
            * safe_mask.unsqueeze(1).to(dtype=attn_weights.dtype)
        )
        if bool(single_visit_rows.any().item()):
            eye = torch.eye(
                resolved_mask.shape[1],
                device=resolved_mask.device,
                dtype=attn_weights.dtype,
            ).unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
            attn_weights = torch.where(single_visit_rows.view(-1, 1, 1), eye, attn_weights)
        if bool(empty_rows.any().item()):
            attended = attended.masked_fill(empty_rows.view(-1, 1, 1), 0.0)
            attn_weights = attn_weights.masked_fill(empty_rows.view(-1, 1, 1), 0.0)
        attn_weights = attn_weights * resolved_mask.unsqueeze(-1).to(dtype=attn_weights.dtype)
        attn_weights = attn_weights * resolved_mask.unsqueeze(1).to(dtype=attn_weights.dtype)
        weight_denom = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        attn_weights = torch.where(
            resolved_mask.unsqueeze(1),
            attn_weights / weight_denom,
            torch.zeros_like(attn_weights),
        )
        if debug_checks_enabled:
            _assert_finite("temporal_attention.attended", attended)
            _assert_finite("temporal_attention.weights", attn_weights)
        return attended, attn_weights


class PatientStateEncoder(nn.Module):
    def __init__(
        self,
        diagnosis_vocab_size: int,
        procedure_vocab_size: int,
        drug_vocab_size: int,
        num_lab_features: int,
        num_vital_features: int,
        *,
        code_embedding_dim: int = 64,
        medication_embedding_dim: int = 64,
        numeric_projection_dim: int = 32,
        time_embedding_dim: int = 32,
        visit_hidden_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        encoder_mode: str = "legacy_gru",
        modality_hidden_dim: int | None = None,
        fusion_hidden_dim: int | None = None,
        modality_dropout: float | None = None,
        use_temporal_attention: bool = True,
        temporal_attention_heads: int = 1,
        temporal_attention_dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.encoder_mode = str(encoder_mode).strip().lower()
        if self.encoder_mode not in {"legacy_gru", "modality_aware_gru"}:
            raise ValueError(
                f"encoder_mode must be 'legacy_gru' or 'modality_aware_gru', got {encoder_mode!r}"
            )
        self.modality_hidden_dim = int(modality_hidden_dim if modality_hidden_dim is not None else hidden_dim)
        self.fusion_hidden_dim = int(fusion_hidden_dim if fusion_hidden_dim is not None else visit_hidden_dim)
        self.visit_hidden_dim = self.fusion_hidden_dim
        self.num_lab_features = int(num_lab_features)
        self.num_vital_features = int(num_vital_features)
        self.use_temporal_attention = bool(use_temporal_attention)
        if self.hidden_dim % int(temporal_attention_heads) != 0:
            raise ValueError(
                "hidden_dim must be divisible by temporal_attention_heads: "
                f"got hidden_dim={self.hidden_dim} and heads={int(temporal_attention_heads)}"
            )

        resolved_modality_dropout = float(dropout if modality_dropout is None else modality_dropout)
        resolved_attention_dropout = float(
            dropout if temporal_attention_dropout is None else temporal_attention_dropout
        )

        self.diagnosis_encoder = DiagnosisEncoder(
            diagnosis_vocab_size,
            code_embedding_dim,
            output_dim=self.modality_hidden_dim,
            padding_idx=0,
            dropout=resolved_modality_dropout,
        )
        self.procedure_encoder = ProcedureEncoder(
            procedure_vocab_size,
            code_embedding_dim,
            output_dim=self.modality_hidden_dim,
            padding_idx=0,
            dropout=resolved_modality_dropout,
        )
        self.medication_history_encoder = MedicationHistoryEncoder(
            drug_vocab_size,
            medication_embedding_dim,
            output_dim=self.modality_hidden_dim,
            padding_idx=0,
            dropout=resolved_modality_dropout,
        )
        numeric_hidden_dim = max(int(numeric_projection_dim), self.modality_hidden_dim)
        self.lab_encoder = LabFeatureEncoder(
            num_lab_features,
            self.modality_hidden_dim,
            hidden_dim=numeric_hidden_dim,
            dropout=resolved_modality_dropout,
        )
        self.vital_encoder = VitalFeatureEncoder(
            num_vital_features,
            self.modality_hidden_dim,
            hidden_dim=numeric_hidden_dim,
            dropout=resolved_modality_dropout,
        )
        self.time_projection = nn.Sequential(
            nn.Linear(1, int(time_embedding_dim)),
            nn.ReLU(),
            nn.Dropout(resolved_modality_dropout),
        )

        legacy_input_dim = self.modality_hidden_dim * 5 + int(time_embedding_dim)
        self.legacy_visit_projection = nn.Sequential(
            nn.Linear(legacy_input_dim, self.fusion_hidden_dim),
            nn.LayerNorm(self.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.numeric_fusion = _ScalarGateFusion(
            self.modality_hidden_dim,
            self.modality_hidden_dim,
            dropout=resolved_modality_dropout,
        )
        self.modality_fusion = _ScalarGateFusion(
            self.modality_hidden_dim,
            self.modality_hidden_dim,
            dropout=resolved_modality_dropout,
        )
        self.modality_visit_projection = nn.Sequential(
            nn.Linear(self.modality_hidden_dim + int(time_embedding_dim), self.fusion_hidden_dim),
            nn.LayerNorm(self.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.gru = nn.GRU(self.fusion_hidden_dim, self.hidden_dim, batch_first=True)
        # Keep a hard on/off switch so we can isolate temporal attention during
        # debugging without changing the rest of the encoder path.
        self.temporal_attention = (
            _LightTemporalAttention(
                self.hidden_dim,
                num_heads=int(temporal_attention_heads),
                dropout=resolved_attention_dropout,
            )
            if self.use_temporal_attention
            else None
        )

    def _encode_modalities(
        self,
        batch: Mapping[str, Any],
        visit_mask: torch.Tensor,
        *,
        debug_checks_enabled: bool,
    ) -> dict[str, torch.Tensor]:
        diag_codes = _resolve_code_like_tensor(batch, "diag_codes", visit_mask=visit_mask)
        proc_codes = _resolve_code_like_tensor(batch, "proc_codes", visit_mask=visit_mask)
        med_history = _resolve_code_like_tensor(batch, "med_history", visit_mask=visit_mask)
        lab_values = _resolve_numeric_tensor(
            batch,
            "lab_values",
            visit_mask=visit_mask,
            feature_size=self.num_lab_features,
        )
        vital_values = _resolve_numeric_tensor(
            batch,
            "vital_values",
            visit_mask=visit_mask,
            feature_size=self.num_vital_features,
        )
        time_delta_hours = _optional_tensor(batch, "time_delta_hours")
        if time_delta_hours is None:
            time_delta_hours = torch.zeros(
                visit_mask.shape[0],
                visit_mask.shape[1],
                device=visit_mask.device,
                dtype=torch.float32,
            )
        if time_delta_hours.ndim == 1 and int(visit_mask.shape[1]) == 1:
            time_delta_hours = time_delta_hours.unsqueeze(1)
        if tuple(time_delta_hours.shape) != tuple(visit_mask.shape):
            raise ValueError(
                "time_delta_hours must align with visit_mask on batch/time dimensions: "
                f"got {tuple(time_delta_hours.shape)} and {tuple(visit_mask.shape)}"
            )
        time_delta_hours = torch.nan_to_num(
            time_delta_hours.to(device=visit_mask.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        diag_mask = _resolve_mask_tensor(batch, "diag_mask", like=diag_codes, default_bool=diag_codes.ne(0))
        proc_mask = _resolve_mask_tensor(batch, "proc_mask", like=proc_codes, default_bool=proc_codes.ne(0))
        med_history_mask = _resolve_mask_tensor(
            batch,
            "med_history_mask",
            like=med_history,
            default_bool=med_history.ne(0),
        )
        lab_mask = _resolve_mask_tensor(
            batch,
            "lab_mask",
            like=lab_values,
            default_bool=torch.isfinite(lab_values),
        )
        vital_mask = _resolve_mask_tensor(
            batch,
            "vital_mask",
            like=vital_values,
            default_bool=torch.isfinite(vital_values),
        )
        lab_values = torch.where(
            lab_mask,
            torch.nan_to_num(lab_values, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(lab_values),
        )
        vital_values = torch.where(
            vital_mask,
            torch.nan_to_num(vital_values, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(vital_values),
        )
        if debug_checks_enabled:
            _assert_finite("lab_values_sanitized", lab_values)
            _assert_finite("vital_values_sanitized", vital_values)
            _assert_finite("time_delta_hours_sanitized", time_delta_hours)

        diag_repr = self.diagnosis_encoder(diag_codes, diag_mask)
        proc_repr = self.procedure_encoder(proc_codes, proc_mask)
        med_repr = self.medication_history_encoder(med_history, med_history_mask)
        lab_repr = self.lab_encoder(lab_values, lab_mask)
        vital_repr = self.vital_encoder(vital_values, vital_mask)
        time_repr = self.time_projection(torch.log1p(torch.clamp_min(time_delta_hours, 0.0)).unsqueeze(-1))
        if debug_checks_enabled:
            _assert_finite("diag_repr", diag_repr)
            _assert_finite("proc_repr", proc_repr)
            _assert_finite("med_repr", med_repr)
            _assert_finite("lab_repr", lab_repr)
            _assert_finite("vital_repr", vital_repr)
            _assert_finite("time_repr", time_repr)
        return {
            "diag": diag_repr,
            "proc": proc_repr,
            "med_history": med_repr,
            "lab": lab_repr,
            "vital": vital_repr,
            "time": time_repr,
        }

    def _encode_visits(
        self,
        modality_embeddings: Mapping[str, torch.Tensor],
        visit_mask: torch.Tensor,
        *,
        debug_checks_enabled: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        _ = visit_mask
        diag_repr = modality_embeddings["diag"]
        proc_repr = modality_embeddings["proc"]
        med_repr = modality_embeddings["med_history"]
        lab_repr = modality_embeddings["lab"]
        vital_repr = modality_embeddings["vital"]
        time_repr = modality_embeddings["time"]

        if self.encoder_mode == "legacy_gru":
            legacy_visit_input = torch.cat([diag_repr, proc_repr, lab_repr, vital_repr, med_repr, time_repr], dim=-1)
            if debug_checks_enabled:
                _assert_finite("legacy.visit_input", legacy_visit_input)
            visit_repr = self.legacy_visit_projection(legacy_visit_input)
            zeros_numeric_gates = {
                "lab": torch.zeros(visit_mask.shape, device=visit_mask.device, dtype=visit_repr.dtype),
                "vital": torch.zeros(visit_mask.shape, device=visit_mask.device, dtype=visit_repr.dtype),
            }
            zeros_modality_gates = {
                "diagnosis": torch.zeros(visit_mask.shape, device=visit_mask.device, dtype=visit_repr.dtype),
                "procedure": torch.zeros(visit_mask.shape, device=visit_mask.device, dtype=visit_repr.dtype),
                "lab_vital": torch.zeros(visit_mask.shape, device=visit_mask.device, dtype=visit_repr.dtype),
                "med_history": torch.zeros(visit_mask.shape, device=visit_mask.device, dtype=visit_repr.dtype),
            }
            debug = {
                "lab_vital": 0.5 * (lab_repr + vital_repr),
                "numeric_gate_weights": zeros_numeric_gates,
                "modality_gate_weights": zeros_modality_gates,
            }
            if debug_checks_enabled:
                _assert_finite("legacy.visit_repr", visit_repr)
            return visit_repr, debug

        lab_vital_repr, numeric_gate_weights = self.numeric_fusion(
            {"lab": lab_repr, "vital": vital_repr},
            debug_checks_enabled=debug_checks_enabled,
        )
        if debug_checks_enabled:
            _assert_finite("modality.numeric_fusion_output", lab_vital_repr)
            _assert_finite("modality.numeric_gate_weights.lab", numeric_gate_weights["lab"])
            _assert_finite("modality.numeric_gate_weights.vital", numeric_gate_weights["vital"])
        fused_modalities, modality_gate_weights = self.modality_fusion(
            {
                "diagnosis": diag_repr,
                "procedure": proc_repr,
                "lab_vital": lab_vital_repr,
                "med_history": med_repr,
            },
            debug_checks_enabled=debug_checks_enabled,
        )
        if debug_checks_enabled:
            _assert_finite("modality.modality_fusion_output", fused_modalities)
            for name, weights in modality_gate_weights.items():
                _assert_finite(f"modality.modality_gate_weights.{name}", weights)
        modality_visit_input = torch.cat([fused_modalities, time_repr], dim=-1)
        if debug_checks_enabled:
            _assert_finite("modality.visit_projection_input", modality_visit_input)
        visit_repr = self.modality_visit_projection(modality_visit_input)
        if debug_checks_enabled:
            _assert_finite("modality.lab_vital_repr", lab_vital_repr)
            _assert_finite("modality.fused_modalities", fused_modalities)
            _assert_finite("modality.visit_repr", visit_repr)
        debug = {
            "lab_vital": lab_vital_repr,
            "numeric_gate_weights": numeric_gate_weights,
            "modality_gate_weights": modality_gate_weights,
        }
        return visit_repr, debug

    def _encode_temporal(
        self,
        visit_repr: torch.Tensor,
        visit_mask: torch.Tensor,
        *,
        debug_checks_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        valid_lengths = visit_mask.sum(dim=-1).clamp(min=1).to(dtype=torch.long)
        packed = pack_padded_sequence(
            visit_repr,
            valid_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, hidden = self.gru(packed)
        gru_state_sequence, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=visit_repr.shape[1],
        )
        gru_state_sequence = gru_state_sequence * visit_mask.unsqueeze(-1).to(dtype=gru_state_sequence.dtype)
        if debug_checks_enabled:
            _assert_finite("temporal.gru_input", visit_repr)
            _assert_finite("temporal.gru_state_sequence", gru_state_sequence)
            _assert_finite("temporal.hidden", hidden[-1])

        temporal_attention_weights: torch.Tensor | None = None
        if self.encoder_mode == "modality_aware_gru" and self.temporal_attention is not None:
            state_sequence, temporal_attention_weights = self.temporal_attention(
                gru_state_sequence,
                visit_mask,
                debug_checks_enabled=debug_checks_enabled,
            )
            temporal_attention_weights = temporal_attention_weights * visit_mask.unsqueeze(-1).to(
                dtype=state_sequence.dtype
            )
        else:
            state_sequence = gru_state_sequence
        if debug_checks_enabled:
            _assert_finite("temporal.state_sequence", state_sequence)
            _assert_finite("temporal_attention_weights", temporal_attention_weights)
        pooled_state = hidden[-1]
        return state_sequence, pooled_state, temporal_attention_weights

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        debug_checks_enabled = _should_debug_check(batch)
        visit_mask = _ensure_visit_mask(batch)
        modality_embeddings = self._encode_modalities(
            batch,
            visit_mask,
            debug_checks_enabled=debug_checks_enabled,
        )
        visit_repr, fusion_debug = self._encode_visits(
            modality_embeddings,
            visit_mask,
            debug_checks_enabled=debug_checks_enabled,
        )
        state_sequence, pooled_state, temporal_attention_weights = self._encode_temporal(
            visit_repr,
            visit_mask,
            debug_checks_enabled=debug_checks_enabled,
        )
        current_state = _extract_last_valid_state(state_sequence, visit_mask)
        pooled_state = current_state

        modality_summary = {
            "diagnosis": _masked_sequence_average(
                modality_embeddings["diag"],
                visit_mask,
                debug_checks_enabled=debug_checks_enabled,
            ),
            "procedure": _masked_sequence_average(
                modality_embeddings["proc"],
                visit_mask,
                debug_checks_enabled=debug_checks_enabled,
            ),
            "lab": _masked_sequence_average(
                modality_embeddings["lab"],
                visit_mask,
                debug_checks_enabled=debug_checks_enabled,
            ),
            "vital": _masked_sequence_average(
                modality_embeddings["vital"],
                visit_mask,
                debug_checks_enabled=debug_checks_enabled,
            ),
            "lab_vital": _masked_sequence_average(
                fusion_debug["lab_vital"],
                visit_mask,
                debug_checks_enabled=debug_checks_enabled,
            ),
            "med_history": _masked_sequence_average(
                modality_embeddings["med_history"],
                visit_mask,
                debug_checks_enabled=debug_checks_enabled,
            ),
            "time": _masked_sequence_average(
                modality_embeddings["time"],
                visit_mask,
                debug_checks_enabled=debug_checks_enabled,
            ),
        }
        modality_gate_weights = torch.stack(
            [fusion_debug["modality_gate_weights"][name] for name in ("diagnosis", "procedure", "lab_vital", "med_history")],
            dim=-1,
        )
        numeric_gate_weights = torch.stack(
            [fusion_debug["numeric_gate_weights"][name] for name in ("lab", "vital")],
            dim=-1,
        )
        if debug_checks_enabled:
            _assert_finite("forward.visit_repr", visit_repr)
            _assert_finite("forward.state_sequence", state_sequence)
            _assert_finite("forward.current_state", current_state)
            _assert_finite("forward.modality_gate_weights", modality_gate_weights)
            _assert_finite("forward.numeric_gate_weights", numeric_gate_weights)
            for name, summary in modality_summary.items():
                _assert_finite(f"forward.modality_summary.{name}", summary)

        if tuple(state_sequence.shape[:2]) != tuple(visit_mask.shape):
            raise AssertionError("state_sequence batch/time shape must match visit_mask")
        if state_sequence.shape[-1] != self.hidden_dim:
            raise AssertionError("state_sequence hidden width does not match encoder hidden_dim")
        if visit_repr.shape[-1] != self.fusion_hidden_dim:
            raise AssertionError("visit_repr width does not match fusion_hidden_dim")

        return {
            "visit_repr": visit_repr,                     # [B, T, V]
            "state_sequence": state_sequence,             # [B, T, H]
            "history_states": state_sequence,             # [B, T, H]
            "pooled_state": pooled_state,                 # [B, H]
            "current_state": current_state,               # [B, H]
            "visit_mask": visit_mask,                     # [B, T]
            "modality_summary": modality_summary,         # dict[str, Tensor]
            "modality_history_states": {
                "diagnosis": modality_embeddings["diag"],
                "procedure": modality_embeddings["proc"],
                "lab_vital": fusion_debug["lab_vital"],
                "medication_history": modality_embeddings["med_history"],
            },
            "modality_gate_weights": modality_gate_weights,   # [B, T, 4]
            "numeric_gate_weights": numeric_gate_weights,     # [B, T, 2]
            "temporal_attention_weights": temporal_attention_weights,
            "encoder_mode": self.encoder_mode,
        }
