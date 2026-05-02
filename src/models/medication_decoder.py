from __future__ import annotations

from typing import Any

import torch
from torch import nn

_VALID_DECODER_MODES = {"legacy", "copy_reuse_v2"}
_VALID_GATE_TYPES = {"scalar", "drug_wise"}
_VALID_COPY_PROJECTIONS = {"none", "linear", "mlp_light"}


def _validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _build_activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported decoder activation: {name!r}")


def _resolve_decoder_mode(*, decoder_mode: str | None, decoder_type: str) -> tuple[str, str]:
    normalized_type = str(decoder_type).strip().lower()
    normalized_mode = None if decoder_mode is None else str(decoder_mode).strip().lower()
    if normalized_mode is None:
        if normalized_type in _VALID_DECODER_MODES:
            return normalized_type, "residual_mlp"
        return "legacy", normalized_type
    if normalized_mode not in _VALID_DECODER_MODES:
        raise ValueError(f"decoder_mode must be one of {_VALID_DECODER_MODES}, got {decoder_mode!r}")
    return normalized_mode, normalized_type


def _require_2d_tensor(
    *,
    name: str,
    value: torch.Tensor | None,
    batch_size: int,
    feature_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch_size, feature_dim, device=device, dtype=dtype)
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape (B, F), got {tuple(tensor.shape)}")
    if tuple(tensor.shape) != (batch_size, feature_dim):
        raise ValueError(
            f"{name} must have shape {(batch_size, feature_dim)}, got {tuple(tensor.shape)}"
        )
    return tensor


def _safe_logit(probabilities: torch.Tensor, eps: float = 1.0e-4) -> torch.Tensor:
    clamped = probabilities.clamp(min=eps, max=1.0 - eps)
    return torch.log(clamped) - torch.log1p(-clamped)


class MedicationDecoder(nn.Module):
    """Decode medication logits from a fused state, optionally with copy/reuse.

    Outputs
    -------
    drug_logits:
        Raw pre-sigmoid scores used as input to ``BCEWithLogitsLoss``.
    drug_probs:
        ``torch.sigmoid(drug_logits)`` in ``[0, 1]``, used for metrics and thresholding.
    logits_new:
        Fresh medication logits predicted from the fused hidden representation.
    logits_copy:
        Copy/reuse logits derived from history/retrieval medication evidence.
    gate:
        Decoder mixing gate after broadcasting to drug space.
    """

    def __init__(
        self,
        hidden_dim: int,
        drug_vocab_size: int,
        *,
        dropout: float = 0.1,
        hidden_multiplier: int = 2,
        activation: str = "relu",
        layer_norm: bool = True,
        decoder_type: str = "residual_mlp",
        decoder_mode: str | None = None,
        gate_type: str = "scalar",
        use_history_copy: bool = True,
        use_retrieval_copy: bool = True,
        use_memory_copy: bool = False,
        copy_projection: str = "none",
        gate_hidden_dim: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.hidden_dim = _validate_positive_int("hidden_dim", int(hidden_dim))
        self.drug_vocab_size = _validate_positive_int("drug_vocab_size", int(drug_vocab_size))
        self.hidden_multiplier = _validate_positive_int("hidden_multiplier", int(hidden_multiplier))
        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout!r}")
        self.decoder_mode, normalized_decoder_type = _resolve_decoder_mode(
            decoder_mode=decoder_mode,
            decoder_type=decoder_type,
        )
        if normalized_decoder_type != "residual_mlp":
            raise ValueError("MedicationDecoder currently supports only decoder_type='residual_mlp'.")
        self.gate_type = str(gate_type).strip().lower()
        if self.gate_type not in _VALID_GATE_TYPES:
            raise ValueError(f"gate_type must be one of {_VALID_GATE_TYPES}, got {gate_type!r}")
        self.copy_projection_mode = str(copy_projection).strip().lower()
        if self.copy_projection_mode not in _VALID_COPY_PROJECTIONS:
            raise ValueError(
                f"copy_projection must be one of {_VALID_COPY_PROJECTIONS}, got {copy_projection!r}"
            )
        self.use_history_copy = bool(use_history_copy)
        self.use_retrieval_copy = bool(use_retrieval_copy)
        self.use_memory_copy = bool(use_memory_copy)

        expanded_dim = self.hidden_dim * self.hidden_multiplier
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, expanded_dim),
            nn.LayerNorm(expanded_dim) if bool(layer_norm) else nn.Identity(),
            _build_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(expanded_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim) if bool(layer_norm) else nn.Identity(),
            _build_activation(activation),
            nn.Dropout(float(dropout)),
        )
        self.fc = nn.Linear(self.hidden_dim, self.drug_vocab_size)
        self.residual_fc = nn.Linear(self.hidden_dim, self.drug_vocab_size)
        gate_width = _validate_positive_int(
            "gate_hidden_dim",
            int(self.hidden_dim if gate_hidden_dim is None else gate_hidden_dim),
        )
        gate_output_dim = 1 if self.gate_type == "scalar" else self.drug_vocab_size
        self.copy_gate_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, gate_width),
            _build_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(gate_width, gate_output_dim),
        )
        self.copy_source_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, gate_width),
            _build_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(gate_width, 3),
        )
        if self.copy_projection_mode == "none":
            self.copy_projection = None
        elif self.copy_projection_mode == "linear":
            self.copy_projection = nn.Linear(self.drug_vocab_size, self.drug_vocab_size)
        else:
            self.copy_projection = nn.Sequential(
                nn.Linear(self.drug_vocab_size, self.drug_vocab_size),
                _build_activation(activation),
                nn.Dropout(float(dropout)),
                nn.Linear(self.drug_vocab_size, self.drug_vocab_size),
            )

    def _compute_logits_new(self, context_vector: torch.Tensor) -> torch.Tensor:
        x = self.proj(context_vector)                            # [B, H] main nonlinear decoder path
        main_logits = self.fc(x)                                 # [B, D]
        residual_logits = self.residual_fc(context_vector)       # [B, D] direct skip path from fused state
        return main_logits + residual_logits                     # [B, D] raw pre-sigmoid logits

    def _build_copy_signal(
        self,
        *,
        current_state: torch.Tensor,
        history_med_bag: torch.Tensor,
        retrieval_med_bag: torch.Tensor,
        medication_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(current_state.shape[0])
        device = current_state.device
        source_tensors = torch.stack(
            (history_med_bag, retrieval_med_bag, medication_memory),
            dim=1,
        )  # [B, 3, D]
        source_mask = torch.stack(
            (
                (history_med_bag.abs().sum(dim=1) > 0)
                if self.use_history_copy
                else torch.zeros(batch_size, device=device, dtype=torch.bool),
                (retrieval_med_bag.abs().sum(dim=1) > 0)
                if self.use_retrieval_copy
                else torch.zeros(batch_size, device=device, dtype=torch.bool),
                (medication_memory.abs().sum(dim=1) > 0)
                if self.use_memory_copy
                else torch.zeros(batch_size, device=device, dtype=torch.bool),
            ),
            dim=1,
        )  # [B, 3]
        source_logits = self.copy_source_gate(current_state)
        masked_logits = source_logits.masked_fill(~source_mask, -1.0e9)
        source_weights = torch.softmax(masked_logits, dim=-1)
        any_source = source_mask.any(dim=1, keepdim=True)
        source_weights = torch.where(any_source, source_weights, torch.zeros_like(source_weights))
        copy_signal = torch.sum(source_weights.unsqueeze(-1) * source_tensors, dim=1)
        return copy_signal.clamp(min=0.0, max=1.0), source_weights, source_mask

    def _compute_logits_copy(self, copy_signal: torch.Tensor) -> torch.Tensor:
        if self.copy_projection is None:
            return _safe_logit(copy_signal)
        return self.copy_projection(copy_signal)

    def forward(
        self,
        context_vector: torch.Tensor,
        *,
        current_state: torch.Tensor | None = None,
        history_context: torch.Tensor | None = None,
        retrieval_context: torch.Tensor | None = None,
        history_med_bag: torch.Tensor | None = None,
        retrieval_med_bag: torch.Tensor | None = None,
        medication_memory: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Return raw medication logits plus copy/reuse decoder auxiliaries."""

        if not isinstance(context_vector, torch.Tensor):
            raise TypeError(f"context_vector must be a torch.Tensor, got {type(context_vector)!r}")
        if context_vector.ndim != 2:
            raise ValueError(f"context_vector must have shape (B, H), got {tuple(context_vector.shape)}")
        if context_vector.shape[1] != self.hidden_dim:
            raise ValueError(
                "context_vector hidden dimension mismatch: "
                f"expected {self.hidden_dim}, got {int(context_vector.shape[1])}"
            )

        batch_size = int(context_vector.shape[0])
        device = context_vector.device
        dtype = context_vector.dtype
        logits_new = self._compute_logits_new(context_vector)

        if self.decoder_mode == "legacy":
            gate_raw = torch.ones(batch_size, 1, device=device, dtype=dtype)
            gate = gate_raw.expand(-1, self.drug_vocab_size)
            logits_copy = torch.zeros_like(logits_new)
            copy_signal = torch.zeros_like(logits_new)
            copy_source_weights = torch.zeros(batch_size, 3, device=device, dtype=dtype)
            copy_source_mask = torch.zeros(batch_size, 3, device=device, dtype=torch.bool)
            drug_logits = logits_new
            drug_probs = torch.sigmoid(drug_logits)
            return {
                "drug_logits": drug_logits,
                "final_logits": drug_logits,
                "drug_probs": drug_probs,
                "logits_new": logits_new,
                "logits_copy": logits_copy,
                "gate": gate,
                "gate_raw": gate_raw,
                "copy_signal": copy_signal,
                "copy_source_weights": copy_source_weights,
                "copy_source_mask": copy_source_mask,
                "decoder_mode": self.decoder_mode,
            }

        resolved_current_state = _require_2d_tensor(
            name="current_state",
            value=context_vector if current_state is None else current_state,
            batch_size=batch_size,
            feature_dim=self.hidden_dim,
            device=device,
            dtype=dtype,
        )
        resolved_history_context = _require_2d_tensor(
            name="history_context",
            value=history_context,
            batch_size=batch_size,
            feature_dim=self.hidden_dim,
            device=device,
            dtype=dtype,
        )
        resolved_retrieval_context = _require_2d_tensor(
            name="retrieval_context",
            value=retrieval_context,
            batch_size=batch_size,
            feature_dim=self.hidden_dim,
            device=device,
            dtype=dtype,
        )
        resolved_history_med_bag = _require_2d_tensor(
            name="history_med_bag",
            value=history_med_bag,
            batch_size=batch_size,
            feature_dim=self.drug_vocab_size,
            device=device,
            dtype=dtype,
        )
        resolved_retrieval_med_bag = _require_2d_tensor(
            name="retrieval_med_bag",
            value=retrieval_med_bag,
            batch_size=batch_size,
            feature_dim=self.drug_vocab_size,
            device=device,
            dtype=dtype,
        )
        resolved_medication_memory = _require_2d_tensor(
            name="medication_memory",
            value=medication_memory,
            batch_size=batch_size,
            feature_dim=self.drug_vocab_size,
            device=device,
            dtype=dtype,
        )

        copy_signal, copy_source_weights, copy_source_mask = self._build_copy_signal(
            current_state=resolved_current_state,
            history_med_bag=resolved_history_med_bag,
            retrieval_med_bag=resolved_retrieval_med_bag,
            medication_memory=resolved_medication_memory,
        )
        logits_copy = self._compute_logits_copy(copy_signal)
        gate_input = torch.cat(
            (resolved_current_state, resolved_history_context, resolved_retrieval_context),
            dim=-1,
        )
        gate_raw = torch.sigmoid(self.copy_gate_mlp(gate_input))
        if self.gate_type == "scalar":
            gate = gate_raw.expand(-1, self.drug_vocab_size)
        else:
            gate = gate_raw
        copy_available = copy_source_mask.any(dim=1, keepdim=True)
        gate_raw = torch.where(copy_available, gate_raw, torch.ones_like(gate_raw))
        gate = torch.where(copy_available, gate, torch.ones_like(gate))
        drug_logits = gate * logits_new + (1.0 - gate) * logits_copy
        drug_probs = torch.sigmoid(drug_logits)                  # [B, D] probabilities for metrics/thresholding
        return {
            "drug_logits": drug_logits,
            "final_logits": drug_logits,
            "drug_probs": drug_probs,
            "logits_new": logits_new,
            "logits_copy": logits_copy,
            "gate": gate,
            "gate_raw": gate_raw,
            "copy_signal": copy_signal,
            "copy_source_weights": copy_source_weights,
            "copy_source_mask": copy_source_mask,
            "decoder_mode": self.decoder_mode,
        }


__all__ = ["MedicationDecoder"]
