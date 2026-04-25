from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_optional_top_k(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be None or a non-negative integer, got {value!r}")
    return value


def _validate_non_negative_float(name: str, value: float) -> float:
    resolved = float(value)
    if resolved < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return resolved


class MedicationDecoder(nn.Module):
    """Decode fused patient representations into medication recommendation scores.

    Parameters
    ----------
    hidden_dim:
        Dimensionality ``H`` of the incoming fused patient representation.
    drug_vocab_size:
        Number of medications ``D`` to score.
    dropout:
        Dropout probability used in the decoder MLP.
    top_k_metadata:
        Default number of top recommendations to expose in the metadata. Set to
        ``None`` to disable top-k metadata by default.

    Notes
    -----
    Input shape:
        ``fused_repr`` has shape ``(B, H)``.

    Output shapes:
        - ``drug_logits``: ``(B, D)``
        - ``drug_probs``: ``(B, D)``
        - ``recommendation_metadata``: dict containing batch-level metadata and
          optional ``topk_indices`` / ``topk_scores`` tensors with shape ``(B, K)``.

    Example
    -------
    >>> decoder = MedicationDecoder(hidden_dim=128, drug_vocab_size=512)
    >>> outputs = decoder(torch.randn(4, 128), top_k=5)
    >>> outputs["drug_logits"].shape
    torch.Size([4, 512])
    """

    def __init__(
        self,
        hidden_dim: int,
        drug_vocab_size: int,
        *,
        dropout: float = 0.1,
        top_k_metadata: int | None = 10,
        label_correlation_enabled: bool = False,
        correlation_dim: int | None = None,
        patient_residual_weight: float = 0.0,
        coprescription_residual_weight: float = 0.0,
        correlation_dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = _validate_positive_int("hidden_dim", hidden_dim)
        self.drug_vocab_size = _validate_positive_int("drug_vocab_size", drug_vocab_size)
        self.top_k_metadata = _validate_optional_top_k("top_k_metadata", top_k_metadata)

        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout!r}")
        self.dropout = float(dropout)
        self.label_correlation_enabled = bool(label_correlation_enabled)
        self.patient_residual_weight = _validate_non_negative_float(
            "patient_residual_weight",
            patient_residual_weight,
        )
        self.coprescription_residual_weight = _validate_non_negative_float(
            "coprescription_residual_weight",
            coprescription_residual_weight,
        )
        if correlation_dropout is None:
            resolved_correlation_dropout = self.dropout
        else:
            resolved_correlation_dropout = float(correlation_dropout)
        if not 0.0 <= resolved_correlation_dropout <= 1.0:
            raise ValueError(
                f"correlation_dropout must be in [0, 1], got {correlation_dropout!r}"
            )
        self.correlation_dropout = resolved_correlation_dropout
        resolved_correlation_dim = (
            min(self.hidden_dim, 64)
            if correlation_dim is None
            else _validate_positive_int("correlation_dim", int(correlation_dim))
        )
        self.correlation_dim = int(resolved_correlation_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.drug_vocab_size),
        )
        self.patient_correlation_projection: nn.Module | None = None
        self.drug_correlation_embedding: nn.Embedding | None = None
        self.correlation_dropout_layer: nn.Module | None = None
        if self.label_correlation_enabled:
            self.patient_correlation_projection = nn.Sequential(
                nn.Linear(self.hidden_dim, self.correlation_dim),
                nn.Tanh(),
                nn.Dropout(self.correlation_dropout),
            )
            self.drug_correlation_embedding = nn.Embedding(
                self.drug_vocab_size,
                self.correlation_dim,
            )
            self.correlation_dropout_layer = nn.Dropout(self.correlation_dropout)
            nn.init.normal_(self.drug_correlation_embedding.weight, mean=0.0, std=0.02)

    def _correlation_logits(
        self,
        *,
        fused_repr: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero_logits = torch.zeros_like(base_logits)
        if (
            not self.label_correlation_enabled
            or self.drug_correlation_embedding is None
            or self.patient_correlation_projection is None
            or self.correlation_dropout_layer is None
        ):
            return {
                "patient_correlation_logits": zero_logits,
                "coprescription_correlation_logits": zero_logits,
                "label_correlation_logits": zero_logits,
            }

        drug_embeddings = F.normalize(
            self.drug_correlation_embedding.weight.to(
                device=base_logits.device,
                dtype=base_logits.dtype,
            ),
            p=2,
            dim=-1,
            eps=1.0e-12,
        )
        patient_query = self.patient_correlation_projection(fused_repr)
        patient_query = F.normalize(patient_query, p=2, dim=-1, eps=1.0e-12)
        patient_logits = patient_query @ drug_embeddings.transpose(0, 1)

        base_probs = torch.sigmoid(base_logits)
        dropped_probs = self.correlation_dropout_layer(base_probs)
        coprescription_context = dropped_probs @ drug_embeddings
        coprescription_logits = coprescription_context @ drug_embeddings.transpose(0, 1)
        label_correlation_logits = (
            self.patient_residual_weight * patient_logits
            + self.coprescription_residual_weight * coprescription_logits
        )
        return {
            "patient_correlation_logits": patient_logits,
            "coprescription_correlation_logits": coprescription_logits,
            "label_correlation_logits": label_correlation_logits,
        }

    def forward(
        self,
        fused_repr: torch.Tensor,
        *,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Run medication decoding for a batch of fused patient states.

        Parameters
        ----------
        fused_repr:
            Fused patient representation with shape ``(B, H)``.
        top_k:
            Optional override for how many top recommendations to expose inside
            ``recommendation_metadata``. ``None`` falls back to ``top_k_metadata``.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``drug_logits``, ``drug_probs``, and
            ``recommendation_metadata``.
        """

        if not isinstance(fused_repr, torch.Tensor):
            raise TypeError(f"fused_repr must be a torch.Tensor, got {type(fused_repr)!r}")
        if fused_repr.ndim != 2:
            raise ValueError(f"fused_repr must have shape (B, H), got {tuple(fused_repr.shape)}")
        if fused_repr.shape[1] != self.hidden_dim:
            raise ValueError(
                "fused_repr hidden dimension mismatch: "
                f"expected {self.hidden_dim}, got {int(fused_repr.shape[1])}"
            )

        resolved_top_k = _validate_optional_top_k("top_k", top_k)
        if resolved_top_k is None:
            resolved_top_k = self.top_k_metadata

        base_drug_logits = self.decoder(fused_repr)
        correlation_outputs = self._correlation_logits(
            fused_repr=fused_repr,
            base_logits=base_drug_logits,
        )
        drug_logits = base_drug_logits + correlation_outputs["label_correlation_logits"]
        drug_probs = torch.sigmoid(drug_logits)

        recommendation_metadata: dict[str, Any] = {
            "batch_size": int(fused_repr.shape[0]),
            "hidden_dim": self.hidden_dim,
            "drug_vocab_size": self.drug_vocab_size,
            "label_correlation_enabled": self.label_correlation_enabled,
            "correlation_dim": self.correlation_dim,
            "patient_residual_weight": self.patient_residual_weight,
            "coprescription_residual_weight": self.coprescription_residual_weight,
        }

        if resolved_top_k is not None and resolved_top_k > 0:
            effective_top_k = min(resolved_top_k, self.drug_vocab_size)
            topk_scores, topk_indices = torch.topk(drug_probs, k=effective_top_k, dim=-1)
            recommendation_metadata["topk_indices"] = topk_indices
            recommendation_metadata["topk_scores"] = topk_scores

        return {
            "drug_logits": drug_logits,
            "drug_probs": drug_probs,
            "base_drug_logits": base_drug_logits,
            **correlation_outputs,
            "recommendation_metadata": recommendation_metadata,
        }


__all__ = ["MedicationDecoder"]
