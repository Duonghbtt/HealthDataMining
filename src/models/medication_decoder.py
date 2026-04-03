from __future__ import annotations

from typing import Any

import torch
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
    ) -> None:
        super().__init__()
        self.hidden_dim = _validate_positive_int("hidden_dim", hidden_dim)
        self.drug_vocab_size = _validate_positive_int("drug_vocab_size", drug_vocab_size)
        self.top_k_metadata = _validate_optional_top_k("top_k_metadata", top_k_metadata)

        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout!r}")
        self.dropout = float(dropout)

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.drug_vocab_size),
        )

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

        drug_logits = self.decoder(fused_repr)
        drug_probs = torch.sigmoid(drug_logits)

        recommendation_metadata: dict[str, Any] = {
            "batch_size": int(fused_repr.shape[0]),
            "hidden_dim": self.hidden_dim,
            "drug_vocab_size": self.drug_vocab_size,
        }

        if resolved_top_k is not None and resolved_top_k > 0:
            effective_top_k = min(resolved_top_k, self.drug_vocab_size)
            topk_scores, topk_indices = torch.topk(drug_probs, k=effective_top_k, dim=-1)
            recommendation_metadata["topk_indices"] = topk_indices
            recommendation_metadata["topk_scores"] = topk_scores

        return {
            "drug_logits": drug_logits,
            "drug_probs": drug_probs,
            "recommendation_metadata": recommendation_metadata,
        }


__all__ = ["MedicationDecoder"]
