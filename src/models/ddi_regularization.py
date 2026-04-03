from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from src.utils.io import load_pt


_VALID_REDUCTIONS = {"mean", "sum", "none"}


def _validate_reduction(reduction: str) -> str:
    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(f"reduction must be one of {_VALID_REDUCTIONS}, got {reduction!r}")
    return reduction


def _resolve_ddi_payload(ddi_source: str | Path | Mapping[str, Any] | torch.Tensor) -> Any:
    if isinstance(ddi_source, torch.Tensor):
        return ddi_source
    if isinstance(ddi_source, Mapping):
        return ddi_source.get("matrix", ddi_source)
    return load_pt(Path(ddi_source))


def load_ddi_matrix(
    ddi_source: str | Path | Mapping[str, Any] | torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load and normalize a DDI adjacency matrix.

    Parameters
    ----------
    ddi_source:
        One of:
        - a raw tensor with shape ``(D, D)``
        - a mapping containing a ``"matrix"`` entry
        - a path to a serialized payload compatible with ``torch.load``
    device:
        Optional device for the returned tensor.
    dtype:
        Output dtype for the returned tensor.

    Returns
    -------
    torch.Tensor
        Dense binary DDI adjacency matrix with shape ``(D, D)``.
    """

    payload = _resolve_ddi_payload(ddi_source)
    if isinstance(payload, Mapping):
        if "matrix" not in payload:
            raise ValueError("DDI payload mapping must contain a `matrix` field")
        payload = payload["matrix"]

    matrix = torch.as_tensor(payload, dtype=dtype)
    if matrix.ndim != 2:
        raise ValueError(f"DDI matrix must have shape (D, D), got {tuple(matrix.shape)}")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"DDI matrix must be square, got {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError("DDI matrix must contain only finite values")

    binary_matrix = (matrix > 0).to(dtype=dtype)
    binary_matrix = torch.maximum(binary_matrix, binary_matrix.transpose(0, 1))
    binary_matrix.fill_diagonal_(0.0)

    if device is not None:
        binary_matrix = binary_matrix.to(device=device)
    return binary_matrix


class DDIRegularizer(nn.Module):
    """Differentiable DDI penalty computed from predicted drug probabilities.

    Notes
    -----
    Input shape:
        ``drug_probs`` must have shape ``(B, D)``.

    Output shape:
        - ``reduction="mean"`` or ``"sum"``: scalar tensor
        - ``reduction="none"``: tensor with shape ``(B,)``
    """

    def __init__(
        self,
        ddi_source: str | Path | Mapping[str, Any] | torch.Tensor,
        *,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reduction = _validate_reduction(reduction)

        ddi_matrix = load_ddi_matrix(ddi_source, dtype=torch.float32)
        ddi_upper = torch.triu(ddi_matrix, diagonal=1)
        pair_normalizer = ddi_upper.sum().clamp(min=1.0)

        self.register_buffer("ddi_upper", ddi_upper)
        self.register_buffer("pair_normalizer", pair_normalizer)

    @property
    def drug_vocab_size(self) -> int:
        return int(self.ddi_upper.shape[0])

    def compute_penalty_per_sample(self, drug_probs: torch.Tensor) -> torch.Tensor:
        """Compute per-sample expected harmful-pair mass.

        Parameters
        ----------
        drug_probs:
            Predicted medication probabilities with shape ``(B, D)``.

        Returns
        -------
        torch.Tensor
            Per-sample DDI penalty with shape ``(B,)``.
        """

        if not isinstance(drug_probs, torch.Tensor):
            raise TypeError(f"drug_probs must be a torch.Tensor, got {type(drug_probs)!r}")
        if drug_probs.ndim != 2:
            raise ValueError(f"drug_probs must have shape (B, D), got {tuple(drug_probs.shape)}")
        if drug_probs.shape[1] != self.drug_vocab_size:
            raise ValueError(
                "drug_probs width must match the DDI matrix width: "
                f"expected {self.drug_vocab_size}, got {int(drug_probs.shape[1])}"
            )
        if not torch.isfinite(drug_probs).all():
            raise ValueError("drug_probs must contain only finite values")

        resolved_probs = drug_probs.to(device=self.ddi_upper.device, dtype=self.ddi_upper.dtype)
        pair_probs = resolved_probs.unsqueeze(2) * resolved_probs.unsqueeze(1)
        raw_penalty = (pair_probs * self.ddi_upper.unsqueeze(0)).sum(dim=(1, 2))
        return raw_penalty / self.pair_normalizer

    def forward(self, drug_probs: torch.Tensor) -> torch.Tensor:
        penalties = self.compute_penalty_per_sample(drug_probs)
        if self.reduction == "mean":
            return penalties.mean()
        if self.reduction == "sum":
            return penalties.sum()
        return penalties


__all__ = ["DDIRegularizer", "load_ddi_matrix"]
