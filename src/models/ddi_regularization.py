from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from src.utils.io import load_pt, read_json


_VALID_REDUCTIONS = {"mean", "sum", "none"}
_DEFAULT_INACTIVE_SOURCE = "fallback_zero"


def _validate_reduction(reduction: str) -> str:
    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(f"reduction must be one of {_VALID_REDUCTIONS}, got {reduction!r}")
    return reduction


def _resolve_ddi_payload(ddi_source: str | Path | Mapping[str, Any] | torch.Tensor) -> Any:
    if isinstance(ddi_source, torch.Tensor):
        return ddi_source
    if isinstance(ddi_source, Mapping):
        return ddi_source
    return load_pt(Path(ddi_source))


def _optional_int(value: Any, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    return int(value)


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _read_adjacent_report(source_path: Path) -> Mapping[str, Any]:
    report_path = source_path.with_name(f"{source_path.stem}_report.json")
    if report_path.exists():
        payload = read_json(report_path)
        if isinstance(payload, Mapping):
            return payload
    return {}


def _normalize_binary_matrix(
    payload: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
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


def _normalize_source_metadata(source: str, payload: Any) -> dict[str, Any]:
    raw = dict(payload) if isinstance(payload, Mapping) else {}
    if source == _DEFAULT_INACTIVE_SOURCE:
        return {
            "kind": str(raw.get("kind") or "fallback_zero"),
            "purpose": str(raw.get("purpose") or "no DDI source configured; inactive fallback artifact"),
            "research_grade": _coerce_bool(raw.get("research_grade"), default=False),
            "pair_schema": str(raw.get("pair_schema") or "none"),
            "display_name": str(raw.get("display_name") or "Fallback Zero DDI"),
        }
    return {
        "kind": str(raw.get("kind") or "unclassified_external"),
        "purpose": str(
            raw.get("purpose")
            or "source metadata missing or incomplete; artifact is runnable but not research-grade by default"
        ),
        "research_grade": _coerce_bool(raw.get("research_grade"), default=False),
        "pair_schema": str(raw.get("pair_schema") or "canonicalized_drug_token_pairs"),
        "display_name": str(raw.get("display_name") or Path(source).name),
    }


def load_ddi_artifact(
    ddi_source: str | Path | Mapping[str, Any] | torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Load a DDI artifact and derive an honest runtime status."""

    source_path = None if isinstance(ddi_source, (torch.Tensor, Mapping)) else Path(ddi_source)
    payload = _resolve_ddi_payload(ddi_source)

    metadata: dict[str, Any] = {}
    matrix_payload: Any = payload
    if isinstance(payload, Mapping):
        metadata.update(dict(payload))
        if "matrix" not in payload:
            raise ValueError("DDI payload mapping must contain a `matrix` field")
        matrix_payload = payload["matrix"]

    if source_path is not None:
        metadata = {**dict(_read_adjacent_report(source_path)), **metadata}

    matrix = _normalize_binary_matrix(matrix_payload, device=device, dtype=dtype)
    nonzero_pairs = int(torch.triu(matrix, diagonal=1).sum().item())
    source = str(metadata.get("source") or (str(source_path.resolve()) if source_path is not None else "in_memory"))
    source_metadata = _normalize_source_metadata(source, metadata.get("source_metadata"))
    matched_pairs = _optional_int(metadata.get("matched_pairs"))
    if matched_pairs is None:
        matched_pairs = nonzero_pairs
    vocab_size = _optional_int(metadata.get("vocab_size"), default=int(matrix.shape[0]))
    if int(vocab_size) != int(matrix.shape[0]):
        raise ValueError(
            "DDI artifact vocab_size must match matrix width: "
            f"got metadata={int(vocab_size)} matrix={int(matrix.shape[0])}"
        )

    if source == _DEFAULT_INACTIVE_SOURCE:
        active = False
        reason = str(metadata.get("reason") or "fallback_zero")
    elif nonzero_pairs <= 0:
        active = False
        reason = str(metadata.get("reason") or "all_zero_matrix")
    elif int(matched_pairs) <= 0:
        active = False
        reason = str(metadata.get("reason") or "no_matched_pairs")
    else:
        active = True
        reason = "available"

    return {
        "matrix": matrix,
        "active": active,
        "reason": reason,
        "source": source,
        "matched_pairs": int(matched_pairs),
        "nonzero_pairs": int(nonzero_pairs),
        "vocab_size": int(vocab_size),
        "source_metadata": source_metadata,
    }


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
    return load_ddi_artifact(ddi_source, device=device, dtype=dtype)["matrix"]


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

        ddi_artifact = load_ddi_artifact(ddi_source, dtype=torch.float32)
        if not bool(ddi_artifact["active"]):
            raise ValueError(
                "Cannot build DDIRegularizer from an inactive DDI artifact: "
                f"reason={ddi_artifact['reason']} source={ddi_artifact['source']}"
            )

        ddi_matrix = ddi_artifact["matrix"]
        ddi_upper = torch.triu(ddi_matrix, diagonal=1)
        pair_i, pair_j = torch.nonzero(ddi_upper, as_tuple=True)
        pair_weights = ddi_upper[pair_i, pair_j]
        pair_normalizer = ddi_upper.sum().clamp(min=1.0)

        self.register_buffer("ddi_upper", ddi_upper)
        self.register_buffer("pair_i", pair_i.to(dtype=torch.long))
        self.register_buffer("pair_j", pair_j.to(dtype=torch.long))
        self.register_buffer("pair_weights", pair_weights.to(dtype=ddi_upper.dtype))
        self.register_buffer("pair_normalizer", pair_normalizer)
        self.ddi_context = {key: value for key, value in ddi_artifact.items() if key != "matrix"}

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

        resolved_probs = drug_probs.to(device=self.pair_weights.device, dtype=self.pair_weights.dtype)
        interacting_probs = resolved_probs.index_select(1, self.pair_i) * resolved_probs.index_select(1, self.pair_j)
        raw_penalty = (interacting_probs * self.pair_weights.unsqueeze(0)).sum(dim=1)
        return raw_penalty / self.pair_normalizer

    def forward(self, drug_probs: torch.Tensor) -> torch.Tensor:
        penalties = self.compute_penalty_per_sample(drug_probs)
        if self.reduction == "mean":
            return penalties.mean()
        if self.reduction == "sum":
            return penalties.sum()
        return penalties

__all__ = ["DDIRegularizer", "load_ddi_artifact", "load_ddi_matrix"]
