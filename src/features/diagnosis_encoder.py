from __future__ import annotations

import torch
from torch import nn


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


def _assert_finite(name: str, value: torch.Tensor) -> None:
    if value.is_floating_point() or value.is_complex():
        if not torch.isfinite(value).all():
            raise ValueError(
                f"Non-finite tensor detected in diagnosis_encoder at `{name}`; "
                f"{_tensor_debug_summary(name, value)}"
            )


class VisitCodeEncoder(nn.Module):
    """Encode visit-level code inputs into dense visit embeddings.

    Expected inputs:
    - padded code ids with shape ``[B, T, C]`` or ``[B, C]``
    - multi-hot / soft code weights with shape ``[B, T, V]`` or ``[B, V]``

    Returns:
    - visit embeddings with shape ``[B, T, H]`` or ``[B, H]``
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        output_dim: int | None = None,
        padding_idx: int = 0,
        dropout: float = 0.0,
        layer_norm: bool = False,
        max_norm: float | None = None,
        scale_grad_by_freq: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.output_dim = int(output_dim if output_dim is not None else embedding_dim)
        self.padding_idx = int(padding_idx)
        self.max_norm = None if max_norm is None else float(max_norm)
        self.scale_grad_by_freq = bool(scale_grad_by_freq)

        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_dim,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            scale_grad_by_freq=self.scale_grad_by_freq,
        )
        self.projection = (
            nn.Identity()
            if self.output_dim == self.embedding_dim
            else nn.Linear(self.embedding_dim, self.output_dim)
        )
        self.norm = nn.LayerNorm(self.output_dim) if bool(layer_norm) else nn.Identity()
        self.dropout = nn.Dropout(float(dropout))
        if 0 <= self.padding_idx < self.vocab_size:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].zero_()

    def _pool_indices(
        self,
        codes: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resolved_codes = codes.to(dtype=torch.long)
        embeddings = self.embedding(resolved_codes)
        _assert_finite("embedding_lookup", embeddings)
        if mask is None:
            mask = resolved_codes.ne(self.padding_idx)
        if tuple(mask.shape) != tuple(codes.shape):
            raise ValueError(
                f"Code mask must match code id shape, got {tuple(mask.shape)} and {tuple(codes.shape)}"
            )
        resolved_mask = mask.to(device=embeddings.device, dtype=torch.bool) & resolved_codes.ne(self.padding_idx)
        weights = resolved_mask.to(dtype=embeddings.dtype).unsqueeze(-1)
        masked_embeddings = torch.where(weights > 0, embeddings, torch.zeros_like(embeddings))
        pooled = masked_embeddings.sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)
        has_any = resolved_mask.any(dim=-1, keepdim=True)
        pooled = torch.where(has_any, pooled, torch.zeros_like(pooled))
        _assert_finite("pooled_indices", pooled)
        return pooled, has_any

    def _pool_multihot(
        self,
        codes: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.clamp_min(torch.nan_to_num(codes, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        if int(weights.shape[-1]) != self.vocab_size:
            raise ValueError(
                f"Multi-hot codes must have last dimension {self.vocab_size}, got {int(weights.shape[-1])}"
            )
        if mask is not None:
            if tuple(mask.shape) != tuple(weights.shape):
                raise ValueError(
                    f"Multi-hot mask must match multi-hot code shape, got {tuple(mask.shape)} and {tuple(weights.shape)}"
                )
            weights = weights * mask.to(device=weights.device, dtype=weights.dtype)
        has_any = weights.sum(dim=-1, keepdim=True) > 0
        normalized_weights = torch.where(
            has_any,
            weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0),
            torch.zeros_like(weights),
        )
        pooled = torch.matmul(normalized_weights, self.embedding.weight)
        pooled = torch.where(has_any, pooled, torch.zeros_like(pooled))
        _assert_finite("pooled_multihot", pooled)
        return pooled, has_any

    def forward(self, codes: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if not isinstance(codes, torch.Tensor):
            raise TypeError(f"codes must be a torch.Tensor, got {type(codes)!r}")
        if codes.ndim not in {2, 3}:
            raise ValueError(f"codes must have shape [B, C] or [B, T, C], got {tuple(codes.shape)}")

        if codes.dtype.is_floating_point:
            pooled, has_any = self._pool_multihot(codes, mask)
        else:
            pooled, has_any = self._pool_indices(codes, mask)

        _assert_finite("pooled", pooled)
        projected = self.projection(pooled)
        projected = self.norm(projected)
        projected = self.dropout(projected)
        # Empty diagnosis visits should produce a true zero vector, not a biased projection.
        projected = projected * has_any.to(device=projected.device, dtype=projected.dtype)
        _assert_finite("projected", projected)
        return projected


class MaskedCodeEmbeddingPool(VisitCodeEncoder):
    """Kept for backward compatibility with older notebooks/scripts."""


class DiagnosisEncoder(VisitCodeEncoder):
    """Diagnosis visit encoder returning visit-level diagnosis embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        output_dim: int | None = None,
        padding_idx: int = 0,
        dropout: float = 0.0,
        layer_norm: bool = False,
        max_norm: float | None = None,
        scale_grad_by_freq: bool = False,
    ) -> None:
        super().__init__(
            vocab_size,
            embedding_dim,
            output_dim=output_dim,
            padding_idx=padding_idx,
            dropout=dropout,
            layer_norm=layer_norm,
            max_norm=max_norm,
            scale_grad_by_freq=scale_grad_by_freq,
        )
