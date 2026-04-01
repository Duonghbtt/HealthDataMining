from __future__ import annotations

from datetime import datetime
from typing import Sequence

import torch
import torch.nn.functional as F

from src.utils.io import parse_datetime


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(tensor.shape)}")
    return F.normalize(tensor.to(dtype=torch.float32), p=2, dim=-1, eps=1.0e-12)


def _datetime_to_day_value(value: str | datetime | int | float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    dt = value if isinstance(value, datetime) else parse_datetime(value)
    if dt is None:
        return 0.0
    return float(dt.toordinal()) + (
        dt.hour / 24.0
        + dt.minute / 1440.0
        + dt.second / 86400.0
    )


def _coerce_time_values(
    values: Sequence[str | datetime | int | float | None] | torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    if values is None:
        return torch.zeros(0, dtype=torch.float32, device=device)
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.float32).flatten()
    return torch.tensor(
        [_datetime_to_day_value(value) for value in values],
        dtype=torch.float32,
        device=device,
    )


def cosine_similarity_matrix(
    query_embeddings: torch.Tensor,
    key_embeddings: torch.Tensor,
) -> torch.Tensor:
    if key_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D key embeddings, got shape {tuple(key_embeddings.shape)}")
    if key_embeddings.shape[0] == 0:
        return torch.zeros(
            query_embeddings.shape[0],
            0,
            dtype=torch.float32,
            device=query_embeddings.device,
        )
    query = _normalize_rows(query_embeddings)
    keys = _normalize_rows(key_embeddings.to(device=query.device))
    return query @ keys.T


def temporal_decay_weights(
    query_times: Sequence[str | datetime | None] | torch.Tensor | None,
    key_times: Sequence[str | datetime | None] | torch.Tensor | None,
    *,
    alpha: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    if alpha <= 0:
        raise ValueError("alpha must be > 0 for temporal decay")
    target_device = device or (
        query_times.device if isinstance(query_times, torch.Tensor) else torch.device("cpu")
    )
    query_tensor = _coerce_time_values(query_times, device=target_device)
    key_tensor = _coerce_time_values(key_times, device=target_device)
    if query_tensor.numel() == 0 or key_tensor.numel() == 0:
        return torch.ones(
            query_tensor.numel(),
            key_tensor.numel(),
            dtype=torch.float32,
            device=target_device,
        )
    time_gap_days = torch.abs(query_tensor.unsqueeze(1) - key_tensor.unsqueeze(0))
    return torch.exp(-float(alpha) * time_gap_days)


def temporal_similarity(
    query_embeddings: torch.Tensor,
    key_embeddings: torch.Tensor,
    *,
    query_times: Sequence[str | datetime | None] | torch.Tensor | None = None,
    key_times: Sequence[str | datetime | None] | torch.Tensor | None = None,
    alpha: float = 0.05,
) -> dict[str, torch.Tensor]:
    cosine_score = cosine_similarity_matrix(query_embeddings, key_embeddings)
    temporal_weight = temporal_decay_weights(
        query_times,
        key_times,
        alpha=alpha,
        device=cosine_score.device,
    )
    if temporal_weight.shape != cosine_score.shape:
        raise ValueError(
            "Temporal weight shape must match cosine score shape, got "
            f"{tuple(temporal_weight.shape)} vs {tuple(cosine_score.shape)}"
        )
    final_score = cosine_score * temporal_weight
    return {
        "cosine_score": cosine_score,
        "temporal_weight": temporal_weight,
        "final_score": final_score,
    }
