from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import torch
from torch import nn


@dataclass
class NumericFeatureStats:
    mean: float = 0.0
    std: float = 1.0
    count: int = 0


class NumericFeatureProcessor:
    """Kept for optional lab/vital pipeline support."""

    def __init__(
        self,
        feature_size: int,
        stats: list[dict[str, float]] | list[NumericFeatureStats] | None = None,
        *,
        eps: float = 1e-6,
    ) -> None:
        self.feature_size = int(feature_size)
        self.eps = float(eps)
        if stats is None:
            self.stats = [NumericFeatureStats() for _ in range(self.feature_size)]
        else:
            self.stats = [
                item if isinstance(item, NumericFeatureStats) else NumericFeatureStats(**item)
                for item in stats
            ]

    @staticmethod
    def init_running_stats(feature_size: int) -> list[dict[str, float]]:
        return [
            {"count": 0.0, "sum": 0.0, "sum_sq": 0.0}
            for _ in range(int(feature_size))
        ]

    @staticmethod
    def update_running_stats(stats: list[dict[str, float]], feature_index: int, value: float) -> None:
        tracker = stats[feature_index]
        tracker["count"] += 1.0
        tracker["sum"] += value
        tracker["sum_sq"] += value * value

    @staticmethod
    def finalize_running_stats(
        stats: list[dict[str, float]],
        *,
        eps: float = 1e-6,
    ) -> list[dict[str, float]]:
        finalized: list[dict[str, float]] = []
        for tracker in stats:
            count = int(tracker["count"])
            if count <= 0:
                finalized.append({"mean": 0.0, "std": 1.0, "count": 0})
                continue
            mean = tracker["sum"] / tracker["count"]
            variance = max((tracker["sum_sq"] / tracker["count"]) - (mean * mean), 0.0)
            std = max(variance ** 0.5, eps)
            finalized.append({"mean": mean, "std": std, "count": count})
        return finalized

    @staticmethod
    def update_latest(
        sparse_store: dict[int, dict[int, tuple[datetime | None, float]]],
        bucket_index: int,
        feature_index: int,
        event_time: datetime | None,
        value: float,
    ) -> None:
        bucket = sparse_store.setdefault(bucket_index, {})
        current = bucket.get(feature_index)
        if current is None or current[0] is None or (event_time is not None and event_time >= current[0]):
            bucket[feature_index] = (event_time, value)

    def build_dense_steps(
        self,
        sparse_store: dict[int, dict[int, tuple[datetime | None, float]]],
        num_steps: int,
    ) -> tuple[list[list[float]], list[list[int]]]:
        values: list[list[float]] = []
        masks: list[list[int]] = []
        for step_index in range(num_steps):
            dense = [0.0] * self.feature_size
            mask = [0] * self.feature_size
            for feature_index, (_, value) in sparse_store.get(step_index, {}).items():
                if feature_index >= self.feature_size:
                    continue
                stat = self.stats[feature_index]
                normalized = (value - stat.mean) / max(stat.std, self.eps)
                dense[feature_index] = normalized
                mask[feature_index] = 1
            values.append(dense)
            masks.append(mask)
        return values, masks


class LabProcessor(NumericFeatureProcessor):
    """Kept for optional lab/vital pipeline support."""


class NumericFeatureEncoder(nn.Module):
    """Encode numeric visit features into dense visit embeddings.

    Expected input shape:
    - ``[B, T, F]`` or ``[B, F]`` for normalized numeric values
    - optional mask with the same shape, where non-zero entries mark observed values
    """

    def __init__(
        self,
        feature_size: int,
        output_dim: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_size = int(feature_size)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else max(self.output_dim, 32))
        if self.feature_size <= 0:
            self.encoder = None
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.feature_size * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.output_dim),
            )

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"values must be a torch.Tensor, got {type(values)!r}")
        if values.ndim not in {2, 3}:
            raise ValueError(f"values must have shape [B, F] or [B, T, F], got {tuple(values.shape)}")
        if int(values.shape[-1]) != self.feature_size:
            raise ValueError(
                f"Expected numeric feature width {self.feature_size}, got {int(values.shape[-1])}"
            )
        if self.encoder is None:
            return values.new_zeros(*values.shape[:-1], self.output_dim)

        finite_mask = torch.isfinite(values)
        observed_mask = finite_mask if mask is None else mask.to(device=values.device, dtype=torch.bool) & finite_mask
        if tuple(observed_mask.shape) != tuple(values.shape):
            raise ValueError(
                f"Numeric mask shape {tuple(observed_mask.shape)} must match values shape {tuple(values.shape)}"
            )
        sanitized = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        masked_values = sanitized * observed_mask.to(dtype=sanitized.dtype)
        encoder_input = torch.cat([masked_values, observed_mask.to(dtype=sanitized.dtype)], dim=-1)
        encoded = self.encoder(encoder_input)
        has_any_observation = observed_mask.any(dim=-1, keepdim=True)
        return encoded * has_any_observation.to(dtype=encoded.dtype)


class LabFeatureEncoder(NumericFeatureEncoder):
    """Numeric lab branch that returns visit-level lab embeddings."""
