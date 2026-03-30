from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class NumericFeatureStats:
    mean: float = 0.0
    std: float = 1.0
    count: int = 0


class NumericFeatureProcessor:
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
    pass
