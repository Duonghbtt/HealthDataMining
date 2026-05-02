from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

_VALID_SIMILARITY_MODES = {"cosine_decay", "cosine_decay_mlp"}


def _as_2d_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape (N, H), got {tuple(value.shape)}")
    return value


def _as_candidate_matrix(
    *,
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    if candidate_embeddings.ndim == 2:
        return candidate_embeddings.unsqueeze(0).expand(query_embeddings.shape[0], -1, -1), True
    if candidate_embeddings.ndim == 3:
        if candidate_embeddings.shape[0] != query_embeddings.shape[0]:
            raise ValueError(
                "candidate_embeddings batch dimension must match query_embeddings: "
                f"got {tuple(candidate_embeddings.shape)} and {tuple(query_embeddings.shape)}"
            )
        return candidate_embeddings, False
    raise ValueError(
        "candidate_embeddings must have shape (N, H) or (B, N, H), "
        f"got {tuple(candidate_embeddings.shape)}"
    )


def _broadcast_scalar_feature(
    *,
    name: str,
    value: torch.Tensor | None,
    batch_size: int,
    num_candidates: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 1:
        if tensor.shape[0] == batch_size:
            return tensor.unsqueeze(1).expand(-1, num_candidates)
        if tensor.shape[0] == num_candidates:
            return tensor.unsqueeze(0).expand(batch_size, -1)
    if tensor.ndim == 2 and tuple(tensor.shape) == (batch_size, num_candidates):
        return tensor
    raise ValueError(
        f"{name} must have shape (B,), (N,), or (B, N); "
        f"got {tuple(tensor.shape)} for batch_size={batch_size}, num_candidates={num_candidates}"
    )


def _normalized_gap(gap: torch.Tensor) -> torch.Tensor:
    gap = torch.clamp(gap, min=0.0)
    return torch.log1p(gap)


class TemporalSimilarity(nn.Module):
    """Cosine similarity with optional temporal decay or a tiny learned scorer."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        similarity_mode: str = "cosine_decay",
        temporal_decay_alpha: float = 0.05,
        mlp_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.similarity_mode = str(similarity_mode).strip().lower()
        if self.similarity_mode not in _VALID_SIMILARITY_MODES:
            raise ValueError(
                f"similarity_mode must be one of {_VALID_SIMILARITY_MODES}, got {similarity_mode!r}"
            )
        self.temporal_decay_alpha = float(temporal_decay_alpha)
        self.score_mlp = (
            nn.Sequential(
                nn.Linear(4, int(mlp_hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(mlp_hidden_dim), 1),
            )
            if self.similarity_mode == "cosine_decay_mlp"
            else None
        )

    def forward(
        self,
        *,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        query_times: torch.Tensor | None = None,
        candidate_times: torch.Tensor | None = None,
        query_indices: torch.Tensor | None = None,
        candidate_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        query_embeddings = _as_2d_tensor("query_embeddings", query_embeddings)
        candidate_matrix, _ = _as_candidate_matrix(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
        )
        if candidate_matrix.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected candidate hidden dim {self.hidden_dim}, got {int(candidate_matrix.shape[-1])}"
            )
        if query_embeddings.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected query hidden dim {self.hidden_dim}, got {int(query_embeddings.shape[-1])}"
            )

        batch_size, num_candidates, _ = candidate_matrix.shape
        query_norm = F.normalize(query_embeddings, dim=-1)
        candidate_norm = F.normalize(candidate_matrix, dim=-1)
        cosine_score = torch.einsum("bh,bnh->bn", query_norm, candidate_norm)

        query_time_matrix = _broadcast_scalar_feature(
            name="query_times",
            value=query_times,
            batch_size=batch_size,
            num_candidates=num_candidates,
            device=query_embeddings.device,
            dtype=query_embeddings.dtype,
        )
        candidate_time_matrix = _broadcast_scalar_feature(
            name="candidate_times",
            value=candidate_times,
            batch_size=batch_size,
            num_candidates=num_candidates,
            device=query_embeddings.device,
            dtype=query_embeddings.dtype,
        )
        query_index_matrix = _broadcast_scalar_feature(
            name="query_indices",
            value=query_indices,
            batch_size=batch_size,
            num_candidates=num_candidates,
            device=query_embeddings.device,
            dtype=query_embeddings.dtype,
        )
        candidate_index_matrix = _broadcast_scalar_feature(
            name="candidate_indices",
            value=candidate_indices,
            batch_size=batch_size,
            num_candidates=num_candidates,
            device=query_embeddings.device,
            dtype=query_embeddings.dtype,
        )

        if query_time_matrix is not None and candidate_time_matrix is not None:
            time_gap = torch.abs(query_time_matrix - candidate_time_matrix)
        elif query_index_matrix is not None and candidate_index_matrix is not None:
            time_gap = torch.abs(query_index_matrix - candidate_index_matrix)
        else:
            time_gap = torch.zeros_like(cosine_score)

        normalized_gap = _normalized_gap(time_gap)
        temporal_penalty = self.temporal_decay_alpha * normalized_gap

        if self.score_mlp is None:
            final_score = cosine_score - temporal_penalty
        else:
            interaction = cosine_score * torch.exp(-normalized_gap)
            score_features = torch.stack(
                (
                    cosine_score,
                    normalized_gap,
                    cosine_score - temporal_penalty,
                    interaction,
                ),
                dim=-1,
            )
            final_score = self.score_mlp(score_features).squeeze(-1)

        return {
            "cosine_score": cosine_score,
            "time_gap": time_gap,
            "temporal_penalty": temporal_penalty,
            "final_score": final_score,
        }


__all__ = ["TemporalSimilarity"]
