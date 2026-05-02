from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    faiss = None


class VisitFaissIndex:
    """Acceleration layer for visit retrieval with a brute-force fallback.

    FAISS is optional. Temporal leakage filtering still happens after search, so this
    index only performs coarse similarity lookup.
    """

    def __init__(
        self,
        *,
        backend: str = "bruteforce",
        use_faiss_if_available: bool = True,
    ) -> None:
        normalized_backend = str(backend).strip().lower()
        if normalized_backend not in {"bruteforce", "faiss"}:
            raise ValueError(f"backend must be `bruteforce` or `faiss`, got {backend!r}")
        self.backend = normalized_backend
        self.use_faiss_if_available = bool(use_faiss_if_available)
        self._index: Any | None = None
        self._embeddings = torch.empty(0, 0, dtype=torch.float32)

    @property
    def faiss_available(self) -> bool:
        return faiss is not None and self.use_faiss_if_available

    @property
    def is_built(self) -> bool:
        return self._embeddings.numel() > 0

    def build_index(self, embeddings: torch.Tensor) -> None:
        resolved = torch.as_tensor(embeddings, dtype=torch.float32).detach().cpu()
        if resolved.ndim != 2:
            raise ValueError(f"embeddings must have shape (N, H), got {tuple(resolved.shape)}")
        self._embeddings = resolved
        self._index = None
        if self.backend == "faiss" and self.faiss_available and resolved.shape[0] > 0:
            normalized = F.normalize(resolved, dim=-1).numpy()
            self._index = faiss.IndexFlatIP(int(resolved.shape[1]))
            self._index.add(normalized)

    def search(self, queries: torch.Tensor, top_k: int) -> dict[str, torch.Tensor]:
        if int(top_k) <= 0:
            raise ValueError(f"top_k must be positive, got {top_k!r}")
        if not self.is_built:
            return {
                "scores": torch.empty(queries.shape[0], 0, dtype=torch.float32),
                "indices": torch.empty(queries.shape[0], 0, dtype=torch.long),
            }

        query_tensor = torch.as_tensor(queries, dtype=torch.float32).detach().cpu()
        if query_tensor.ndim != 2:
            raise ValueError(f"queries must have shape (B, H), got {tuple(query_tensor.shape)}")
        if query_tensor.shape[1] != self._embeddings.shape[1]:
            raise ValueError(
                "Query width must match index width: "
                f"got {int(query_tensor.shape[1])} and {int(self._embeddings.shape[1])}"
            )

        resolved_top_k = min(int(top_k), int(self._embeddings.shape[0]))
        if resolved_top_k == 0:
            return {
                "scores": torch.empty(query_tensor.shape[0], 0, dtype=torch.float32),
                "indices": torch.empty(query_tensor.shape[0], 0, dtype=torch.long),
            }

        if self._index is not None:
            search_scores, search_indices = self._index.search(
                F.normalize(query_tensor, dim=-1).numpy(),
                resolved_top_k,
            )
            return {
                "scores": torch.from_numpy(search_scores).to(dtype=torch.float32),
                "indices": torch.from_numpy(search_indices).to(dtype=torch.long),
            }

        normalized_queries = F.normalize(query_tensor, dim=-1)
        normalized_keys = F.normalize(self._embeddings, dim=-1)
        similarity = normalized_queries @ normalized_keys.T
        top_scores, top_indices = torch.topk(similarity, k=resolved_top_k, dim=-1)
        return {
            "scores": top_scores.to(dtype=torch.float32),
            "indices": top_indices.to(dtype=torch.long),
        }


__all__ = ["VisitFaissIndex"]
