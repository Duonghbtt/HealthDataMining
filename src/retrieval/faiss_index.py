from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


try:
    import faiss  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    faiss = None


@dataclass
class FaissIndex:
    dimension: int
    index: object

    @classmethod
    def is_available(cls) -> bool:
        return faiss is not None

    @classmethod
    def build(cls, embeddings: torch.Tensor) -> "FaissIndex":
        if faiss is None:
            raise RuntimeError("FAISS is not available. Falling back to brute-force retrieval is required.")
        normalized = F.normalize(embeddings.to(dtype=torch.float32), p=2, dim=-1).cpu().numpy()
        index = faiss.IndexFlatIP(int(normalized.shape[1]))
        index.add(normalized)
        return cls(dimension=int(normalized.shape[1]), index=index)

    def search(self, query_embeddings: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = F.normalize(query_embeddings.to(dtype=torch.float32), p=2, dim=-1).cpu().numpy()
        scores, indices = self.index.search(normalized, int(k))
        return (
            torch.from_numpy(scores).to(dtype=torch.float32),
            torch.from_numpy(indices).to(dtype=torch.long),
        )
