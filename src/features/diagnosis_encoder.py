from __future__ import annotations

import torch
from torch import nn


class MaskedCodeEmbeddingPool(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, *, padding_idx: int = 0) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    def forward(self, codes: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = self.embedding(codes)
        if mask is None:
            mask = codes.ne(self.padding_idx)
        mask = mask.to(dtype=embeddings.dtype)
        masked = embeddings * mask.unsqueeze(-1)
        denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return masked.sum(dim=-2) / denom


class DiagnosisEncoder(MaskedCodeEmbeddingPool):
    pass
