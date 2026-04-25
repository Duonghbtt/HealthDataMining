from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _as_subject_tensor(subject_ids: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(subject_ids, torch.Tensor):
        return subject_ids.to(device=device, dtype=torch.long).view(-1)
    return torch.as_tensor(subject_ids, device=device, dtype=torch.long).view(-1)


def compute_contrastive_loss(
    embeddings: torch.Tensor,
    subject_ids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute supervised in-batch InfoNCE over patient-state embeddings."""

    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(f"embeddings must be a torch.Tensor, got {type(embeddings)!r}")
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must have shape (B, D), got {tuple(embeddings.shape)}")
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature!r}")

    batch_size = int(embeddings.shape[0])
    if batch_size <= 1:
        return embeddings.new_zeros(())

    resolved_subject_ids = _as_subject_tensor(subject_ids, device=embeddings.device)
    if int(resolved_subject_ids.numel()) != batch_size:
        raise ValueError(
            "subject_ids must contain one value per embedding: "
            f"got {int(resolved_subject_ids.numel())} ids for batch size {batch_size}"
        )

    normalized = F.normalize(embeddings.float(), dim=-1)
    logits = torch.matmul(normalized, normalized.T) / float(temperature)
    non_self_mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    pos_mask = resolved_subject_ids.unsqueeze(0).eq(resolved_subject_ids.unsqueeze(1)) & non_self_mask
    valid_anchor_mask = pos_mask.any(dim=1)
    if not bool(valid_anchor_mask.any().item()):
        return embeddings.new_zeros(())

    masked_logits = logits.masked_fill(~non_self_mask, float("-inf"))
    log_denominator = torch.logsumexp(masked_logits, dim=1)
    positive_logits = logits.masked_fill(~pos_mask, float("-inf"))
    log_numerator = torch.logsumexp(positive_logits, dim=1)
    losses = -(log_numerator[valid_anchor_mask] - log_denominator[valid_anchor_mask])
    return losses.mean().to(dtype=embeddings.dtype)


def compute_embedding_similarity_stats(embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return scalar diagnostics for pooled-state norms and off-diagonal similarities."""

    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("embeddings must be a tensor with shape (B, D)")

    embeddings_float = embeddings.detach().float()
    norms = embeddings_float.norm(dim=-1)
    zero = embeddings_float.new_zeros(())
    if norms.numel() <= 0:
        norm_mean = zero
        norm_std = zero
    else:
        norm_mean = norms.mean()
        norm_std = norms.std(unbiased=False) if norms.numel() > 1 else zero

    if embeddings_float.shape[0] <= 1:
        similarity_mean = zero
        similarity_std = zero
    else:
        normalized = F.normalize(embeddings_float, dim=-1)
        sim_matrix = torch.matmul(normalized, normalized.T)
        non_self_mask = ~torch.eye(
            int(embeddings_float.shape[0]),
            dtype=torch.bool,
            device=embeddings_float.device,
        )
        off_diagonal = sim_matrix[non_self_mask]
        similarity_mean = off_diagonal.mean() if off_diagonal.numel() > 0 else zero
        similarity_std = off_diagonal.std(unbiased=False) if off_diagonal.numel() > 1 else zero

    return {
        "embedding_norm_mean": norm_mean.to(device=embeddings.device),
        "embedding_norm_std": norm_std.to(device=embeddings.device),
        "similarity_mean": similarity_mean.to(device=embeddings.device),
        "similarity_std": similarity_std.to(device=embeddings.device),
    }


__all__ = ["compute_contrastive_loss", "compute_embedding_similarity_stats"]
