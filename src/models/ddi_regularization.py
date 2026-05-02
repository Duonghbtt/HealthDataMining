from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from src.utils.io import load_pt


_VALID_REDUCTIONS = {"mean", "sum", "none", "batchmean"}
_VALID_DECODE_MODES = {"threshold", "topk", "soft_constrained_rerank"}


def _validate_reduction(reduction: str) -> str:
    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(f"reduction must be one of {_VALID_REDUCTIONS}, got {reduction!r}")
    return reduction


def _reduce_penalties(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError(f"Expected per-sample DDI penalties with shape (B,), got {tuple(values.shape)}")
    if reduction in {"mean", "batchmean"}:
        return values.mean()
    if reduction == "sum":
        return values.sum()
    return values


def _resolve_ddi_payload(ddi_source: str | Path | Mapping[str, Any] | torch.Tensor) -> Any:
    if isinstance(ddi_source, torch.Tensor):
        return ddi_source
    if isinstance(ddi_source, Mapping):
        return ddi_source.get("matrix", ddi_source)
    return load_pt(Path(ddi_source))


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

    payload = _resolve_ddi_payload(ddi_source)
    if isinstance(payload, Mapping):
        if "matrix" not in payload:
            raise ValueError("DDI payload mapping must contain a `matrix` field")
        payload = payload["matrix"]

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


def _resolve_probabilities(
    drug_scores: torch.Tensor,
    *,
    input_is_logits: bool,
) -> torch.Tensor:
    if not isinstance(drug_scores, torch.Tensor):
        raise TypeError(f"drug_scores must be a torch.Tensor, got {type(drug_scores)!r}")
    if drug_scores.ndim != 2:
        raise ValueError(f"drug_scores must have shape (B, D), got {tuple(drug_scores.shape)}")
    if not torch.isfinite(drug_scores).all():
        raise ValueError("drug_scores must contain only finite values")

    if bool(input_is_logits):
        return torch.sigmoid(drug_scores)

    if bool(((drug_scores < 0.0) | (drug_scores > 1.0)).any().item()):
        raise ValueError("drug_scores must be probabilities in [0, 1] when input_is_logits=False")
    return drug_scores


def _resolve_probability_vector(
    drug_scores: torch.Tensor | Sequence[float],
    *,
    input_is_logits: bool,
) -> torch.Tensor:
    vector = torch.as_tensor(drug_scores, dtype=torch.float32)
    if vector.ndim != 1:
        raise ValueError(f"drug_scores must have shape (D,), got {tuple(vector.shape)}")
    if not torch.isfinite(vector).all():
        raise ValueError("drug_scores must contain only finite values")
    if bool(input_is_logits):
        return torch.sigmoid(vector)
    if bool(((vector < 0.0) | (vector > 1.0)).any().item()):
        raise ValueError("drug_scores must be probabilities in [0, 1] when input_is_logits=False")
    return vector


def _resolve_ddi_upper(
    ddi_matrix: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    resolved_ddi = load_ddi_matrix(ddi_matrix, device=device, dtype=dtype)
    return torch.triu(resolved_ddi, diagonal=1)


def _selected_set_to_mask(
    selected_set: torch.Tensor | Sequence[int],
    *,
    drug_vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(selected_set, torch.Tensor):
        resolved = selected_set.to(device=device)
    else:
        resolved = torch.as_tensor(selected_set, device=device)

    if resolved.ndim != 1:
        raise ValueError(f"selected_set must have shape (D,) or (K,), got {tuple(resolved.shape)}")
    if int(resolved.numel()) == int(drug_vocab_size) and resolved.dtype not in {torch.int64, torch.int32, torch.int16, torch.int8, torch.long}:
        return (resolved > 0).to(dtype=torch.bool)
    if int(resolved.numel()) == int(drug_vocab_size) and resolved.dtype == torch.bool:
        return resolved.to(dtype=torch.bool)

    indices = resolved.to(dtype=torch.long).reshape(-1)
    mask = torch.zeros(drug_vocab_size, dtype=torch.bool, device=device)
    if int(indices.numel()) <= 0:
        return mask
    valid_indices = indices[(indices >= 0) & (indices < drug_vocab_size)]
    if int(valid_indices.numel()) > 0:
        mask[valid_indices] = True
    return mask


def compute_ddi_penalty_for_set(
    selected_set: torch.Tensor | Sequence[int],
    ddi_matrix: torch.Tensor,
    *,
    drug_vocab_size: int | None = None,
) -> dict[str, float]:
    """Compute DDI statistics for one predicted medication set.

    Parameters
    ----------
    selected_set:
        Either a binary mask in drug space ``(D,)`` or a 1D list/tensor of
        selected drug indices.
    ddi_matrix:
        DDI adjacency matrix aligned with the medication vocabulary.
    drug_vocab_size:
        Required when ``selected_set`` is a list of indices.
    """

    ddi_upper = _resolve_ddi_upper(ddi_matrix)
    resolved_vocab_size = int(drug_vocab_size if drug_vocab_size is not None else ddi_upper.shape[0])
    selected_mask = _selected_set_to_mask(
        selected_set,
        drug_vocab_size=resolved_vocab_size,
        device=ddi_upper.device,
    )
    if int(selected_mask.shape[0]) != int(ddi_upper.shape[0]):
        raise ValueError(
            "selected_set width must match ddi_matrix width: "
            f"got {int(selected_mask.shape[0])} and {int(ddi_upper.shape[0])}"
        )

    selected_indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
    num_selected_drugs = int(selected_indices.numel())
    if num_selected_drugs < 2:
        return {
            "num_selected_drugs": float(num_selected_drugs),
            "num_pairs": 0.0,
            "num_ddi_pairs": 0.0,
            "ddi_rate": 0.0,
            "normalized_penalty": 0.0,
        }

    num_pairs = float(num_selected_drugs * (num_selected_drugs - 1) // 2)
    sample_ddi = ddi_upper.index_select(0, selected_indices).index_select(1, selected_indices)
    num_ddi_pairs = float(sample_ddi.sum(dtype=torch.float32).item())
    ddi_rate = 0.0 if num_pairs <= 0.0 else num_ddi_pairs / num_pairs
    return {
        "num_selected_drugs": float(num_selected_drugs),
        "num_pairs": float(num_pairs),
        "num_ddi_pairs": float(num_ddi_pairs),
        "ddi_rate": float(ddi_rate),
        "normalized_penalty": float(ddi_rate),
    }


def _compute_size_penalty(
    *,
    num_selected_drugs: int,
    min_drugs: int = 0,
    max_drugs: int | None = None,
    target_avg_drugs: float | None = None,
) -> float:
    penalty = 0.0
    if num_selected_drugs < int(min_drugs):
        penalty += float(int(min_drugs) - num_selected_drugs)
    if max_drugs is not None and num_selected_drugs > int(max_drugs):
        penalty += float(num_selected_drugs - int(max_drugs))
    if target_avg_drugs is not None:
        resolved_target = max(float(target_avg_drugs), 1.0)
        penalty += abs(float(num_selected_drugs) - resolved_target) / resolved_target
    return float(penalty)


def score_candidate_set(
    drug_scores: torch.Tensor | Sequence[float],
    selected_set: torch.Tensor | Sequence[int],
    ddi_matrix: torch.Tensor,
    *,
    input_is_logits: bool = False,
    alpha_utility: float = 1.0,
    beta_ddi: float = 0.5,
    gamma_size: float = 0.1,
    min_drugs: int = 0,
    max_drugs: int | None = None,
    target_avg_drugs: float | None = None,
) -> dict[str, float]:
    """Score one candidate medication set with utility-safety trade-off."""

    probabilities = _resolve_probability_vector(drug_scores, input_is_logits=input_is_logits)
    selected_mask = _selected_set_to_mask(
        selected_set,
        drug_vocab_size=int(probabilities.shape[0]),
        device=probabilities.device,
    )
    if int(selected_mask.shape[0]) != int(probabilities.shape[0]):
        raise ValueError(
            "selected_set width must match drug_scores width: "
            f"got {int(selected_mask.shape[0])} and {int(probabilities.shape[0])}"
        )

    ddi_stats = compute_ddi_penalty_for_set(
        selected_mask,
        ddi_matrix,
        drug_vocab_size=int(probabilities.shape[0]),
    )
    utility_score = float(probabilities[selected_mask].sum().item())
    size_penalty = _compute_size_penalty(
        num_selected_drugs=int(ddi_stats["num_selected_drugs"]),
        min_drugs=int(min_drugs),
        max_drugs=max_drugs,
        target_avg_drugs=target_avg_drugs,
    )
    total_score = (
        float(alpha_utility) * utility_score
        - float(beta_ddi) * float(ddi_stats["normalized_penalty"])
        - float(gamma_size) * size_penalty
    )
    return {
        "utility_score": float(utility_score),
        "ddi_penalty": float(ddi_stats["normalized_penalty"]),
        "ddi_rate": float(ddi_stats["ddi_rate"]),
        "num_ddi_pairs": float(ddi_stats["num_ddi_pairs"]),
        "size_penalty": float(size_penalty),
        "total_score": float(total_score),
        "num_predicted_drugs": float(ddi_stats["num_selected_drugs"]),
    }


def _binary_topk(probabilities: torch.Tensor, top_k: int) -> torch.Tensor:
    resolved_top_k = min(max(int(top_k), 0), int(probabilities.shape[0]))
    predictions = torch.zeros_like(probabilities, dtype=torch.bool)
    if resolved_top_k <= 0:
        return predictions
    top_indices = torch.topk(probabilities, k=resolved_top_k, dim=0).indices
    predictions[top_indices] = True
    return predictions


def _initial_rerank_mask(
    *,
    candidate_indices: torch.Tensor,
    probabilities: torch.Tensor,
    threshold: float,
    min_drugs: int,
    max_drugs: int | None,
    ensure_non_empty: bool,
) -> torch.Tensor:
    initial_mask = torch.zeros_like(probabilities, dtype=torch.bool)
    if int(candidate_indices.numel()) > 0:
        threshold_indices = candidate_indices[probabilities[candidate_indices] >= float(threshold)]
        if int(threshold_indices.numel()) > 0:
            initial_mask[threshold_indices] = True
    if max_drugs is not None and int(initial_mask.sum().item()) > int(max_drugs):
        kept = candidate_indices[initial_mask[candidate_indices]][: int(max_drugs)]
        initial_mask.zero_()
        if int(kept.numel()) > 0:
            initial_mask[kept] = True
    if int(initial_mask.sum().item()) < int(min_drugs):
        initial_mask.zero_()
        forced_indices = candidate_indices[: max(int(min_drugs), 1 if ensure_non_empty else 0)]
        if int(forced_indices.numel()) > 0:
            initial_mask[forced_indices] = True
    return initial_mask


def _rerank_single_prediction_set(
    probabilities: torch.Tensor,
    ddi_matrix: torch.Tensor,
    *,
    threshold: float,
    top_m: int,
    alpha_utility: float,
    beta_ddi: float,
    gamma_size: float,
    min_drugs: int,
    max_drugs: int | None,
    target_avg_drugs: float | None,
    ensure_non_empty: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    resolved_top_m = min(max(int(top_m), 1), int(probabilities.shape[0]))
    candidate_indices = torch.topk(probabilities, k=resolved_top_m, dim=0).indices
    selected_mask = _initial_rerank_mask(
        candidate_indices=candidate_indices,
        probabilities=probabilities,
        threshold=threshold,
        min_drugs=min_drugs,
        max_drugs=max_drugs,
        ensure_non_empty=ensure_non_empty,
    )
    current_stats = score_candidate_set(
        probabilities,
        selected_mask,
        ddi_matrix,
        input_is_logits=False,
        alpha_utility=alpha_utility,
        beta_ddi=beta_ddi,
        gamma_size=gamma_size,
        min_drugs=min_drugs,
        max_drugs=max_drugs,
        target_avg_drugs=target_avg_drugs,
    )

    if int(selected_mask.sum().item()) > int(min_drugs):
        selected_indices = candidate_indices[selected_mask[candidate_indices]]
        removal_order = selected_indices[torch.argsort(probabilities[selected_indices], descending=False)]
        for index in removal_order.tolist():
            candidate_mask = selected_mask.clone()
            candidate_mask[int(index)] = False
            if ensure_non_empty and not bool(candidate_mask.any().item()):
                continue
            candidate_stats = score_candidate_set(
                probabilities,
                candidate_mask,
                ddi_matrix,
                input_is_logits=False,
                alpha_utility=alpha_utility,
                beta_ddi=beta_ddi,
                gamma_size=gamma_size,
                min_drugs=min_drugs,
                max_drugs=max_drugs,
                target_avg_drugs=target_avg_drugs,
            )
            if float(candidate_stats["total_score"]) >= float(current_stats["total_score"]):
                selected_mask = candidate_mask
                current_stats = candidate_stats

    for index in candidate_indices.tolist():
        if bool(selected_mask[int(index)].item()):
            continue
        if max_drugs is not None and int(selected_mask.sum().item()) >= int(max_drugs):
            break
        candidate_mask = selected_mask.clone()
        candidate_mask[int(index)] = True
        candidate_stats = score_candidate_set(
            probabilities,
            candidate_mask,
            ddi_matrix,
            input_is_logits=False,
            alpha_utility=alpha_utility,
            beta_ddi=beta_ddi,
            gamma_size=gamma_size,
            min_drugs=min_drugs,
            max_drugs=max_drugs,
            target_avg_drugs=target_avg_drugs,
        )
        if (
            float(candidate_stats["total_score"]) > float(current_stats["total_score"])
            or int(selected_mask.sum().item()) < int(min_drugs)
        ):
            selected_mask = candidate_mask
            current_stats = candidate_stats

    if ensure_non_empty and not bool(selected_mask.any().item()) and int(candidate_indices.numel()) > 0:
        selected_mask[int(candidate_indices[0].item())] = True
        current_stats = score_candidate_set(
            probabilities,
            selected_mask,
            ddi_matrix,
            input_is_logits=False,
            alpha_utility=alpha_utility,
            beta_ddi=beta_ddi,
            gamma_size=gamma_size,
            min_drugs=min_drugs,
            max_drugs=max_drugs,
            target_avg_drugs=target_avg_drugs,
        )

    return selected_mask, current_stats


def rerank_prediction_set(
    drug_scores: torch.Tensor,
    ddi_matrix: torch.Tensor,
    *,
    input_is_logits: bool = False,
    decode_mode: str = "threshold",
    threshold: float = 0.5,
    top_k: int | None = None,
    top_m: int = 20,
    alpha_utility: float = 1.0,
    beta_ddi: float = 0.5,
    gamma_size: float = 0.1,
    min_drugs: int = 1,
    max_drugs: int | None = None,
    target_avg_drugs: float | None = None,
    ensure_non_empty: bool | None = None,
) -> dict[str, Any]:
    """Decode medication sets with baseline or soft constrained reranking."""

    mode = str(decode_mode).strip().lower()
    if mode not in _VALID_DECODE_MODES:
        raise ValueError(f"decode_mode must be one of {_VALID_DECODE_MODES}, got {decode_mode!r}")
    probabilities = _resolve_probabilities(
        torch.as_tensor(drug_scores, dtype=torch.float32),
        input_is_logits=input_is_logits,
    )
    ddi_upper = _resolve_ddi_upper(ddi_matrix, device=probabilities.device, dtype=probabilities.dtype)
    if int(probabilities.shape[1]) != int(ddi_upper.shape[0]):
        raise ValueError(
            "drug_scores width must match ddi_matrix width: "
            f"got {int(probabilities.shape[1])} and {int(ddi_upper.shape[0])}"
        )

    batch_size, drug_vocab_size = probabilities.shape
    selected_masks = torch.zeros(batch_size, drug_vocab_size, dtype=torch.bool, device=probabilities.device)
    utility_scores = torch.zeros(batch_size, dtype=probabilities.dtype, device=probabilities.device)
    ddi_penalties = torch.zeros(batch_size, dtype=probabilities.dtype, device=probabilities.device)
    size_penalties = torch.zeros(batch_size, dtype=probabilities.dtype, device=probabilities.device)
    total_scores = torch.zeros(batch_size, dtype=probabilities.dtype, device=probabilities.device)
    ddi_rates = torch.zeros(batch_size, dtype=probabilities.dtype, device=probabilities.device)
    num_ddi_pairs = torch.zeros(batch_size, dtype=probabilities.dtype, device=probabilities.device)
    num_predicted_drugs = torch.zeros(batch_size, dtype=probabilities.dtype, device=probabilities.device)
    selected_indices: list[list[int]] = []

    resolved_ensure_non_empty = (
        bool(mode == "soft_constrained_rerank") if ensure_non_empty is None else bool(ensure_non_empty)
    )

    for sample_index in range(batch_size):
        sample_probs = probabilities[sample_index]
        if mode == "threshold":
            sample_mask = sample_probs >= float(threshold)
        elif mode == "topk":
            if top_k is None:
                raise ValueError("top_k must be provided when decode_mode='topk'")
            sample_mask = _binary_topk(sample_probs, int(top_k))
        else:
            sample_mask, _ = _rerank_single_prediction_set(
                sample_probs,
                ddi_upper,
                threshold=threshold,
                top_m=top_m,
                alpha_utility=alpha_utility,
                beta_ddi=beta_ddi,
                gamma_size=gamma_size,
                min_drugs=min_drugs,
                max_drugs=max_drugs,
                target_avg_drugs=target_avg_drugs,
                ensure_non_empty=resolved_ensure_non_empty,
            )

        if resolved_ensure_non_empty and mode != "topk" and not bool(sample_mask.any().item()):
            sample_mask = _binary_topk(sample_probs, 1)

        stats = score_candidate_set(
            sample_probs,
            sample_mask,
            ddi_upper,
            input_is_logits=False,
            alpha_utility=alpha_utility,
            beta_ddi=beta_ddi,
            gamma_size=gamma_size,
            min_drugs=min_drugs,
            max_drugs=max_drugs,
            target_avg_drugs=target_avg_drugs,
        )
        selected_masks[sample_index] = sample_mask
        selected_indices.append(torch.nonzero(sample_mask, as_tuple=False).flatten().tolist())
        utility_scores[sample_index] = float(stats["utility_score"])
        ddi_penalties[sample_index] = float(stats["ddi_penalty"])
        size_penalties[sample_index] = float(stats["size_penalty"])
        total_scores[sample_index] = float(stats["total_score"])
        ddi_rates[sample_index] = float(stats["ddi_rate"])
        num_ddi_pairs[sample_index] = float(stats["num_ddi_pairs"])
        num_predicted_drugs[sample_index] = float(stats["num_predicted_drugs"])

    return {
        "prediction_mask": selected_masks,
        "selected_mask": selected_masks,
        "selected_indices": selected_indices,
        "utility_score": utility_scores,
        "ddi_penalty": ddi_penalties,
        "size_penalty": size_penalties,
        "total_score": total_scores,
        "ddi_rate_per_sample": ddi_rates,
        "num_ddi_pairs": num_ddi_pairs,
        "num_predicted_drugs": num_predicted_drugs,
        "drug_probs": probabilities,
        "decode_mode": mode,
    }


def compute_ddi_penalty(
    drug_scores: torch.Tensor,
    ddi_matrix: torch.Tensor,
    *,
    input_is_logits: bool = False,
    normalize: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute expected harmful co-medication strength from predicted scores.

    Parameters
    ----------
    drug_scores:
        Raw logits or probabilities with shape ``(B, D)``.
    ddi_matrix:
        Binary or weighted DDI adjacency matrix with shape ``(D, D)``.
    input_is_logits:
        When ``True``, apply ``sigmoid`` before computing pair strength.
    normalize:
        When ``True``, divide each sample penalty by the number of interacting
        pairs present in the DDI matrix. The default keeps the raw expected
        interaction mass so upstream loss weighting remains explicit.
    reduction:
        One of ``"none"``, ``"mean"``, ``"batchmean"``, or ``"sum"``.
    """

    resolved_reduction = _validate_reduction(reduction)
    resolved_probs = _resolve_probabilities(drug_scores, input_is_logits=input_is_logits)
    resolved_ddi = load_ddi_matrix(
        ddi_matrix,
        device=resolved_probs.device,
        dtype=resolved_probs.dtype,
    )
    if int(resolved_probs.shape[1]) != int(resolved_ddi.shape[0]):
        raise ValueError(
            "drug_scores width must match the DDI matrix width: "
            f"got {int(resolved_probs.shape[1])} and {int(resolved_ddi.shape[0])}"
        )

    ddi_upper = torch.triu(resolved_ddi, diagonal=1)
    pair_probs = resolved_probs.unsqueeze(2) * resolved_probs.unsqueeze(1)
    penalties = (pair_probs * ddi_upper.unsqueeze(0)).sum(dim=(1, 2))
    if bool(normalize):
        penalties = penalties / ddi_upper.sum().clamp(min=1.0)
    return _reduce_penalties(penalties, resolved_reduction)


def compute_ddi_loss(
    drug_probs: torch.Tensor,
    ddi_matrix: torch.Tensor,
    *,
    input_is_logits: bool = False,
    normalize: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """Backward-compatible alias for DDI regularization."""

    return compute_ddi_penalty(
        drug_probs,
        ddi_matrix,
        input_is_logits=input_is_logits,
        normalize=normalize,
        reduction=reduction,
    )


class DDIRegularizer(nn.Module):
    """Differentiable DDI penalty over predicted co-medication probabilities.

    Notes
    -----
    Input shape:
        ``drug_scores`` must have shape ``(B, D)``.

    Output shape:
        - ``reduction="mean"``, ``"batchmean"``, or ``"sum"``: scalar tensor
        - ``reduction="none"``: tensor with shape ``(B,)``
    """

    def __init__(
        self,
        ddi_source: str | Path | Mapping[str, Any] | torch.Tensor,
        *,
        input_is_logits: bool = False,
        normalize: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reduction = _validate_reduction(reduction)
        self.input_is_logits = bool(input_is_logits)
        self.normalize = bool(normalize)

        ddi_matrix = load_ddi_matrix(ddi_source, dtype=torch.float32)
        ddi_upper = torch.triu(ddi_matrix, diagonal=1)
        pair_normalizer = ddi_upper.sum().clamp(min=1.0)

        self.register_buffer("ddi_upper", ddi_upper)
        self.register_buffer("pair_normalizer", pair_normalizer)

    @property
    def drug_vocab_size(self) -> int:
        return int(self.ddi_upper.shape[0])

    def compute_penalty_per_sample(
        self,
        drug_scores: torch.Tensor,
        *,
        input_is_logits: bool | None = None,
    ) -> torch.Tensor:
        """Compute per-sample expected harmful-pair mass.

        Parameters
        ----------
        drug_scores:
            Predicted medication logits or probabilities with shape ``(B, D)``.
        input_is_logits:
            Optional override for whether the input should be passed through
            ``sigmoid`` before computing DDI pair strength.

        Returns
        -------
        torch.Tensor
            Per-sample DDI penalty with shape ``(B,)``.
        """

        resolved_input_is_logits = self.input_is_logits if input_is_logits is None else bool(input_is_logits)
        resolved_probs = _resolve_probabilities(drug_scores, input_is_logits=resolved_input_is_logits)
        if resolved_probs.shape[1] != self.drug_vocab_size:
            raise ValueError(
                "drug_scores width must match the DDI matrix width: "
                f"expected {self.drug_vocab_size}, got {int(resolved_probs.shape[1])}"
            )

        resolved_probs = resolved_probs.to(device=self.ddi_upper.device, dtype=self.ddi_upper.dtype)
        pair_probs = resolved_probs.unsqueeze(2) * resolved_probs.unsqueeze(1)
        penalties = (pair_probs * self.ddi_upper.unsqueeze(0)).sum(dim=(1, 2))
        if self.normalize:
            penalties = penalties / self.pair_normalizer
        return penalties

    def forward(self, drug_scores: torch.Tensor, *, input_is_logits: bool | None = None) -> torch.Tensor:
        penalties = self.compute_penalty_per_sample(drug_scores, input_is_logits=input_is_logits)
        return _reduce_penalties(penalties, self.reduction)


__all__ = [
    "DDIRegularizer",
    "compute_ddi_loss",
    "compute_ddi_penalty",
    "compute_ddi_penalty_for_set",
    "load_ddi_matrix",
    "rerank_prediction_set",
    "score_candidate_set",
]
