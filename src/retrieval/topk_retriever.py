from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.temporal_similarity import TemporalSimilarity
from src.retrieval.faiss_index import VisitFaissIndex
from src.retrieval.memory_bank import VisitMemoryBank


class TopKVisitRetriever(nn.Module):
    """Retrieve similar historical visits and aggregate only medication evidence.

    Leakage policy:
    - exact same visit can be blocked by `(patient_id, visit_index)`
    - same-patient future visits can be blocked by visit order
    - cross-patient future filtering is only claimed when absolute visit time is available
    """

    def __init__(
        self,
        hidden_dim: int,
        drug_vocab_size: int,
        *,
        top_k: int = 5,
        backend: str = "bruteforce",
        use_faiss_if_available: bool = True,
        similarity_mode: str = "cosine_decay",
        temporal_decay_alpha: float = 0.05,
        allow_same_patient: bool = False,
        exclude_future: bool = True,
        exclude_exact_match: bool = True,
        exclude_future_same_patient: bool | None = None,
        exclude_future_all_patients_if_absolute_time: bool = True,
        require_absolute_time_for_cross_patient_temporal_filter: bool = False,
        use_time_gap: bool = True,
        dropout: float = 0.1,
        coarse_search_multiplier: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.drug_vocab_size = int(drug_vocab_size)
        self.top_k = int(top_k)
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k!r}")
        self.allow_same_patient = bool(allow_same_patient)
        self.exclude_future = bool(exclude_future)
        self.exclude_future_same_patient = bool(
            exclude_future if exclude_future_same_patient is None else exclude_future_same_patient
        )
        self.exclude_future_all_patients_if_absolute_time = bool(
            exclude_future_all_patients_if_absolute_time
        )
        self.require_absolute_time_for_cross_patient_temporal_filter = bool(
            require_absolute_time_for_cross_patient_temporal_filter
        )
        self.exclude_exact_match = bool(exclude_exact_match)
        self.use_time_gap = bool(use_time_gap)
        self.coarse_search_multiplier = max(int(coarse_search_multiplier), 1)
        self.search_index = VisitFaissIndex(
            backend=backend,
            use_faiss_if_available=use_faiss_if_available,
        )
        self.temporal_similarity = TemporalSimilarity(
            hidden_dim=self.hidden_dim,
            similarity_mode=similarity_mode,
            temporal_decay_alpha=temporal_decay_alpha,
        )
        self.medication_projection = nn.Sequential(
            nn.Linear(self.drug_vocab_size, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(self.hidden_dim),
        )
        self.memory_bank: VisitMemoryBank | None = None

    @property
    def retrieval_enabled(self) -> bool:
        """Compatibility helper; retained for older retrieval/memory-bank workflows."""

        return self.memory_bank is not None and self.memory_bank.num_visits > 0

    def describe_leakage_policy(
        self,
        *,
        memory_bank: VisitMemoryBank | None = None,
    ) -> dict[str, Any]:
        resolved_bank = memory_bank if memory_bank is not None else self.memory_bank
        if resolved_bank is None:
            return {
                "memory_bank_split": None,
                "has_absolute_time": False,
                "all_visits_have_absolute_time": False,
                "exact_match_blocked": bool(self.exclude_exact_match),
                "same_patient_future_blocked": bool(self.exclude_future_same_patient),
                "cross_patient_absolute_temporal_filter": False,
                "require_absolute_time_for_cross_patient_temporal_filter": bool(
                    self.require_absolute_time_for_cross_patient_temporal_filter
                ),
                "notes": "Retrieval memory bank is not initialized yet.",
            }
        cross_patient_absolute_temporal_filter = bool(
            self.exclude_future_all_patients_if_absolute_time
            and resolved_bank.has_absolute_time
            and (
                resolved_bank.all_visits_have_absolute_time
                or self.require_absolute_time_for_cross_patient_temporal_filter
            )
        )
        policy = resolved_bank.describe_temporal_policy(
            exact_match_blocked=self.exclude_exact_match,
            same_patient_future_blocked=self.exclude_future_same_patient,
            cross_patient_absolute_temporal_filter=cross_patient_absolute_temporal_filter,
            require_absolute_time_for_cross_patient_temporal_filter=(
                self.require_absolute_time_for_cross_patient_temporal_filter
            ),
        )
        policy.update(
            {
                "allow_same_patient": bool(self.allow_same_patient),
                "retrieval_backend": str(self.search_index.backend),
            }
        )
        return policy

    def set_memory_bank(self, memory_bank: VisitMemoryBank | None) -> None:
        self.memory_bank = memory_bank
        if memory_bank is None or memory_bank.num_visits <= 0:
            self.search_index.build_index(torch.empty(0, self.hidden_dim, dtype=torch.float32))
            return
        embeddings = memory_bank.export_embeddings()
        if embeddings.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Memory bank embedding width must match retriever hidden_dim={self.hidden_dim}, got {tuple(embeddings.shape)}"
            )
        self.search_index.build_index(embeddings)

    def aggregate_medication_context(
        self,
        *,
        medication_evidence: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if medication_evidence.ndim != 2:
            raise ValueError(
                f"medication_evidence must have shape (K, D), got {tuple(medication_evidence.shape)}"
            )
        if weights.ndim != 1 or weights.shape[0] != medication_evidence.shape[0]:
            raise ValueError(
                "weights must have shape (K,) aligned with medication_evidence: "
                f"got {tuple(weights.shape)} and {tuple(medication_evidence.shape)}"
            )
        raw_context = torch.matmul(weights.unsqueeze(0), medication_evidence).squeeze(0)
        projected_context = self.medication_projection(raw_context.unsqueeze(0)).squeeze(0)
        return raw_context, projected_context

    def _empty_outputs(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        return {
            "aggregated_retrieval_context": torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
            "retrieval_medication_context": torch.zeros(batch_size, self.drug_vocab_size, device=device, dtype=dtype),
            "retrieved_indices": torch.full((batch_size, self.top_k), -1, device=device, dtype=torch.long),
            "retrieved_scores": torch.zeros(batch_size, self.top_k, device=device, dtype=dtype),
            "retrieval_weights": torch.zeros(batch_size, self.top_k, device=device, dtype=dtype),
            "retrieved_medication_evidence": torch.zeros(
                batch_size,
                self.top_k,
                self.drug_vocab_size,
                device=device,
                dtype=dtype,
            ),
            "retrieved_metadata": [[] for _ in range(batch_size)],
            "valid_candidate_counts": torch.zeros(batch_size, device=device, dtype=torch.long),
            "avg_valid_candidates": 0.0,
            "avg_retrieved_score": 0.0,
        }

    def retrieve(
        self,
        *,
        current_state: torch.Tensor,
        current_patient_ids: torch.Tensor | list[int],
        current_visit_indices: torch.Tensor | list[int],
        current_visit_times: torch.Tensor | None = None,
        current_visit_times_are_absolute: torch.Tensor | list[bool] | None = None,
        memory_bank: VisitMemoryBank | None = None,
        return_metadata: bool = True,
    ) -> dict[str, Any]:
        if current_state.ndim != 2:
            raise ValueError(f"current_state must have shape (B, H), got {tuple(current_state.shape)}")
        if current_state.shape[1] != self.hidden_dim:
            raise ValueError(
                f"current_state hidden dim must equal {self.hidden_dim}, got {int(current_state.shape[1])}"
            )

        resolved_bank = memory_bank if memory_bank is not None else self.memory_bank
        batch_size = current_state.shape[0]
        if resolved_bank is None or resolved_bank.num_visits <= 0:
            return self._empty_outputs(batch_size=batch_size, device=current_state.device, dtype=current_state.dtype)

        patient_ids = torch.as_tensor(current_patient_ids, device=current_state.device, dtype=torch.long).reshape(-1)
        visit_indices = torch.as_tensor(current_visit_indices, device=current_state.device, dtype=torch.long).reshape(-1)
        if patient_ids.shape[0] != batch_size or visit_indices.shape[0] != batch_size:
            raise ValueError(
                "current_patient_ids and current_visit_indices must align with batch size: "
                f"got {patient_ids.shape[0]}, {visit_indices.shape[0]}, batch_size={batch_size}"
            )
        visit_times = None
        if current_visit_times is not None:
            visit_times = torch.as_tensor(current_visit_times, device=current_state.device, dtype=torch.float32).reshape(-1)
            if visit_times.shape[0] != batch_size:
                raise ValueError(
                    f"current_visit_times must align with batch size {batch_size}, got {visit_times.shape[0]}"
                )
        visit_times_are_absolute = None
        if current_visit_times_are_absolute is not None:
            visit_times_are_absolute = torch.as_tensor(
                current_visit_times_are_absolute,
                device=current_state.device,
                dtype=torch.bool,
            ).reshape(-1)
            if visit_times_are_absolute.shape[0] != batch_size:
                raise ValueError(
                    "current_visit_times_are_absolute must align with batch size "
                    f"{batch_size}, got {visit_times_are_absolute.shape[0]}"
                )

        outputs = self._empty_outputs(batch_size=batch_size, device=current_state.device, dtype=current_state.dtype)
        coarse_pool = None
        if self.search_index.is_built:
            coarse_pool = self.search_index.search(
                current_state.detach(),
                top_k=min(
                    max(self.top_k * self.coarse_search_multiplier, self.top_k),
                    max(resolved_bank.num_visits, 1),
                ),
            )

        non_empty_scores: list[torch.Tensor] = []
        retrieved_metadata: list[list[dict[str, Any]]] = []
        for batch_index in range(batch_size):
            candidate_pool = resolved_bank.get_candidate_pool(
                patient_id=int(patient_ids[batch_index].item()),
                visit_index=int(visit_indices[batch_index].item()),
                visit_time=None if visit_times is None else float(visit_times[batch_index].item()),
                query_has_absolute_time=bool(
                    visit_times is not None
                    and visit_times_are_absolute is not None
                    and visit_times_are_absolute[batch_index].item()
                ),
                allow_same_patient=self.allow_same_patient,
                exclude_future=self.exclude_future_same_patient,
                exclude_exact_match=self.exclude_exact_match,
                exclude_future_all_patients_if_absolute_time=(
                    self.exclude_future_all_patients_if_absolute_time
                ),
                require_absolute_time_for_cross_patient_temporal_filter=(
                    self.require_absolute_time_for_cross_patient_temporal_filter
                ),
            )
            if coarse_pool is not None and candidate_pool["indices"].numel() > 0:
                coarse_indices = coarse_pool["indices"][batch_index]
                coarse_mask = torch.isin(candidate_pool["indices"], coarse_indices)
                if bool(coarse_mask.any().item()):
                    keep = coarse_mask
                    candidate_pool = {
                        key: value[keep]
                        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == coarse_mask.shape[0]
                        else [item for item, flag in zip(value, keep.tolist()) if flag]
                        if isinstance(value, list)
                        else value
                        for key, value in candidate_pool.items()
                    }
                elif candidate_pool["indices"].numel() > self.top_k:
                    # FAISS can drop all valid points after temporal filtering; fall back to the full filtered pool.
                    candidate_pool = resolved_bank.get_candidate_pool(
                        patient_id=int(patient_ids[batch_index].item()),
                        visit_index=int(visit_indices[batch_index].item()),
                        visit_time=None if visit_times is None else float(visit_times[batch_index].item()),
                        query_has_absolute_time=bool(
                            visit_times is not None
                            and visit_times_are_absolute is not None
                            and visit_times_are_absolute[batch_index].item()
                        ),
                        allow_same_patient=self.allow_same_patient,
                        exclude_future=self.exclude_future_same_patient,
                        exclude_exact_match=self.exclude_exact_match,
                        exclude_future_all_patients_if_absolute_time=(
                            self.exclude_future_all_patients_if_absolute_time
                        ),
                        require_absolute_time_for_cross_patient_temporal_filter=(
                            self.require_absolute_time_for_cross_patient_temporal_filter
                        ),
                    )

            num_candidates = int(candidate_pool["indices"].numel())
            outputs["valid_candidate_counts"][batch_index] = num_candidates
            if num_candidates <= 0:
                retrieved_metadata.append([])
                continue

            candidate_embeddings = candidate_pool["visit_embeddings"].to(
                device=current_state.device,
                dtype=current_state.dtype,
            )
            use_absolute_time_gap = bool(
                self.use_time_gap
                and visit_times is not None
                and visit_times_are_absolute is not None
                and visit_times_are_absolute[batch_index].item()
                and bool(candidate_pool["has_absolute_time"].all().item())
            )
            candidate_times = (
                candidate_pool["visit_times"].to(device=current_state.device, dtype=current_state.dtype)
                if use_absolute_time_gap
                else None
            )
            candidate_indices = candidate_pool["visit_indices"].to(device=current_state.device, dtype=torch.long)
            similarity_outputs = self.temporal_similarity(
                query_embeddings=current_state[batch_index : batch_index + 1],
                candidate_embeddings=candidate_embeddings,
                query_times=None if not use_absolute_time_gap else visit_times[batch_index : batch_index + 1],
                candidate_times=candidate_times,
                query_indices=visit_indices[batch_index : batch_index + 1].to(dtype=current_state.dtype),
                candidate_indices=candidate_indices.to(dtype=current_state.dtype),
            )
            final_score = similarity_outputs["final_score"].squeeze(0)
            resolved_top_k = min(self.top_k, int(final_score.shape[0]))
            top_scores, local_top_indices = torch.topk(final_score, k=resolved_top_k, dim=-1)
            weights = torch.softmax(top_scores, dim=-1)
            global_indices = candidate_pool["indices"][local_top_indices.cpu()].to(device=current_state.device, dtype=torch.long)
            medication_evidence = candidate_pool["medication_evidence"][local_top_indices.cpu()].to(
                device=current_state.device,
                dtype=current_state.dtype,
            )
            raw_context, projected_context = self.aggregate_medication_context(
                medication_evidence=medication_evidence,
                weights=weights,
            )

            outputs["aggregated_retrieval_context"][batch_index] = projected_context
            outputs["retrieval_medication_context"][batch_index] = raw_context
            outputs["retrieved_indices"][batch_index, :resolved_top_k] = global_indices
            outputs["retrieved_scores"][batch_index, :resolved_top_k] = top_scores
            outputs["retrieval_weights"][batch_index, :resolved_top_k] = weights
            outputs["retrieved_medication_evidence"][batch_index, :resolved_top_k] = medication_evidence
            if return_metadata:
                retrieved_metadata.append(
                    [
                        {
                            "patient_id": int(candidate_pool["patient_ids"][int(index)].item()),
                            "visit_index": int(candidate_pool["visit_indices"][int(index)].item()),
                            "visit_time": float(candidate_pool["visit_times"][int(index)].item()),
                            **dict(candidate_pool["metadata"][int(index)]),
                        }
                        for index in local_top_indices.cpu().tolist()
                    ]
                )
            else:
                retrieved_metadata.append([])
            non_empty_scores.append(top_scores.detach().cpu())

        outputs["retrieved_metadata"] = retrieved_metadata
        if non_empty_scores:
            stacked_scores = torch.cat(non_empty_scores)
            outputs["avg_retrieved_score"] = float(stacked_scores.mean().item())
        outputs["avg_valid_candidates"] = float(outputs["valid_candidate_counts"].to(dtype=torch.float32).mean().item())
        return outputs


__all__ = ["TopKVisitRetriever"]
