from __future__ import annotations

import copy
import time
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from src.graph.group_encoder import GroupEncoder
from src.models.ddi_regularization import DDIRegularizer
from src.models.fusion import FusionModule
from src.models.history_selector import HistorySelector
from src.models.medication_decoder import MedicationDecoder
from src.models.patient_state_encoder import PatientStateEncoder
from src.retrieval.memory_bank import MemoryBank, build_last_visit_queries
from src.retrieval.topk_retriever import retrieve_topk
from src.utils.runtime_truth import (
    build_core_runtime_truth,
    build_extension_runtime_truth,
    ddi_truth_fields,
    normalize_ddi_context,
)


_VALID_MODES = {"core", "extended"}


def _resolve_mode(default_mode: str, override: str | None) -> str:
    resolved = default_mode if override is None else override
    if resolved not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {resolved!r}")
    return resolved


def _extract_last_visit_state(
    state_sequence: torch.Tensor,
    visit_mask: torch.Tensor,
) -> torch.Tensor:
    if state_sequence.ndim != 3:
        raise ValueError(f"state_sequence must have shape (B, T, H), got {tuple(state_sequence.shape)}")
    if visit_mask.ndim != 2:
        raise ValueError(f"visit_mask must have shape (B, T), got {tuple(visit_mask.shape)}")
    if tuple(state_sequence.shape[:2]) != tuple(visit_mask.shape):
        raise ValueError(
            "state_sequence and visit_mask must align on batch/time dimensions: "
            f"got {tuple(state_sequence.shape[:2])} and {tuple(visit_mask.shape)}"
        )

    resolved_mask = visit_mask.to(device=state_sequence.device, dtype=torch.bool)
    valid_counts = resolved_mask.sum(dim=1)
    if bool((valid_counts <= 0).any().item()):
        raise ValueError("Each sample must contain at least one valid visit")

    last_indices = valid_counts.to(dtype=torch.long) - 1
    batch_indices = torch.arange(state_sequence.shape[0], device=state_sequence.device)
    return state_sequence[batch_indices, last_indices]


def _extract_last_valid_targets(
    target_drugs: torch.Tensor,
    visit_mask: torch.Tensor,
) -> torch.Tensor:
    if target_drugs.ndim == 2:
        return target_drugs
    if target_drugs.ndim != 3:
        raise ValueError(f"target_drugs must have shape (B, D) or (B, T, D), got {tuple(target_drugs.shape)}")
    if visit_mask.ndim != 2:
        raise ValueError(f"visit_mask must have shape (B, T), got {tuple(visit_mask.shape)}")
    if tuple(target_drugs.shape[:2]) != tuple(visit_mask.shape):
        raise ValueError(
            "target_drugs and visit_mask must align on batch/time dimensions: "
            f"got {tuple(target_drugs.shape[:2])} and {tuple(visit_mask.shape)}"
        )

    resolved_mask = visit_mask.to(device=target_drugs.device, dtype=torch.bool)
    valid_counts = resolved_mask.sum(dim=1)
    if bool((valid_counts <= 0).any().item()):
        raise ValueError("Each sample must contain at least one valid visit")

    last_indices = valid_counts.to(dtype=torch.long) - 1
    batch_indices = torch.arange(target_drugs.shape[0], device=target_drugs.device)
    return target_drugs[batch_indices, last_indices]


class RetrievalEvidenceFusionModel(nn.Module):
    """End-to-end recommendation model for verified core mode and experimental extensions."""

    def __init__(
        self,
        encoder: PatientStateEncoder,
        history_selector: HistorySelector,
        fusion_module: FusionModule,
        *,
        group_encoder: GroupEncoder | None = None,
        medication_decoder: MedicationDecoder | None = None,
        ddi_regularizer: DDIRegularizer | None = None,
        ddi_context: Mapping[str, Any] | None = None,
        mode: str = "core",
        retrieval_top_k: int = 5,
        temporal_decay_alpha: float = 0.05,
        retrieval_backend: str = "bruteforce",
        use_faiss_if_available: bool = True,
        allow_cross_split: bool = False,
        retrieval_scoring_mode: str = "temporal_relevance",
        cross_split_policy: str | None = None,
        core_retrieval_enabled: bool = False,
        retrieval_leakage_safe: bool = True,
    ) -> None:
        super().__init__()
        if int(retrieval_top_k) <= 0:
            raise ValueError(f"retrieval_top_k must be positive, got {retrieval_top_k!r}")
        if float(temporal_decay_alpha) <= 0.0:
            raise ValueError(f"temporal_decay_alpha must be > 0, got {temporal_decay_alpha!r}")

        self.encoder = encoder
        self.history_selector = history_selector
        self.fusion_module = fusion_module
        self.group_encoder = group_encoder
        self.medication_decoder = medication_decoder
        self.ddi_regularizer = ddi_regularizer
        self.ddi_context = normalize_ddi_context(ddi_context)
        self.mode = _resolve_mode(mode, None)
        self.retrieval_top_k = int(retrieval_top_k)
        self.temporal_decay_alpha = float(temporal_decay_alpha)
        self.retrieval_backend = retrieval_backend
        self.use_faiss_if_available = bool(use_faiss_if_available)
        self.allow_cross_split = bool(allow_cross_split)
        self.retrieval_scoring_mode = str(retrieval_scoring_mode)
        self.cross_split_policy = None if cross_split_policy is None else str(cross_split_policy)
        self.core_retrieval_enabled = bool(core_retrieval_enabled)
        self.retrieval_leakage_safe = bool(retrieval_leakage_safe)

    def _base_safety_metadata(self, *, mode: str, retrieval_used: bool) -> dict[str, Any]:
        ddi_truth = ddi_truth_fields(self.ddi_context)
        ddi_active = bool(ddi_truth["ddi_active"])
        return {
            **ddi_truth,
            "ddi_available": ddi_active,
            "ddi_regularizer_present": self.ddi_regularizer is not None,
            "ddi_reason": self.ddi_context.get("reason", "available" if ddi_active else "unavailable"),
            "matched_pairs": self.ddi_context.get("matched_pairs"),
            "nonzero_pairs": self.ddi_context.get("nonzero_pairs"),
            "vocab_size": self.ddi_context.get("vocab_size"),
            "source_metadata": copy.deepcopy(dict(self.ddi_context.get("source_metadata") or {})),
            "mode": mode,
            "retrieval_used": retrieval_used,
        }

    def _build_runtime_truth(
        self,
        *,
        mode: str,
        retrieval_used: bool,
        retrieval_payload_source: str,
        fusion_strategy: str,
    ) -> dict[str, Any]:
        if mode == "core":
            runtime_truth = build_core_runtime_truth(
                fusion_strategy=fusion_strategy,
                ddi_context=self.ddi_context,
                retrieval_active=retrieval_used,
                retrieval_status="active" if retrieval_used else "disabled",
                retrieval_top_k=self.retrieval_top_k if self.core_retrieval_enabled else None,
                retrieval_scoring_mode=self.retrieval_scoring_mode if self.core_retrieval_enabled else None,
                retrieval_cross_split_policy=(
                    self.cross_split_policy or ("allow_all" if self.allow_cross_split else "same_split")
                )
                if self.core_retrieval_enabled
                else None,
                retrieval_leakage_safe=self.retrieval_leakage_safe if self.core_retrieval_enabled else None,
            )
        else:
            runtime_truth = build_extension_runtime_truth(
                fusion_strategy=fusion_strategy,
                ddi_context=self.ddi_context,
                retrieval_active=retrieval_used,
                group_encoder_active=self.group_encoder is not None,
                retrieval_scoring_mode=self.retrieval_scoring_mode,
                retrieval_cross_split_policy=self.cross_split_policy
                or ("allow_all" if self.allow_cross_split else "same_split"),
            )
        runtime_truth["retrieval_mode"] = str(self.retrieval_scoring_mode if retrieval_used else retrieval_payload_source)
        runtime_truth["retrieval_payload_source"] = str(retrieval_payload_source)
        runtime_truth["retrieval_scoring_mode"] = str(self.retrieval_scoring_mode)
        runtime_truth["retrieval_cross_split_policy"] = str(
            self.cross_split_policy or ("allow_all" if self.allow_cross_split else "same_split")
        )
        return runtime_truth

    def _build_retrieval_payload_if_possible(
        self,
        encoder_outputs: Mapping[str, torch.Tensor],
        *,
        mode: str,
        retrieval_payload: Mapping[str, Any] | None,
        memory_bank: MemoryBank | None,
        records: Sequence[Mapping[str, Any]] | None,
        query_metadata: Mapping[str, Any] | None,
        query_states: torch.Tensor | None,
    ) -> tuple[Mapping[str, Any] | None, str, dict[str, float]]:
        if retrieval_payload is not None:
            return retrieval_payload, "provided", {"retrieval_time": 0.0}
        if mode == "core" and not self.core_retrieval_enabled:
            return None, "disabled_in_core", {"retrieval_time": 0.0}
        if mode not in {"core", "extended"}:
            return None, "disabled_unknown_mode", {"retrieval_time": 0.0}
        if memory_bank is None:
            return None, "disabled_no_memory_bank", {"retrieval_time": 0.0}

        resolved_query_states: torch.Tensor | None = None
        resolved_query_metadata: Mapping[str, Any] | None = None

        if records is not None:
            resolved_query_states, resolved_query_metadata = build_last_visit_queries(
                records,
                encoder_outputs,
            )
        elif query_metadata is not None:
            resolved_query_metadata = query_metadata
            if query_states is None:
                resolved_query_states = _extract_last_visit_state(
                    encoder_outputs["state_sequence"],
                    encoder_outputs["visit_mask"],
                ).detach().cpu()
            else:
                resolved_query_states = torch.as_tensor(query_states, dtype=torch.float32).cpu()

        if resolved_query_states is None or resolved_query_metadata is None:
            return None, "disabled_no_queries", {"retrieval_time": 0.0}

        retrieval_start = time.perf_counter()
        payload = retrieve_topk(
            resolved_query_states,
            resolved_query_metadata,
            memory_bank,
            top_k=self.retrieval_top_k,
            temporal_decay_alpha=self.temporal_decay_alpha,
            backend=self.retrieval_backend,
            use_faiss_if_available=self.use_faiss_if_available,
            allow_cross_split=self.allow_cross_split,
            cross_split_policy=self.cross_split_policy,
            scoring_mode=self.retrieval_scoring_mode,
        )
        return payload, "built", {"retrieval_time": time.perf_counter() - retrieval_start}

    def _build_group_outputs(
        self,
        *,
        current_state: torch.Tensor,
        retrieval_payload: Mapping[str, Any] | None,
        memory_bank: MemoryBank | None,
    ) -> dict[str, Any]:
        group_outputs: dict[str, Any] = {
            "group_context": None,
            "group_available_mask": torch.zeros(
                current_state.shape[0],
                dtype=torch.bool,
                device=current_state.device,
            ),
            "group_metadata": [],
        }
        if self.group_encoder is None or retrieval_payload is None or memory_bank is None:
            return group_outputs
        return self.group_encoder(
            current_state=current_state,
            retrieval_payload=retrieval_payload,
            memory_bank=memory_bank,
        )

    def _build_ddi_outputs(
        self,
        *,
        drug_probs: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
        mode: str,
        retrieval_used: bool,
        compute_ddi_metrics: bool,
    ) -> dict[str, Any]:
        safety_metadata = self._base_safety_metadata(mode=mode, retrieval_used=retrieval_used)
        if self.ddi_regularizer is None or drug_probs is None or not compute_ddi_metrics:
            return {
                "ddi_penalty_per_sample": None,
                "ddi_penalty_mean": None,
                "safety_metadata": safety_metadata,
            }

        ddi_penalty_per_sample = self.ddi_regularizer.compute_penalty_per_sample(drug_probs)
        ddi_penalty_per_sample = ddi_penalty_per_sample.to(device=device, dtype=dtype)
        ddi_penalty_mean = ddi_penalty_per_sample.mean()
        return {
            "ddi_penalty_per_sample": ddi_penalty_per_sample,
            "ddi_penalty_mean": ddi_penalty_mean,
            "safety_metadata": safety_metadata,
        }

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        retrieval_payload: Mapping[str, Any] | None = None,
        memory_bank: MemoryBank | None = None,
        records: Sequence[Mapping[str, Any]] | None = None,
        query_metadata: Mapping[str, Any] | None = None,
        query_states: torch.Tensor | None = None,
        mode: str | None = None,
        attribute_payload: Mapping[str, Any] | None = None,
        decoder_top_k: int | None = None,
        compute_ddi_metrics: bool = True,
    ) -> dict[str, Any]:
        encoder_outputs = self.encoder(dict(batch))
        current_state = encoder_outputs["pooled_state"]
        resolved_mode = _resolve_mode(self.mode, mode)

        resolved_retrieval_payload, retrieval_mode, runtime_timing = self._build_retrieval_payload_if_possible(
            encoder_outputs,
            mode=resolved_mode,
            retrieval_payload=retrieval_payload,
            memory_bank=memory_bank,
            records=records,
            query_metadata=query_metadata,
            query_states=query_states,
        )
        retrieval_available = resolved_retrieval_payload is not None
        retrieval_used = retrieval_available and memory_bank is not None

        group_outputs = self._build_group_outputs(
            current_state=current_state,
            retrieval_payload=resolved_retrieval_payload if retrieval_used else None,
            memory_bank=memory_bank if retrieval_used else None,
        )

        selection_outputs = self.history_selector(
            current_state=current_state,
            state_sequence=encoder_outputs["state_sequence"],
            visit_mask=encoder_outputs["visit_mask"],
            retrieval_payload=resolved_retrieval_payload,
            memory_bank=memory_bank,
            group_context=group_outputs.get("group_context"),
            group_available_mask=group_outputs.get("group_available_mask"),
            attribute_payload=attribute_payload,
        )
        evidence_metadata = dict(selection_outputs["evidence_metadata"])
        evidence_metadata["group_metadata"] = group_outputs.get("group_metadata", [])

        fusion_outputs = self.fusion_module(
            current_state=current_state,
            self_history_context=selection_outputs["self_history_context"],
            neighbor_history_context=selection_outputs["neighbor_history_context"],
            group_context=selection_outputs.get("group_context"),
            branch_masks={
                "current": torch.ones(
                    current_state.shape[0],
                    dtype=torch.bool,
                    device=current_state.device,
                ),
                "self": evidence_metadata["self_history_available_mask"],
                "neighbor": evidence_metadata["neighbor_available_mask"],
                "group": evidence_metadata["group_available_mask"],
            },
        )

        drug_logits: torch.Tensor | None
        drug_probs: torch.Tensor | None
        recommendation_metadata: dict[str, Any]
        if self.medication_decoder is None:
            drug_logits = None
            drug_probs = None
            recommendation_metadata = {
                "decoder_available": False,
                "mode": resolved_mode,
                "retrieval_used": retrieval_used,
            }
        else:
            decoder_outputs = self.medication_decoder(
                fusion_outputs["fused_repr"],
                top_k=decoder_top_k,
            )
            drug_logits = decoder_outputs["drug_logits"]
            drug_probs = decoder_outputs["drug_probs"]
            decoder_metadata = dict(decoder_outputs.get("recommendation_metadata", {}))
            recommendation_metadata = {
                "decoder_available": True,
                "mode": resolved_mode,
                "retrieval_used": retrieval_used,
                "decoder_metadata": decoder_metadata,
                **decoder_metadata,
            }

        ddi_outputs = self._build_ddi_outputs(
            drug_probs=drug_probs,
            device=current_state.device,
            dtype=current_state.dtype,
            mode=resolved_mode,
            retrieval_used=retrieval_used,
            compute_ddi_metrics=compute_ddi_metrics,
        )
        runtime_truth = self._build_runtime_truth(
            mode=resolved_mode,
            retrieval_used=retrieval_used,
            retrieval_payload_source=retrieval_mode,
            fusion_strategy=str(fusion_outputs["fusion_strategy"]),
        )

        final_target_drugs_payload = batch.get("final_target_drugs")
        target_drugs = batch.get("target_drugs")
        resolved_target_drugs: torch.Tensor | None = None
        final_target_drugs: torch.Tensor | None = None
        if final_target_drugs_payload is not None:
            final_target_drugs = torch.as_tensor(
                final_target_drugs_payload,
                device=current_state.device,
                dtype=current_state.dtype,
            )
        if target_drugs is not None:
            resolved_target_drugs = torch.as_tensor(
                target_drugs,
                device=current_state.device,
                dtype=current_state.dtype,
            )
            if final_target_drugs is None:
                final_target_drugs = _extract_last_valid_targets(
                    resolved_target_drugs,
                    encoder_outputs["visit_mask"],
                )

        return {
            **encoder_outputs,
            **selection_outputs,
            **fusion_outputs,
            "evidence_metadata": evidence_metadata,
            "group_metadata": group_outputs.get("group_metadata", []),
            "retrieval_payload": resolved_retrieval_payload,
            "retrieval_used": retrieval_used,
            "retrieval_active": retrieval_used,
            "retrieval_available": retrieval_available,
            "retrieval_status": "active" if retrieval_used else retrieval_mode,
            "retrieval_leakage_safe": self.retrieval_leakage_safe if self.core_retrieval_enabled else False,
            "retrieval_mode": retrieval_mode,
            "retrieval_scoring_mode": self.retrieval_scoring_mode,
            "runtime_timing": runtime_timing,
            "drug_logits": drug_logits,
            "drug_probs": drug_probs,
            "recommendation_metadata": recommendation_metadata,
            "ddi_penalty_per_sample": ddi_outputs["ddi_penalty_per_sample"],
            "ddi_penalty_mean": ddi_outputs["ddi_penalty_mean"],
            "safety_metadata": ddi_outputs["safety_metadata"],
            "runtime_truth": runtime_truth,
            "target_drugs": resolved_target_drugs,
            "final_target_drugs": final_target_drugs,
        }
