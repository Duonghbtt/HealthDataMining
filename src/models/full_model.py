from __future__ import annotations

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
    """End-to-end recommendation model with soft fallbacks for core and extended modes."""

    def __init__(
        self,
        encoder: PatientStateEncoder,
        history_selector: HistorySelector,
        fusion_module: FusionModule,
        *,
        group_encoder: GroupEncoder | None = None,
        medication_decoder: MedicationDecoder | None = None,
        ddi_regularizer: DDIRegularizer | None = None,
        mode: str = "core",
        retrieval_top_k: int = 5,
        temporal_decay_alpha: float = 0.05,
        retrieval_backend: str = "bruteforce",
        use_faiss_if_available: bool = True,
        allow_cross_split: bool = False,
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
        self.mode = _resolve_mode(mode, None)
        self.retrieval_top_k = int(retrieval_top_k)
        self.temporal_decay_alpha = float(temporal_decay_alpha)
        self.retrieval_backend = retrieval_backend
        self.use_faiss_if_available = bool(use_faiss_if_available)
        self.allow_cross_split = bool(allow_cross_split)

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
    ) -> tuple[Mapping[str, Any] | None, str]:
        if retrieval_payload is not None:
            return retrieval_payload, "provided"
        if mode != "extended" or memory_bank is None:
            return None, "disabled"

        resolved_query_states: torch.Tensor | None = None
        resolved_query_metadata: Mapping[str, Any] | None = None

        if records is not None:
            resolved_query_states, resolved_query_metadata = build_last_visit_queries(
                records,
                encoder_outputs,
                split=memory_bank.split,
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
            return None, "disabled"

        payload = retrieve_topk(
            resolved_query_states,
            resolved_query_metadata,
            memory_bank,
            top_k=self.retrieval_top_k,
            temporal_decay_alpha=self.temporal_decay_alpha,
            backend=self.retrieval_backend,
            use_faiss_if_available=self.use_faiss_if_available,
            allow_cross_split=self.allow_cross_split,
        )
        return payload, "built"

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
    ) -> dict[str, Any]:
        if self.ddi_regularizer is None or drug_probs is None:
            return {
                "ddi_penalty_per_sample": None,
                "ddi_penalty_mean": None,
                "safety_metadata": {
                    "ddi_available": False,
                    "ddi_regularizer_present": self.ddi_regularizer is not None,
                    "mode": mode,
                    "retrieval_used": retrieval_used,
                },
            }

        ddi_penalty_per_sample = self.ddi_regularizer.compute_penalty_per_sample(drug_probs)
        ddi_penalty_per_sample = ddi_penalty_per_sample.to(device=device, dtype=dtype)
        ddi_penalty_mean = ddi_penalty_per_sample.mean()
        return {
            "ddi_penalty_per_sample": ddi_penalty_per_sample,
            "ddi_penalty_mean": ddi_penalty_mean,
            "safety_metadata": {
                "ddi_available": True,
                "ddi_regularizer_present": True,
                "mode": mode,
                "retrieval_used": retrieval_used,
            },
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
    ) -> dict[str, Any]:
        encoder_outputs = self.encoder(dict(batch))
        current_state = encoder_outputs["pooled_state"]
        resolved_mode = _resolve_mode(self.mode, mode)

        resolved_retrieval_payload, retrieval_mode = self._build_retrieval_payload_if_possible(
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
        )

        target_drugs = batch.get("target_drugs")
        resolved_target_drugs: torch.Tensor | None = None
        final_target_drugs: torch.Tensor | None = None
        if target_drugs is not None:
            resolved_target_drugs = torch.as_tensor(
                target_drugs,
                device=current_state.device,
                dtype=current_state.dtype,
            )
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
            "retrieval_available": retrieval_available,
            "retrieval_mode": retrieval_mode,
            "drug_logits": drug_logits,
            "drug_probs": drug_probs,
            "recommendation_metadata": recommendation_metadata,
            "ddi_penalty_per_sample": ddi_outputs["ddi_penalty_per_sample"],
            "ddi_penalty_mean": ddi_outputs["ddi_penalty_mean"],
            "safety_metadata": ddi_outputs["safety_metadata"],
            "target_drugs": resolved_target_drugs,
            "final_target_drugs": final_target_drugs,
        }
