from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.graph.group_encoder import GroupEncoder
from src.models.fusion import FusionModule
from src.models.history_selector import HistorySelector
from src.models.patient_state_encoder import PatientStateEncoder
from src.retrieval.memory_bank import MemoryBank


class RetrievalEvidenceFusionModel(nn.Module):
    def __init__(
        self,
        encoder: PatientStateEncoder,
        history_selector: HistorySelector,
        fusion_module: FusionModule,
        *,
        group_encoder: GroupEncoder | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.history_selector = history_selector
        self.fusion_module = fusion_module
        self.group_encoder = group_encoder

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        retrieval_payload: Mapping[str, Any] | None = None,
        memory_bank: MemoryBank | None = None,
    ) -> dict[str, Any]:
        encoder_outputs = self.encoder(dict(batch))
        group_outputs: dict[str, Any] = {
            "group_context": None,
            "group_available_mask": torch.zeros(
                encoder_outputs["pooled_state"].shape[0],
                dtype=torch.bool,
                device=encoder_outputs["pooled_state"].device,
            ),
            "group_metadata": [],
        }
        if self.group_encoder is not None and retrieval_payload is not None and memory_bank is not None:
            group_outputs = self.group_encoder(
                current_state=encoder_outputs["pooled_state"],
                retrieval_payload=retrieval_payload,
                memory_bank=memory_bank,
            )

        selection_outputs = self.history_selector(
            current_state=encoder_outputs["pooled_state"],
            state_sequence=encoder_outputs["state_sequence"],
            visit_mask=encoder_outputs["visit_mask"],
            retrieval_payload=retrieval_payload,
            memory_bank=memory_bank,
            group_context=group_outputs.get("group_context"),
            group_available_mask=group_outputs.get("group_available_mask"),
        )
        evidence_metadata = dict(selection_outputs["evidence_metadata"])
        evidence_metadata["group_metadata"] = group_outputs.get("group_metadata", [])

        fusion_outputs = self.fusion_module(
            current_state=encoder_outputs["pooled_state"],
            self_history_context=selection_outputs["self_history_context"],
            neighbor_history_context=selection_outputs["neighbor_history_context"],
            group_context=selection_outputs.get("group_context"),
            branch_masks={
                "current": torch.ones(
                    encoder_outputs["pooled_state"].shape[0],
                    dtype=torch.bool,
                    device=encoder_outputs["pooled_state"].device,
                ),
                "self": evidence_metadata["self_history_available_mask"],
                "neighbor": evidence_metadata["neighbor_available_mask"],
                "group": evidence_metadata["group_available_mask"],
            },
        )

        return {
            **encoder_outputs,
            **selection_outputs,
            **fusion_outputs,
            "evidence_metadata": evidence_metadata,
            "group_metadata": group_outputs.get("group_metadata", []),
        }
