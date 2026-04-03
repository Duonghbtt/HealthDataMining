from __future__ import annotations

from typing import Any, Mapping

import pytest


torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from src.explainability.counterfactual import run_counterfactual_analysis
from src.explainability.nl_explainer import build_nl_explanation
from src.models.full_model import RetrievalEvidenceFusionModel
from src.models.medication_decoder import MedicationDecoder


class ToyEncoder(nn.Module):
    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        diag_count = batch["diag_mask"].to(dtype=torch.float32).sum(dim=-1)
        proc_count = batch["proc_mask"].to(dtype=torch.float32).sum(dim=-1)
        lab_sum = batch["lab_values"].sum(dim=-1)
        vital_sum = batch["vital_values"].sum(dim=-1)
        med_count = batch["med_history_mask"].to(dtype=torch.float32).sum(dim=-1)
        time_delta = batch["time_delta_hours"]
        diag_code_sum = batch["diag_codes"].to(dtype=torch.float32).sum(dim=-1)
        proc_code_sum = batch["proc_codes"].to(dtype=torch.float32).sum(dim=-1)

        state_sequence = torch.stack(
            [
                diag_count,
                proc_count,
                lab_sum,
                vital_sum,
                med_count,
                time_delta,
                diag_code_sum,
                proc_code_sum,
            ],
            dim=-1,
        )
        visit_mask = batch["visit_mask"]
        last_index = visit_mask.to(dtype=torch.long).sum(dim=1) - 1
        pooled_state = state_sequence[
            torch.arange(state_sequence.shape[0], device=state_sequence.device),
            last_index,
        ]
        return {
            "visit_repr": state_sequence,
            "state_sequence": state_sequence,
            "pooled_state": pooled_state,
            "visit_mask": visit_mask,
        }


class ToyHistorySelector(nn.Module):
    def forward(
        self,
        *,
        current_state: torch.Tensor,
        state_sequence: torch.Tensor,
        visit_mask: torch.Tensor,
        retrieval_payload: Mapping[str, Any] | None = None,
        memory_bank: Any = None,
        group_context: torch.Tensor | None = None,
        group_available_mask: torch.Tensor | None = None,
        attribute_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = retrieval_payload, memory_bank, group_context, group_available_mask, attribute_payload
        batch_size, _, hidden_dim = state_sequence.shape
        device = state_sequence.device

        has_self = visit_mask.sum(dim=1) > 1
        self_context = torch.where(
            has_self.unsqueeze(-1),
            state_sequence[:, 0, :],
            torch.zeros(batch_size, hidden_dim, dtype=state_sequence.dtype, device=device),
        )
        neighbor_context = torch.zeros_like(current_state)
        empty_float = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
        empty_long = torch.full((batch_size, 1), -1, dtype=torch.long, device=device)
        empty_bool = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        self_attr_weights = torch.tensor(
            [[[0.6, 0.1, 0.1, 0.1, 0.1]]],
            dtype=torch.float32,
            device=device,
        ).expand(batch_size, -1, -1)
        self_attr_mask = torch.ones(batch_size, 1, 5, dtype=torch.bool, device=device)

        evidence_metadata = {
            "attribute_order": ["diagnosis", "procedure", "lab", "vital", "medication"],
            "self_history_available_mask": has_self,
            "neighbor_available_mask": torch.zeros(batch_size, dtype=torch.bool, device=device),
            "group_available_mask": torch.zeros(batch_size, dtype=torch.bool, device=device),
            "self_history_weights": has_self.to(dtype=torch.float32).unsqueeze(-1),
            "self_history_mask": has_self.unsqueeze(-1),
            "self_history_indices": torch.zeros(batch_size, 1, dtype=torch.long, device=device),
            "neighbor_weights": empty_float,
            "neighbor_mask": empty_bool,
            "neighbor_indices": empty_long,
            "neighbor_stay_ids": empty_long,
            "neighbor_matched_visit_indices": empty_long,
            "self_attribute_weights": self_attr_weights,
            "self_attribute_mask": self_attr_mask,
            "neighbor_attribute_weights": torch.zeros(batch_size, 1, 5, dtype=torch.float32, device=device),
            "neighbor_attribute_mask": torch.zeros(batch_size, 1, 5, dtype=torch.bool, device=device),
        }
        return {
            "self_history_context": self_context,
            "neighbor_history_context": neighbor_context,
            "group_context": None,
            "evidence_metadata": evidence_metadata,
        }


class ToyFusionModule(nn.Module):
    def forward(
        self,
        *,
        current_state: torch.Tensor,
        self_history_context: torch.Tensor | None = None,
        neighbor_history_context: torch.Tensor | None = None,
        group_context: torch.Tensor | None = None,
        branch_masks: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        device = current_state.device
        batch_size = current_state.shape[0]
        masks = branch_masks or {}
        branch_order = ["current", "self", "neighbor", "group"]
        branch_contexts = {
            "current": current_state,
            "self": torch.zeros_like(current_state) if self_history_context is None else self_history_context,
            "neighbor": torch.zeros_like(current_state) if neighbor_history_context is None else neighbor_history_context,
            "group": torch.zeros_like(current_state) if group_context is None else group_context,
        }
        logits = torch.stack(
            [
                torch.full((batch_size,), 0.4, dtype=current_state.dtype, device=device),
                torch.full((batch_size,), 0.8, dtype=current_state.dtype, device=device),
                torch.full((batch_size,), 0.1, dtype=current_state.dtype, device=device),
                torch.full((batch_size,), -0.2, dtype=current_state.dtype, device=device),
            ],
            dim=1,
        )
        branch_mask = torch.stack(
            [
                masks.get("current", torch.ones(batch_size, dtype=torch.bool, device=device)),
                masks.get("self", torch.zeros(batch_size, dtype=torch.bool, device=device)),
                masks.get("neighbor", torch.zeros(batch_size, dtype=torch.bool, device=device)),
                masks.get("group", torch.zeros(batch_size, dtype=torch.bool, device=device)),
            ],
            dim=1,
        )
        masked_logits = logits.masked_fill(~branch_mask, float("-inf"))
        fusion_weights = torch.softmax(masked_logits, dim=1)
        fusion_weights = torch.where(branch_mask, fusion_weights, torch.zeros_like(fusion_weights))
        fusion_weights = fusion_weights / fusion_weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        fused_repr = (
            branch_contexts["current"] * fusion_weights[:, 0:1]
            + branch_contexts["self"] * fusion_weights[:, 1:2]
            + branch_contexts["neighbor"] * fusion_weights[:, 2:3]
            + branch_contexts["group"] * fusion_weights[:, 3:4]
        )
        dominant_branch_index = fusion_weights.argmax(dim=1)
        dominant_branch_name = [branch_order[int(index)] for index in dominant_branch_index.tolist()]
        return {
            "fused_repr": fused_repr,
            "fusion_weights": fusion_weights,
            "branch_mask": branch_mask,
            "branch_order": branch_order,
            "dominant_branch_name": dominant_branch_name,
        }


@pytest.fixture
def toy_batch() -> dict[str, torch.Tensor]:
    return {
        "diag_codes": torch.tensor([[[1, 2], [3, 4]]], dtype=torch.long),
        "diag_mask": torch.tensor([[[1, 1], [1, 1]]], dtype=torch.bool),
        "proc_codes": torch.tensor([[[1, 0], [2, 3]]], dtype=torch.long),
        "proc_mask": torch.tensor([[[1, 0], [1, 1]]], dtype=torch.bool),
        "lab_values": torch.tensor([[[0.2, 0.0], [0.5, -0.1]]], dtype=torch.float32),
        "lab_mask": torch.tensor([[[1, 0], [1, 1]]], dtype=torch.bool),
        "vital_values": torch.tensor([[[0.1, 0.3], [0.6, 0.2]]], dtype=torch.float32),
        "vital_mask": torch.tensor([[[1, 1], [1, 1]]], dtype=torch.bool),
        "med_history": torch.tensor([[[1, 0], [1, 2]]], dtype=torch.long),
        "med_history_mask": torch.tensor([[[1, 0], [1, 1]]], dtype=torch.bool),
        "time_delta_hours": torch.tensor([[0.0, 12.0]], dtype=torch.float32),
        "visit_mask": torch.tensor([[1, 1]], dtype=torch.bool),
        "target_drugs": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]], dtype=torch.float32),
    }


@pytest.fixture
def toy_model() -> RetrievalEvidenceFusionModel:
    torch.manual_seed(7)
    return RetrievalEvidenceFusionModel(
        ToyEncoder(),
        ToyHistorySelector(),
        ToyFusionModule(),
        medication_decoder=MedicationDecoder(
            hidden_dim=8,
            drug_vocab_size=5,
            dropout=0.0,
            top_k_metadata=3,
        ),
        mode="core",
    )


def test_counterfactual_returns_nonempty_payload_schema(
    toy_model: RetrievalEvidenceFusionModel,
    toy_batch: dict[str, torch.Tensor],
) -> None:
    payload = run_counterfactual_analysis(
        toy_model,
        toy_batch,
        threshold=0.5,
        top_k=3,
        mode="core",
    )

    assert set(payload.keys()) == {"baseline", "interventions", "best_counterfactual", "metadata"}
    assert payload["baseline"]["top_recommendations"]
    assert "evidence_summary" in payload["baseline"]
    assert "dominant_branch" in payload["baseline"]["evidence_summary"]
    assert payload["interventions"]
    assert any(item.get("available") for item in payload["interventions"])


def test_nl_explainer_returns_key_text_fields(
    toy_model: RetrievalEvidenceFusionModel,
    toy_batch: dict[str, torch.Tensor],
) -> None:
    payload = run_counterfactual_analysis(
        toy_model,
        toy_batch,
        threshold=0.5,
        top_k=3,
        mode="core",
    )
    explanation = build_nl_explanation(payload)

    assert set(explanation.keys()) == {
        "recommendation_text",
        "evidence_text",
        "safety_text",
        "counterfactual_text",
        "summary_text",
    }
    assert explanation["summary_text"]
    assert any(bool(explanation[key]) for key in explanation)
