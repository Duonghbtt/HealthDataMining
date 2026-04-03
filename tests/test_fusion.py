from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from src.graph.hypergraph_builder import build_patient_hypergraph
from src.graph.group_encoder import GroupEncoder
from src.models.fusion import BRANCH_ORDER, FusionModule
from src.models.full_model import RetrievalEvidenceFusionModel
from src.models.history_selector import HistorySelector
from src.retrieval.memory_bank import MemoryBank, build_last_visit_queries
from src.retrieval.topk_retriever import retrieve_topk

from tests.helpers import load_handover_records, load_vocab_size


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ablation_module = _load_module("test_evaluate_ablation_module", "src/evaluation/evaluate_ablation.py")
_attention_export_module = _load_module("test_attention_export_module", "src/explainability/attention_export.py")
build_ablation_summary = _ablation_module.build_ablation_summary
save_ablation_report = _ablation_module.save_ablation_report
save_attention_artifacts = _attention_export_module.save_attention_artifacts


def test_fusion_module_handles_missing_group_context() -> None:
    module = FusionModule(hidden_dim=4, dropout=0.0)
    out = module(
        current_state=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        self_history_context=torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
        neighbor_history_context=torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
        branch_masks={
            "current": torch.tensor([True]),
            "self": torch.tensor([True]),
            "neighbor": torch.tensor([True]),
            "group": torch.tensor([False]),
        },
    )
    assert out["fused_repr"].shape == (1, 4)
    assert out["fusion_weights"].shape == (1, len(BRANCH_ORDER))
    assert torch.allclose(out["fusion_weights"].sum(dim=-1), torch.ones(1))
    assert out["fusion_weights"][0, BRANCH_ORDER.index("group")].item() == 0.0
    assert out["fusion_entropy_loss"].shape == (1,)
    assert out["fusion_balance_loss"].shape == (1,)
    assert torch.isfinite(out["fusion_entropy_loss"]).all()
    assert torch.isfinite(out["fusion_balance_loss"]).all()


def test_fusion_module_with_group_mask_degrades_gracefully() -> None:
    module = FusionModule(hidden_dim=4, dropout=0.0)
    out = module(
        current_state=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]], dtype=torch.float32),
        self_history_context=torch.ones(2, 4, dtype=torch.float32),
        neighbor_history_context=torch.ones(2, 4, dtype=torch.float32) * 0.5,
        group_context=torch.ones(2, 4, dtype=torch.float32) * 0.25,
        branch_masks={
            "current": torch.tensor([True, True]),
            "self": torch.tensor([True, True]),
            "neighbor": torch.tensor([True, False]),
            "group": torch.tensor([True, False]),
        },
    )
    assert torch.isfinite(out["fused_repr"]).all()
    assert out["fusion_weights"][1, BRANCH_ORDER.index("group")].item() == 0.0
    assert out["fusion_weights"][1, BRANCH_ORDER.index("neighbor")].item() == 0.0
    assert out["branch_collapse_score"].shape == (2,)
    assert out["branch_balance_score"].shape == (2,)
    assert torch.isfinite(out["branch_collapse_score"]).all()
    assert torch.isfinite(out["branch_balance_score"]).all()


def test_hypergraph_builder_emits_semantic_edges() -> None:
    bank = MemoryBank(
        visit_states=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        visit_repr=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        subject_ids=[1, 2, 3],
        hadm_ids=[11, 22, 33],
        stay_ids=[111, 222, 333],
        visit_index=[0, 0, 0],
        visit_time_days=[1.0, 2.0, 6.0],
        visit_time_text=["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-06 00:00:00"],
        target_drugs=[(5,), (5, 6), (9,)],
        num_steps=[1, 1, 1],
        diag_code_sets=[(10, 20), (10, 30), (20, 40)],
        proc_code_sets=[(1,), (1, 2), (3,)],
        lab_feature_sets=[(0,), (0, 1), (2,)],
        vital_feature_sets=[(0,), (0, 1), (3,)],
        split="train",
    )
    graph = build_patient_hypergraph(
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        neighbor_indices=torch.tensor([0, 1, 2], dtype=torch.long),
        neighbor_scores=torch.tensor([0.9, 0.8, 0.3], dtype=torch.float32),
        neighbor_stay_ids=torch.tensor([111, 222, 333], dtype=torch.long),
        neighbor_time_gaps_days=torch.tensor([1.0, 2.0, 6.0], dtype=torch.float32),
        memory_bank=bank,
        query_metadata={"diag_code_sets": (10, 20), "proc_code_sets": (1,), "lab_feature_sets": (0,), "vital_feature_sets": (0,)},
        use_semantic_edges=True,
        use_weighted_edges=True,
        prototype_top_k=2,
    )
    assert graph["metadata"]["semantic_edge_count"] > 0
    assert "diagnosis_pattern" in graph["metadata"]["edge_type_counts"]
    assert len(graph["metadata"]["edge_labels"]) == len(graph["metadata"]["edge_types"])


def test_ablation_summary_handles_multi_seed_and_extra_variants() -> None:
    summary = build_ablation_summary(
        {
            "Base": {"jaccard": [0.10, 0.11, 0.12]},
            "TempSim": {"jaccard": [0.13, 0.14, 0.13]},
            "SelfSel": {"jaccard": [0.15, 0.16, 0.15]},
            "NbrSel": {"jaccard": [0.14, 0.15, 0.14]},
            "NoAttrGate": {"jaccard": [0.16, 0.17, 0.16]},
            "NoGroupAware": {"jaccard": [0.17, 0.17, 0.18]},
            "NoFusionReg": {"jaccard": [0.18, 0.18, 0.18]},
            "Full Core": {"jaccard": [0.18, 0.19, 0.20]},
            "Extended": {"jaccard": [0.21, 0.22, 0.23]},
            "FusionGated": {"jaccard": [0.20, 0.21, 0.22]},
            "FusionConcat": {"jaccard": [0.17, 0.18, 0.19]},
        }
    )
    assert summary["rows"][-1]["variant"] == "FusionGated"
    assert summary["comparisons"]["selection_vs_tempsim"]["win_rate"] == 1.0
    assert summary["comparisons"]["fusion_gated_vs_concat"]["mean_delta"] is not None
    assert summary["questions"]["attribute_gate_helps"] is True
    assert summary["questions"]["group_aware_helps"] is True
    assert summary["questions"]["fusion_reg_helps"] is True
    assert summary["diagnostics"]["person3"]["attribute_gate_vs_off"] is not None


def test_end_to_end_real_handover_pipeline_exports_attention_and_ablation() -> None:
    pyarrow = pytest.importorskip("pyarrow")
    _ = pyarrow
    from src.data.dataset import collate_batch
    from src.models.patient_state_encoder import PatientStateEncoder

    records = load_handover_records(limit=5)
    batch = collate_batch(records)
    encoder = PatientStateEncoder(
        diagnosis_vocab_size=load_vocab_size("diagnosis"),
        procedure_vocab_size=load_vocab_size("procedure"),
        drug_vocab_size=load_vocab_size("drug"),
        num_lab_features=batch["lab_values"].shape[-1],
        num_vital_features=batch["vital_values"].shape[-1],
        code_embedding_dim=8,
        medication_embedding_dim=8,
        numeric_projection_dim=4,
        time_embedding_dim=4,
        visit_hidden_dim=16,
        hidden_dim=12,
        dropout=0.0,
    )
    encoder_outputs = encoder(batch)
    bank = MemoryBank.build_from_batch(records, encoder_outputs, split="train")
    query_states, query_metadata = build_last_visit_queries(records, encoder_outputs, split="train")
    retrieval_payload = retrieve_topk(
        query_states,
        query_metadata,
        bank,
        top_k=2,
        temporal_decay_alpha=0.05,
    )

    pipeline = RetrievalEvidenceFusionModel(
        encoder,
        HistorySelector(hidden_dim=12, dropout=0.0),
        FusionModule(hidden_dim=12, dropout=0.0),
        group_encoder=GroupEncoder(hidden_dim=12, num_layers=2, dropout=0.0, num_group_prototypes=4),
    )
    out = pipeline(batch, retrieval_payload=retrieval_payload, memory_bank=bank)
    assert out["fused_repr"].shape == (5, 12)
    assert out["fusion_weights"].shape == (5, len(BRANCH_ORDER))
    assert len(out["group_metadata"]) == 5
    assert torch.isfinite(out["fused_repr"]).all()

    artifact_root = PROJECT_ROOT / "test_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    attention_paths = save_attention_artifacts(
        artifact_root,
        name="real_batch",
        selection_outputs=out,
        fusion_outputs=out,
    )
    assert attention_paths["json"].exists()
    assert attention_paths["csv"].exists()
    attention_payload = attention_paths["json"].read_text(encoding="utf-8")
    assert "selection_summary" in attention_payload
    assert "branch_summary" in attention_payload
    assert "mean_attribute_weights" in attention_payload
    assert "group_aware_selection_rate" in attention_payload

    summary = build_ablation_summary(
        {
            "Base": {"jaccard": 0.11, "ddi_rate": 0.09},
            "TempSim": {"jaccard": 0.14, "ddi_rate": 0.08},
            "SelfSel": {"jaccard": 0.15, "ddi_rate": 0.08},
            "NbrSel": {"jaccard": 0.16, "ddi_rate": 0.08},
            "Full Core": {"jaccard": 0.19, "ddi_rate": 0.07},
            "Extended": {"jaccard": 0.21, "ddi_rate": 0.07},
        }
    )
    assert summary["questions"]["selection_beats_tempsim"] is True
    assert summary["questions"]["hypergraph_beats_full_core"] is True

    report_paths = save_ablation_report(
        artifact_root,
        {
            "Base": {"jaccard": 0.11, "ddi_rate": 0.09},
            "TempSim": {"jaccard": 0.14, "ddi_rate": 0.08},
            "SelfSel": {"jaccard": 0.15, "ddi_rate": 0.08},
            "NbrSel": {"jaccard": 0.16, "ddi_rate": 0.08},
            "Full Core": {"jaccard": 0.19, "ddi_rate": 0.07},
            "Extended": {"jaccard": 0.21, "ddi_rate": 0.07},
        },
    )
    assert report_paths["json"].exists()
    assert report_paths["csv"].exists()
    assert report_paths["markdown"].exists()
