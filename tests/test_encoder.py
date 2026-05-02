from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.features.diagnosis_encoder import DiagnosisEncoder
from src.features.lab_processor import LabFeatureEncoder
from src.features.medication_history import MedicationHistoryEncoder
from src.features.procedure_encoder import ProcedureEncoder
from src.features.vital_processor import VitalFeatureEncoder
from src.models.patient_state_encoder import PatientStateEncoder


@pytest.fixture
def batch() -> dict[str, torch.Tensor]:
    return {
        "diag_codes": torch.tensor(
            [
                [[2, 3, 0], [2, 0, 0], [0, 0, 0]],
                [[4, 5, 6], [7, 0, 0], [8, 9, 0]],
            ],
            dtype=torch.long,
        ),
        "diag_mask": torch.tensor(
            [
                [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [1, 0, 0], [1, 1, 0]],
            ],
            dtype=torch.bool,
        ),
        "proc_codes": torch.tensor(
            [
                [[2, 0], [3, 4], [0, 0]],
                [[5, 0], [6, 0], [7, 8]],
            ],
            dtype=torch.long,
        ),
        "proc_mask": torch.tensor(
            [
                [[1, 0], [1, 1], [0, 0]],
                [[1, 0], [1, 0], [1, 1]],
            ],
            dtype=torch.bool,
        ),
        "lab_values": torch.tensor(
            [
                [[0.1, -0.3], [0.2, 0.0], [0.0, 0.0]],
                [[-0.4, 0.8], [0.0, 0.0], [0.5, -0.2]],
            ],
            dtype=torch.float32,
        ),
        "lab_mask": torch.tensor(
            [
                [[1, 1], [1, 0], [0, 0]],
                [[1, 1], [0, 0], [1, 1]],
            ],
            dtype=torch.bool,
        ),
        "vital_values": torch.tensor(
            [
                [[0.7, 0.2], [0.1, -0.1], [0.0, 0.0]],
                [[-0.2, 0.4], [0.0, 0.0], [0.3, 0.6]],
            ],
            dtype=torch.float32,
        ),
        "vital_mask": torch.tensor(
            [
                [[1, 1], [1, 1], [0, 0]],
                [[1, 1], [0, 0], [1, 1]],
            ],
            dtype=torch.bool,
        ),
        "med_history": torch.tensor(
            [
                [[2, 0, 0], [2, 3, 0], [0, 0, 0]],
                [[4, 5, 0], [4, 0, 0], [6, 7, 8]],
            ],
            dtype=torch.long,
        ),
        "med_history_mask": torch.tensor(
            [
                [[1, 0, 0], [1, 1, 0], [0, 0, 0]],
                [[1, 1, 0], [1, 0, 0], [1, 1, 1]],
            ],
            dtype=torch.bool,
        ),
        "time_delta_hours": torch.tensor(
            [[0.0, 24.0, 0.0], [0.0, 24.0, 24.0]],
            dtype=torch.float32,
        ),
        "visit_mask": torch.tensor(
            [[1, 1, 0], [1, 1, 1]],
            dtype=torch.bool,
        ),
    }


def test_modality_branch_forward_shapes(batch: dict[str, torch.Tensor]) -> None:
    diag_encoder = DiagnosisEncoder(32, 8, output_dim=10, padding_idx=0, dropout=0.0)
    proc_encoder = ProcedureEncoder(24, 8, output_dim=10, padding_idx=0, dropout=0.0)
    med_encoder = MedicationHistoryEncoder(32, 8, output_dim=10, padding_idx=0, dropout=0.0)
    lab_encoder = LabFeatureEncoder(2, 10, hidden_dim=12, dropout=0.0)
    vital_encoder = VitalFeatureEncoder(2, 10, hidden_dim=12, dropout=0.0)

    diag_repr = diag_encoder(batch["diag_codes"], batch["diag_mask"])
    proc_repr = proc_encoder(batch["proc_codes"], batch["proc_mask"])
    med_repr = med_encoder(batch["med_history"], batch["med_history_mask"])
    lab_repr = lab_encoder(batch["lab_values"], batch["lab_mask"])
    vital_repr = vital_encoder(batch["vital_values"], batch["vital_mask"])

    assert diag_repr.shape == (2, 3, 10)
    assert proc_repr.shape == (2, 3, 10)
    assert med_repr.shape == (2, 3, 10)
    assert lab_repr.shape == (2, 3, 10)
    assert vital_repr.shape == (2, 3, 10)
    assert torch.isfinite(diag_repr).all()
    assert torch.isfinite(proc_repr).all()
    assert torch.isfinite(med_repr).all()
    assert torch.isfinite(lab_repr).all()
    assert torch.isfinite(vital_repr).all()


def test_numeric_branch_is_nan_safe(batch: dict[str, torch.Tensor]) -> None:
    lab_encoder = LabFeatureEncoder(2, 10, hidden_dim=12, dropout=0.0)
    values = batch["lab_values"].clone()
    values[0, 0, 0] = float("nan")
    outputs = lab_encoder(values, batch["lab_mask"])

    assert outputs.shape == (2, 3, 10)
    assert torch.isfinite(outputs).all()


def test_patient_state_encoder_forward_shapes(batch: dict[str, torch.Tensor]) -> None:
    model = PatientStateEncoder(
        diagnosis_vocab_size=32,
        procedure_vocab_size=24,
        drug_vocab_size=32,
        num_lab_features=2,
        num_vital_features=2,
        code_embedding_dim=8,
        medication_embedding_dim=8,
        numeric_projection_dim=4,
        time_embedding_dim=4,
        visit_hidden_dim=16,
        hidden_dim=12,
        dropout=0.0,
        encoder_mode="modality_aware_gru",
        modality_hidden_dim=10,
        fusion_hidden_dim=16,
        modality_dropout=0.0,
        use_temporal_attention=True,
        temporal_attention_heads=1,
        temporal_attention_dropout=0.0,
    )

    outputs = model(batch)
    assert outputs["visit_repr"].shape == (2, 3, 16)
    assert outputs["state_sequence"].shape == (2, 3, 12)
    assert outputs["history_states"].shape == (2, 3, 12)
    assert outputs["pooled_state"].shape == (2, 12)
    assert outputs["current_state"].shape == (2, 12)
    assert outputs["modality_gate_weights"].shape == (2, 3, 4)
    assert outputs["numeric_gate_weights"].shape == (2, 3, 2)
    assert outputs["temporal_attention_weights"] is not None
    assert torch.equal(outputs["visit_mask"], batch["visit_mask"])
    assert torch.isfinite(outputs["visit_repr"]).all()
    assert torch.isfinite(outputs["state_sequence"]).all()
    assert torch.isfinite(outputs["pooled_state"]).all()
    assert set(outputs["modality_summary"].keys()) >= {
        "diagnosis",
        "procedure",
        "lab_vital",
        "med_history",
    }
    assert set(outputs["modality_history_states"].keys()) == {
        "diagnosis",
        "procedure",
        "lab_vital",
        "medication_history",
    }
    assert outputs["modality_history_states"]["diagnosis"].shape[:2] == (2, 3)
    assert outputs["modality_history_states"]["procedure"].shape[:2] == (2, 3)
    assert outputs["modality_history_states"]["lab_vital"].shape[:2] == (2, 3)
    assert outputs["modality_history_states"]["medication_history"].shape[:2] == (2, 3)


def test_patient_state_encoder_legacy_mode_keeps_backward_compatible_keys(batch: dict[str, torch.Tensor]) -> None:
    model = PatientStateEncoder(
        diagnosis_vocab_size=32,
        procedure_vocab_size=24,
        drug_vocab_size=32,
        num_lab_features=2,
        num_vital_features=2,
        code_embedding_dim=8,
        medication_embedding_dim=8,
        numeric_projection_dim=4,
        time_embedding_dim=4,
        visit_hidden_dim=16,
        hidden_dim=12,
        dropout=0.0,
        encoder_mode="legacy_gru",
        modality_hidden_dim=10,
        fusion_hidden_dim=16,
        modality_dropout=0.0,
        use_temporal_attention=False,
    )

    outputs = model(batch)

    assert outputs["visit_repr"].shape == (2, 3, 16)
    assert outputs["state_sequence"].shape == (2, 3, 12)
    assert outputs["history_states"].shape == outputs["state_sequence"].shape
    assert outputs["current_state"].shape == (2, 12)
    assert outputs["pooled_state"].shape == (2, 12)
    assert outputs["temporal_attention_weights"] is None
