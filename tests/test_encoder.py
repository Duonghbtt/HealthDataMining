from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.models.patient_state_encoder import PatientStateEncoder


def test_patient_state_encoder_forward_shapes() -> None:
    batch = {
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
    )

    outputs = model(batch)
    assert outputs["visit_repr"].shape == (2, 3, 16)
    assert outputs["state_sequence"].shape == (2, 3, 12)
    assert outputs["pooled_state"].shape == (2, 12)
    assert torch.equal(outputs["visit_mask"], batch["visit_mask"])
    assert torch.isfinite(outputs["visit_repr"]).all()
    assert torch.isfinite(outputs["state_sequence"]).all()
    assert torch.isfinite(outputs["pooled_state"]).all()
