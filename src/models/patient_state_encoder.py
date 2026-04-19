from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.features.diagnosis_encoder import DiagnosisEncoder
from src.features.procedure_encoder import ProcedureEncoder


def _masked_average(
    embedding: nn.Embedding,
    indices: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    vectors = embedding(indices)
    weights = mask.to(dtype=vectors.dtype).unsqueeze(-1)
    summed = (vectors * weights).sum(dim=-2)
    denom = weights.sum(dim=-2).clamp(min=1.0)
    return summed / denom


class PatientStateEncoder(nn.Module):
    def __init__(
        self,
        diagnosis_vocab_size: int,
        procedure_vocab_size: int,
        drug_vocab_size: int,
        num_lab_features: int,
        num_vital_features: int,
        *,
        code_embedding_dim: int = 64,
        medication_embedding_dim: int = 64,
        numeric_projection_dim: int = 32,
        time_embedding_dim: int = 32,
        visit_hidden_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        resolved_num_layers = max(int(num_layers), 1)
        self.diagnosis_encoder = DiagnosisEncoder(diagnosis_vocab_size, code_embedding_dim, padding_idx=0)
        self.procedure_encoder = ProcedureEncoder(procedure_vocab_size, code_embedding_dim, padding_idx=0)
        self.medication_embedding = nn.Embedding(drug_vocab_size, medication_embedding_dim, padding_idx=0)
        self.lab_projection = nn.Linear(num_lab_features, numeric_projection_dim)
        self.vital_projection = nn.Linear(num_vital_features, numeric_projection_dim)
        self.time_projection = nn.Linear(1, time_embedding_dim)

        fused_dim = (
            code_embedding_dim
            + code_embedding_dim
            + medication_embedding_dim
            + numeric_projection_dim
            + numeric_projection_dim
            + time_embedding_dim
        )
        self.visit_projection = nn.Sequential(
            nn.Linear(fused_dim, visit_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            visit_hidden_dim,
            hidden_dim,
            num_layers=resolved_num_layers,
            dropout=float(dropout) if resolved_num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        diag_repr = self.diagnosis_encoder(batch["diag_codes"], batch["diag_mask"])
        proc_repr = self.procedure_encoder(batch["proc_codes"], batch["proc_mask"])
        med_repr = _masked_average(
            self.medication_embedding,
            batch["med_history"],
            batch["med_history_mask"],
        )

        lab_repr = self.lab_projection(batch["lab_values"])
        vital_repr = self.vital_projection(batch["vital_values"])
        time_repr = self.time_projection(torch.log1p(batch["time_delta_hours"]).unsqueeze(-1))

        visit_repr = self.visit_projection(
            torch.cat([diag_repr, proc_repr, med_repr, lab_repr, vital_repr, time_repr], dim=-1)
        )
        visit_mask = batch["visit_mask"]
        visit_lengths = batch.get("visit_lengths")
        if visit_lengths is None:
            lengths = visit_mask.sum(dim=-1).clamp(min=1).cpu()
        else:
            lengths = visit_lengths.to(dtype=torch.long, device="cpu").clamp(min=1)
        packed = pack_padded_sequence(visit_repr, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed)
        state_sequence, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=visit_repr.size(1),
        )
        pooled_state = hidden[-1]
        return {
            "visit_repr": visit_repr,
            "state_sequence": state_sequence,
            "pooled_state": pooled_state,
            "visit_mask": visit_mask,
        }
