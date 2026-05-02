from __future__ import annotations

from src.features.diagnosis_encoder import VisitCodeEncoder


class ProcedureEncoder(VisitCodeEncoder):
    """Procedure visit encoder returning visit-level procedure embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        output_dim: int | None = None,
        padding_idx: int = 0,
        dropout: float = 0.0,
        layer_norm: bool = False,
        max_norm: float | None = None,
        scale_grad_by_freq: bool = False,
    ) -> None:
        super().__init__(
            vocab_size,
            embedding_dim,
            output_dim=output_dim,
            padding_idx=padding_idx,
            dropout=dropout,
            layer_norm=layer_norm,
            max_norm=max_norm,
            scale_grad_by_freq=scale_grad_by_freq,
        )
