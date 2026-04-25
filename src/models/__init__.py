from src.models.fusion import BRANCH_ORDER, FusionModule
from src.models.history_selector import HistorySelector
from src.models.patient_state_encoder import PatientStateEncoder
from src.models.temporal_similarity import cosine_similarity_matrix, temporal_decay_weights, temporal_similarity

__all__ = [
    "BRANCH_ORDER",
    "FusionModule",
    "HistorySelector",
    "PatientStateEncoder",
    "RetrievalEvidenceFusionModel",
    "cosine_similarity_matrix",
    "temporal_decay_weights",
    "temporal_similarity",
]


def __getattr__(name: str):
    if name == "RetrievalEvidenceFusionModel":
        from src.models.full_model import RetrievalEvidenceFusionModel

        return RetrievalEvidenceFusionModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
