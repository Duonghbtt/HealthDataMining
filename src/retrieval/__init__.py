from src.retrieval.faiss_index import VisitFaissIndex
from src.retrieval.memory_bank import VisitMemoryBank, VisitMemoryRecord


def __getattr__(name: str):
    if name == "TopKVisitRetriever":
        from src.retrieval.topk_retriever import TopKVisitRetriever

        return TopKVisitRetriever
    raise AttributeError(name)

__all__ = [
    "TopKVisitRetriever",
    "VisitFaissIndex",
    "VisitMemoryBank",
    "VisitMemoryRecord",
]
