from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.retrieval.topk_retriever import validate_retrieval_payload
from src.utils.io import ensure_dir, read_json, resolve_path, write_json


def build_edge_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_retrieval_payload(payload)
    edges: list[dict[str, Any]] = []
    neighbor_scores = payload["neighbor_scores"].tolist()
    neighbor_time_gaps_days = payload["neighbor_time_gaps_days"].tolist()
    neighbor_stay_ids = payload["neighbor_stay_ids"].tolist()
    matched_visit_indices = payload.get("matched_visit_indices")
    matched_visit_indices_list = None if matched_visit_indices is None else matched_visit_indices.tolist()
    for row_index, query_stay_id in enumerate(payload["query_stay_ids"]):
        for rank, (dst_stay_id, score, time_gap_days) in enumerate(
            zip(
                neighbor_stay_ids[row_index],
                neighbor_scores[row_index],
                neighbor_time_gaps_days[row_index],
            ),
            start=1,
        ):
            if int(dst_stay_id) < 0:
                continue
            edges.append(
                {
                    "src_stay_id": int(query_stay_id),
                    "dst_stay_id": int(dst_stay_id),
                    "score": float(score),
                    "time_gap_days": float(time_gap_days),
                    "rank": rank,
                    "split": payload["bank_split"],
                }
            )
            if matched_visit_indices_list is not None:
                edges[-1]["matched_visit_index"] = int(matched_visit_indices_list[row_index][rank - 1])
    return {
        "split": payload["bank_split"],
        "backend": payload["backend"],
        "edges": edges,
    }


def artifact_path(project_root: str | Path, split: str) -> Path:
    return ensure_dir(resolve_path(project_root, "data/artifacts/dynamic_graph")) / f"{split}_edges.json"


def save_edge_artifact(project_root: str | Path, split: str, payload: Mapping[str, Any]) -> Path:
    return write_json(artifact_path(project_root, split), build_edge_artifact(payload))


def load_edge_artifact(project_root: str | Path, split: str) -> dict[str, Any]:
    return read_json(artifact_path(project_root, split))
