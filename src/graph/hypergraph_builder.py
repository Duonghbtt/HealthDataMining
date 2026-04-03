from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.retrieval.memory_bank import MemoryBank
from src.utils.io import ensure_dir, read_json, resolve_path, write_json


def _validate_hypergraph_payload(payload: Mapping[str, Any]) -> None:
    required_fields = ("neighbor_indices", "neighbor_scores", "neighbor_stay_ids")
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"Retrieval payload is missing required fields for hypergraph building: {missing}")
    row_count = None
    for field in required_fields:
        value = payload[field]
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"Retrieval payload field `{field}` must be a 2D torch.Tensor")
        if row_count is None:
            row_count = int(value.shape[0])
        elif int(value.shape[0]) != row_count:
            raise ValueError("neighbor_indices, neighbor_scores, and neighbor_stay_ids must share the same batch size")


def _build_edge_list(num_nodes: int) -> list[tuple[int, ...]]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if num_nodes == 1:
        return [(0,)]
    edges = [tuple(range(num_nodes))]
    for node_index in range(1, num_nodes):
        edges.append((0, node_index))
    return edges


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(int(item) for item in left)
    right_set = set(int(item) for item in right)
    if not left_set and not right_set:
        return 0.0
    union = left_set | right_set
    return 0.0 if not union else len(left_set & right_set) / float(len(union))


def _time_bucket(value: float) -> str:
    if value <= 2.0:
        return "short"
    if value <= 7.0:
        return "mid"
    return "long"


def _edge_strength_from_members(members: Sequence[int], retrieval_scores: torch.Tensor) -> float:
    neighbor_scores = [
        float(1.0 + torch.sigmoid(retrieval_scores[int(member) - 1]).item())
        for member in members
        if int(member) > 0 and int(member) - 1 < retrieval_scores.shape[0]
    ]
    if not neighbor_scores:
        return 1.0
    return sum(neighbor_scores) / float(len(neighbor_scores))


def _append_edge(
    edges: list[tuple[int, ...]],
    edge_types: list[str],
    edge_labels: list[str],
    edge_strengths: list[float],
    seen: set[tuple[tuple[int, ...], str, str]],
    nodes: tuple[int, ...],
    *,
    edge_type: str,
    edge_label: str,
    strength: float,
) -> None:
    normalized = tuple(sorted(dict.fromkeys(int(node) for node in nodes)))
    key = (normalized, edge_type, edge_label)
    if len(normalized) <= 1 or key in seen:
        return
    seen.add(key)
    edges.append(normalized)
    edge_types.append(edge_type)
    edge_labels.append(edge_label)
    edge_strengths.append(float(strength))


def _token_semantic_edges(
    rows: Sequence[Sequence[int]],
    *,
    query_tokens: Sequence[int] | None,
    retrieval_scores: torch.Tensor,
    edge_type: str,
    max_edges: int = 4,
) -> list[tuple[tuple[int, ...], str, str, float]]:
    token_to_members: dict[int, set[int]] = {}
    query_token_set = {int(token) for token in (query_tokens or ())}
    for token in query_token_set:
        token_to_members.setdefault(int(token), set()).add(0)
    for local_index, row in enumerate(rows):
        for token in row:
            token_to_members.setdefault(int(token), set()).add(local_index + 1)

    candidates: list[tuple[tuple[int, ...], str, str, float]] = []
    for token, members in token_to_members.items():
        if len(members) <= 1:
            continue
        member_tuple = tuple(sorted(members))
        strength = _edge_strength_from_members(member_tuple, retrieval_scores)
        if 0 in member_tuple:
            strength += 0.1
        candidates.append((member_tuple, f"{edge_type}_pattern", f"{edge_type}:{token}", strength))

    candidates.sort(key=lambda item: (-len(item[0]), -item[3], item[2]))
    return candidates[:max_edges]


def _time_bucket_edges(
    time_gaps: torch.Tensor | None,
    *,
    retrieval_scores: torch.Tensor,
) -> list[tuple[tuple[int, ...], str, str, float]]:
    if time_gaps is None or time_gaps.numel() <= 0:
        return []
    buckets: dict[str, list[int]] = {}
    for local_index, gap in enumerate(time_gaps.tolist()):
        buckets.setdefault(_time_bucket(float(gap)), []).append(local_index + 1)

    edges = []
    for bucket_name, members in sorted(buckets.items()):
        if not members:
            continue
        member_tuple = tuple([0, *members])
        max_gap = max(float(time_gaps[index - 1].item()) for index in members)
        strength = _edge_strength_from_members(member_tuple, retrieval_scores) / (1.0 + max_gap)
        edges.append((member_tuple, "time_bucket", f"time:{bucket_name}", strength))
    return edges


def _semantic_edges(
    valid_indices: torch.Tensor,
    memory_bank: MemoryBank,
    *,
    retrieval_scores: torch.Tensor,
    time_gaps: torch.Tensor | None = None,
    query_metadata: Mapping[str, Any] | None = None,
    prototype_top_k: int = 2,
    include_time_edges: bool = True,
    include_prototype_edges: bool = True,
) -> tuple[list[tuple[int, ...]], list[str], list[str], list[float]]:
    edges: list[tuple[int, ...]] = []
    edge_types: list[str] = []
    edge_labels: list[str] = []
    edge_strengths: list[float] = []
    seen: set[tuple[tuple[int, ...], str, str]] = set()
    if valid_indices.numel() <= 0:
        return edges, edge_types, edge_labels, edge_strengths

    neighbor_meta = memory_bank.slice_metadata(valid_indices)
    semantic_fields = (
        ("diagnosis", neighbor_meta["diag_code_sets"], (query_metadata or {}).get("diag_code_sets")),
        ("procedure", neighbor_meta["proc_code_sets"], (query_metadata or {}).get("proc_code_sets")),
        ("medication", neighbor_meta["target_drugs"], (query_metadata or {}).get("target_drugs")),
        ("lab", neighbor_meta["lab_feature_sets"], (query_metadata or {}).get("lab_feature_sets")),
        ("vital", neighbor_meta["vital_feature_sets"], (query_metadata or {}).get("vital_feature_sets")),
    )
    for edge_type, rows, query_tokens in semantic_fields:
        for nodes, semantic_type, label, strength in _token_semantic_edges(
            rows,
            query_tokens=query_tokens,
            retrieval_scores=retrieval_scores,
            edge_type=edge_type,
        ):
            _append_edge(
                edges,
                edge_types,
                edge_labels,
                edge_strengths,
                seen,
                nodes,
                edge_type=semantic_type,
                edge_label=label,
                strength=strength,
            )

    if include_time_edges:
        for nodes, edge_type, label, strength in _time_bucket_edges(time_gaps, retrieval_scores=retrieval_scores):
            _append_edge(
                edges,
                edge_types,
                edge_labels,
                edge_strengths,
                seen,
                nodes,
                edge_type=edge_type,
                edge_label=label,
                strength=strength,
            )

    if include_prototype_edges and retrieval_scores.numel() > 0:
        keep = min(max(int(prototype_top_k), 1), int(retrieval_scores.shape[0]))
        topk = torch.topk(retrieval_scores, k=keep, dim=0)
        prototype_nodes = tuple([0, *[int(index) + 1 for index in topk.indices.tolist()]])
        _append_edge(
            edges,
            edge_types,
            edge_labels,
            edge_strengths,
            seen,
            prototype_nodes,
            edge_type="prototype",
            edge_label=f"prototype:top{keep}",
            strength=float((1.0 + torch.sigmoid(topk.values)).mean().item()),
        )

    return edges, edge_types, edge_labels, edge_strengths


def _row_query_metadata(retrieval_payload: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in (
        "query_diag_code_sets",
        "query_proc_code_sets",
        "query_lab_feature_sets",
        "query_vital_feature_sets",
        "query_target_drugs",
    ):
        if key in retrieval_payload:
            payload[key.replace("query_", "")] = retrieval_payload[key][row_index]
    return payload


def build_patient_hypergraph(
    current_state: torch.Tensor,
    *,
    neighbor_indices: torch.Tensor,
    neighbor_scores: torch.Tensor,
    neighbor_stay_ids: torch.Tensor,
    neighbor_time_gaps_days: torch.Tensor | None = None,
    memory_bank: MemoryBank,
    query_metadata: Mapping[str, Any] | None = None,
    use_semantic_edges: bool = True,
    use_weighted_edges: bool = True,
    prototype_top_k: int = 2,
    include_time_edges: bool = True,
    include_prototype_edges: bool = True,
) -> dict[str, Any]:
    current_state = torch.as_tensor(current_state, dtype=torch.float32).flatten()
    neighbor_indices = torch.as_tensor(neighbor_indices, dtype=torch.long).flatten()
    neighbor_scores = torch.as_tensor(neighbor_scores, dtype=torch.float32).flatten()
    neighbor_stay_ids = torch.as_tensor(neighbor_stay_ids, dtype=torch.long).flatten()
    valid_mask = neighbor_indices >= 0
    valid_indices = neighbor_indices[valid_mask]
    valid_scores = neighbor_scores[valid_mask]
    valid_stay_ids = neighbor_stay_ids[valid_mask]
    valid_time_gaps = (
        torch.as_tensor(neighbor_time_gaps_days, dtype=torch.float32).flatten()[valid_mask]
        if neighbor_time_gaps_days is not None
        else None
    )

    neighbor_states = (
        memory_bank.visit_states.index_select(0, valid_indices.cpu()).to(dtype=torch.float32)
        if valid_indices.numel()
        else torch.zeros(0, current_state.shape[0], dtype=torch.float32)
    )
    node_features = torch.cat([current_state.unsqueeze(0), neighbor_states], dim=0)
    edges = _build_edge_list(int(node_features.shape[0]))
    edge_types = ["global", *["current_neighbor"] * max(int(node_features.shape[0]) - 1, 0)]
    edge_labels = ["global:all", *[f"pair:{node_index}" for node_index in range(1, int(node_features.shape[0]))]]
    edge_strengths = [1.0, *[float(1.0 + torch.sigmoid(score).item()) for score in valid_scores]]
    if use_semantic_edges:
        semantic_edges, semantic_edge_types, semantic_edge_labels, semantic_edge_strengths = _semantic_edges(
            valid_indices,
            memory_bank,
            retrieval_scores=valid_scores,
            time_gaps=valid_time_gaps,
            query_metadata=query_metadata,
            prototype_top_k=prototype_top_k,
            include_time_edges=include_time_edges,
            include_prototype_edges=include_prototype_edges,
        )
        edges.extend(semantic_edges)
        edge_types.extend(semantic_edge_types)
        edge_labels.extend(semantic_edge_labels)
        edge_strengths.extend(semantic_edge_strengths)

    incidence = torch.zeros(node_features.shape[0], len(edges), dtype=torch.float32)
    for edge_index, edge_nodes in enumerate(edges):
        incidence[list(edge_nodes), edge_index] = 1.0
    edge_weights = torch.tensor(edge_strengths if use_weighted_edges else [1.0] * len(edges), dtype=torch.float32)
    edge_type_counts: dict[str, int] = {}
    for edge_type in edge_types:
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

    return {
        "node_features": node_features,
        "incidence": incidence,
        "edge_weights": edge_weights,
        "metadata": {
            "neighbor_indices": valid_indices,
            "neighbor_stay_ids": valid_stay_ids,
            "node_types": ["current", *["neighbor"] * int(valid_indices.shape[0])],
            "edge_types": edge_types,
            "edge_labels": edge_labels,
            "edge_strengths": edge_strengths,
            "edge_type_counts": edge_type_counts,
            "semantic_edge_count": int(sum(1 for edge_type in edge_types if edge_type not in {"global", "current_neighbor"})),
            "num_nodes": int(node_features.shape[0]),
            "num_neighbors": int(valid_indices.shape[0]),
            "use_semantic_edges": bool(use_semantic_edges),
            "use_weighted_edges": bool(use_weighted_edges),
            "include_time_edges": bool(include_time_edges),
            "include_prototype_edges": bool(include_prototype_edges),
        },
    }


def build_batch_hypergraphs(
    current_states: torch.Tensor,
    retrieval_payload: Mapping[str, Any],
    memory_bank: MemoryBank,
    *,
    use_semantic_edges: bool = True,
    use_weighted_edges: bool = True,
    prototype_top_k: int = 2,
    include_time_edges: bool = True,
    include_prototype_edges: bool = True,
) -> list[dict[str, Any]]:
    _validate_hypergraph_payload(retrieval_payload)
    current_states = torch.as_tensor(current_states, dtype=torch.float32).cpu()
    neighbor_indices = torch.as_tensor(retrieval_payload["neighbor_indices"], dtype=torch.long)
    neighbor_scores = torch.as_tensor(retrieval_payload["neighbor_scores"], dtype=torch.float32)
    neighbor_stay_ids = torch.as_tensor(retrieval_payload["neighbor_stay_ids"], dtype=torch.long)
    neighbor_time_gaps_days = torch.as_tensor(
        retrieval_payload.get("neighbor_time_gaps_days", torch.full_like(neighbor_scores, float("inf"))),
        dtype=torch.float32,
    )
    graphs = []
    for row_index in range(current_states.shape[0]):
        graphs.append(
            build_patient_hypergraph(
                current_states[row_index],
                neighbor_indices=neighbor_indices[row_index],
                neighbor_scores=neighbor_scores[row_index],
                neighbor_stay_ids=neighbor_stay_ids[row_index],
                neighbor_time_gaps_days=neighbor_time_gaps_days[row_index],
                memory_bank=memory_bank,
                query_metadata=_row_query_metadata(retrieval_payload, row_index),
                use_semantic_edges=use_semantic_edges,
                use_weighted_edges=use_weighted_edges,
                prototype_top_k=prototype_top_k,
                include_time_edges=include_time_edges,
                include_prototype_edges=include_prototype_edges,
            )
        )
    return graphs


def build_hypergraph_artifact(
    current_states: torch.Tensor,
    retrieval_payload: Mapping[str, Any],
    memory_bank: MemoryBank,
    *,
    use_semantic_edges: bool = True,
    use_weighted_edges: bool = True,
    prototype_top_k: int = 2,
    include_time_edges: bool = True,
    include_prototype_edges: bool = True,
) -> dict[str, Any]:
    payload = {"graphs": []}
    for query_index, graph in enumerate(
        build_batch_hypergraphs(
            current_states,
            retrieval_payload,
            memory_bank,
            use_semantic_edges=use_semantic_edges,
            use_weighted_edges=use_weighted_edges,
            prototype_top_k=prototype_top_k,
            include_time_edges=include_time_edges,
            include_prototype_edges=include_prototype_edges,
        )
    ):
        payload["graphs"].append(
            {
                "query_index": query_index,
                "num_nodes": graph["metadata"]["num_nodes"],
                "num_neighbors": graph["metadata"]["num_neighbors"],
                "neighbor_indices": graph["metadata"]["neighbor_indices"].tolist(),
                "neighbor_stay_ids": graph["metadata"]["neighbor_stay_ids"].tolist(),
                "incidence": graph["incidence"].tolist(),
                "edge_weights": graph["edge_weights"].tolist(),
                "node_types": graph["metadata"]["node_types"],
                "edge_types": graph["metadata"]["edge_types"],
                "edge_labels": graph["metadata"]["edge_labels"],
                "edge_strengths": graph["metadata"]["edge_strengths"],
                "edge_type_counts": graph["metadata"]["edge_type_counts"],
                "semantic_edge_count": graph["metadata"]["semantic_edge_count"],
            }
        )
    return payload


def artifact_path(project_root: str | Path, split: str) -> Path:
    return ensure_dir(resolve_path(project_root, "data/artifacts/hypergraph")) / f"{split}.json"


def save_hypergraph_artifact(
    project_root: str | Path,
    split: str,
    current_states: torch.Tensor,
    retrieval_payload: Mapping[str, Any],
    memory_bank: MemoryBank,
    *,
    use_semantic_edges: bool = True,
    use_weighted_edges: bool = True,
    prototype_top_k: int = 2,
    include_time_edges: bool = True,
    include_prototype_edges: bool = True,
) -> Path:
    return write_json(
        artifact_path(project_root, split),
        build_hypergraph_artifact(
            current_states,
            retrieval_payload,
            memory_bank,
            use_semantic_edges=use_semantic_edges,
            use_weighted_edges=use_weighted_edges,
            prototype_top_k=prototype_top_k,
            include_time_edges=include_time_edges,
            include_prototype_edges=include_prototype_edges,
        ),
    )


def load_hypergraph_artifact(project_root: str | Path, split: str) -> dict[str, Any]:
    return read_json(artifact_path(project_root, split))
