from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.graph.hypergraph_builder import build_batch_hypergraphs
from src.graph.hypergraph_layers import HypergraphEncoder
from src.retrieval.memory_bank import MemoryBank


class GroupEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_group_prototypes: int = 8,
        use_semantic_edges: bool = True,
        use_weighted_edges: bool = True,
        prototype_top_k: int = 2,
        include_time_edges: bool = True,
        include_prototype_edges: bool = True,
    ) -> None:
        super().__init__()
        self.use_semantic_edges = bool(use_semantic_edges)
        self.use_weighted_edges = bool(use_weighted_edges)
        self.prototype_top_k = int(prototype_top_k)
        self.include_time_edges = bool(include_time_edges)
        self.include_prototype_edges = bool(include_prototype_edges)
        self.hypergraph_encoder = HypergraphEncoder(hidden_dim, num_layers=num_layers, dropout=dropout)
        self.group_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cluster_head = nn.Linear(hidden_dim, num_group_prototypes)

    def forward(
        self,
        *,
        current_state: torch.Tensor,
        retrieval_payload: Mapping[str, Any],
        memory_bank: MemoryBank,
    ) -> dict[str, Any]:
        device = current_state.device
        contexts = []
        available_mask = []
        cluster_logits = []
        metadata = []

        for graph in build_batch_hypergraphs(
            current_state,
            retrieval_payload,
            memory_bank,
            use_semantic_edges=self.use_semantic_edges,
            use_weighted_edges=self.use_weighted_edges,
            prototype_top_k=self.prototype_top_k,
            include_time_edges=self.include_time_edges,
            include_prototype_edges=self.include_prototype_edges,
        ):
            node_features = graph["node_features"].to(device=device, dtype=torch.float32)
            incidence = graph["incidence"].to(device=device, dtype=torch.float32)
            edge_weights = graph["edge_weights"].to(device=device, dtype=torch.float32)
            if graph["metadata"]["num_neighbors"] <= 0:
                contexts.append(torch.zeros(current_state.shape[1], dtype=torch.float32, device=device))
                available_mask.append(False)
                cluster_logits.append(torch.zeros(self.cluster_head.out_features, dtype=torch.float32, device=device))
                metadata.append(graph["metadata"])
                continue

            encoded = self.hypergraph_encoder(node_features, incidence, edge_weights=edge_weights)
            group_context = self.group_projection(
                torch.cat([encoded["current_embedding"], encoded["graph_embedding"]], dim=-1)
            )
            contexts.append(group_context)
            available_mask.append(True)
            row_cluster_logits = self.cluster_head(group_context)
            cluster_logits.append(row_cluster_logits)
            cluster_probability = torch.softmax(row_cluster_logits, dim=-1)
            dominant_cluster = int(cluster_probability.argmax(dim=-1).item())
            dominant_score = float(cluster_probability[dominant_cluster].item())
            edge_type_counts = dict(graph["metadata"].get("edge_type_counts", {}))
            dominant_edge_type = max(edge_type_counts.items(), key=lambda item: item[1])[0] if edge_type_counts else "global"
            metadata.append(
                {
                    **graph["metadata"],
                    "graph_embedding_norm": float(encoded["graph_embedding"].norm().item()),
                    "cluster_confidence": dominant_score,
                    "dominant_cluster": dominant_cluster,
                    "dominant_edge_type": dominant_edge_type,
                    "include_time_edges": self.include_time_edges,
                    "include_prototype_edges": self.include_prototype_edges,
                }
            )

        stacked_logits = torch.stack(cluster_logits, dim=0)
        cluster_probability = torch.softmax(stacked_logits, dim=-1)
        return {
            "group_context": torch.stack(contexts, dim=0),
            "group_available_mask": torch.tensor(available_mask, dtype=torch.bool, device=device),
            "cluster_logits": stacked_logits,
            "cluster_probability": cluster_probability,
            "cluster_label": stacked_logits.argmax(dim=-1),
            "group_metadata": metadata,
        }
