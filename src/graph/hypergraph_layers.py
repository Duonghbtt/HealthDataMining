from __future__ import annotations

import torch
from torch import nn


def hypergraph_propagation(
    node_features: torch.Tensor,
    incidence: torch.Tensor,
    *,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    node_features = torch.as_tensor(node_features, dtype=torch.float32)
    incidence = torch.as_tensor(incidence, dtype=torch.float32, device=node_features.device)
    if node_features.ndim != 2 or incidence.ndim != 2:
        raise ValueError("node_features and incidence must both be 2D tensors")
    if node_features.shape[0] != incidence.shape[0]:
        raise ValueError("node_features and incidence must agree on node count")

    num_edges = incidence.shape[1]
    resolved_edge_weights = (
        torch.ones(num_edges, dtype=torch.float32, device=node_features.device)
        if edge_weights is None
        else torch.as_tensor(edge_weights, dtype=torch.float32, device=node_features.device).flatten()
    )
    if resolved_edge_weights.shape[0] != num_edges:
        raise ValueError("edge_weights length must match the number of hyperedges")

    node_degree = (incidence * resolved_edge_weights.unsqueeze(0)).sum(dim=1).clamp(min=1.0)
    edge_degree = incidence.sum(dim=0).clamp(min=1.0)
    normed_nodes = node_features * node_degree.rsqrt().unsqueeze(-1)
    edge_messages = incidence.transpose(0, 1) @ normed_nodes
    edge_messages = edge_messages * edge_degree.reciprocal().unsqueeze(-1)
    edge_messages = edge_messages * resolved_edge_weights.unsqueeze(-1)
    propagated = incidence @ edge_messages
    return propagated * node_degree.rsqrt().unsqueeze(-1)


class HypergraphConv(nn.Module):
    def __init__(self, hidden_dim: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        incidence: torch.Tensor,
        *,
        edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        propagated = hypergraph_propagation(node_features, incidence, edge_weights=edge_weights)
        updated = self.linear(propagated)
        if updated.shape == node_features.shape:
            updated = updated + node_features
        return self.dropout(self.activation(updated))


class HypergraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, *, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [HypergraphConv(hidden_dim, dropout=dropout) for _ in range(max(int(num_layers), 1))]
        )

    def forward(
        self,
        node_features: torch.Tensor,
        incidence: torch.Tensor,
        *,
        edge_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        embeddings = torch.as_tensor(node_features, dtype=torch.float32)
        for layer in self.layers:
            embeddings = layer(embeddings, incidence, edge_weights=edge_weights)
        return {
            "node_embeddings": embeddings,
            "current_embedding": embeddings[0],
            "graph_embedding": embeddings.mean(dim=0),
        }
