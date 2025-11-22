from __future__ import annotations

import torch
import torch.nn as nn


class WeightedMessagePassing(nn.Module):
    def __init__(self, feature_dim: int, message_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(feature_dim, message_dim),
            nn.GELU(),
            nn.Linear(message_dim, message_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(feature_dim + message_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        neighbor_idx: torch.Tensor,
        weights: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, nodes, dim = node_features.shape
        k = neighbor_idx.shape[-1]
        safe_idx = neighbor_idx.clamp(min=0)
        gather_index = safe_idx.unsqueeze(-1).expand(batch, nodes, k, dim)
        neighbors = torch.gather(
            node_features.unsqueeze(1).expand(batch, nodes, nodes, dim),
            2,
            gather_index,
        )
        messages = self.message(neighbors)
        if neighbor_mask is not None:
            messages = messages * neighbor_mask.unsqueeze(-1)
            weights = weights * neighbor_mask
        aggregated = torch.sum(messages * weights.unsqueeze(-1), dim=-2)
        updated = self.update(torch.cat([node_features, aggregated], dim=-1))
        return updated


class HierarchicalConsensus(nn.Module):
    """
    Runs intra-agent and inter-leader message passing returning latent embeddings.
    """

    def __init__(
        self,
        feature_dim: int,
        message_dim: int,
        intra_steps: int,
        inter_steps: int,
    ) -> None:
        super().__init__()
        self.intra = WeightedMessagePassing(feature_dim, message_dim)
        self.inter = WeightedMessagePassing(feature_dim, message_dim)
        self.intra_steps = intra_steps
        self.inter_steps = inter_steps

    def forward(
        self,
        agent_features: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_weights: torch.Tensor,
        leader_features: torch.Tensor,
        leader_neighbors: torch.Tensor,
        leader_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = agent_features
        for _ in range(self.intra_steps):
            latent = self.intra(latent, neighbor_idx, neighbor_weights)

        leader_latent = leader_features
        for _ in range(self.inter_steps):
            leader_latent = self.inter(
                leader_latent,
                leader_neighbors,
                leader_weights,
                neighbor_mask=(leader_neighbors >= 0).float(),
            )

        return latent, leader_latent
