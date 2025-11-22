from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FuzzyTopKGrouping(nn.Module):
    """
    Compute soft memberships for top-k neighbor groups using distance-aware weights.
    """

    def __init__(self, k_neighbors: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.k = k_neighbors
        self.temperature = temperature

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        positions:
            Tensor of shape [batch, agents, pos_dim]

        Returns
        -------
        neighbor_idx:
            Long tensor [batch, agents, k] with neighbor indices.
        membership:
            Tensor [batch, agents, k] with soft weights (sum to 1 over k).
        """
        batch, agents, _ = positions.shape
        if agents <= 1:
            idx = torch.zeros(batch, agents, 1, dtype=torch.long, device=positions.device)
            weights = torch.ones(batch, agents, 1, device=positions.device)
            return idx, weights

        distances = torch.cdist(positions, positions, p=2)
        mask_self = torch.eye(agents, device=positions.device, dtype=torch.bool).unsqueeze(0)
        distances = distances.masked_fill(mask_self, float("inf"))

        k = min(self.k, agents - 1)
        dist_values, neighbor_idx = torch.topk(distances, k=k, dim=-1, largest=False)

        scores = -dist_values / max(self.temperature, 1e-6)
        membership = F.softmax(scores, dim=-1)
        return neighbor_idx, membership
