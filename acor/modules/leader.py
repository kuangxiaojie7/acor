from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class LeaderPotential(nn.Module):
    """
    Predict scalar leader potential from agent embeddings.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def elect_leaders(potentials: torch.Tensor, neighbor_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Elect leaders via local maxima rule.

    Parameters
    ----------
    potentials:
        Tensor [batch, agents]
    neighbor_idx:
        Tensor [batch, agents, k]

    Returns
    -------
    leader_mask:
        Bool tensor [batch, agents] indicating leaders.
    leader_indices:
        List of indices per batch as padded tensor with -1 for unused entries.
    """
    batch, agents = potentials.shape
    k = neighbor_idx.shape[-1]
    expanded_potential = potentials.unsqueeze(-1).expand(batch, agents, k)
    neighbor_potentials = torch.gather(
        potentials.unsqueeze(1).expand(batch, agents, agents),
        2,
        neighbor_idx,
    )
    is_leader = expanded_potential >= neighbor_potentials
    leader_mask = torch.all(is_leader, dim=-1)

    max_leaders = agents
    leader_indices = torch.full((batch, max_leaders), -1, device=potentials.device, dtype=torch.long)
    for b in range(batch):
        leaders = torch.nonzero(leader_mask[b], as_tuple=False).flatten()
        if leaders.numel() == 0:
            max_idx = torch.argmax(potentials[b])
            leaders = torch.tensor([max_idx], device=potentials.device)
            leader_mask[b, max_idx] = True
        count = leaders.numel()
        leader_indices[b, :count] = leaders
    return leader_mask, leader_indices
