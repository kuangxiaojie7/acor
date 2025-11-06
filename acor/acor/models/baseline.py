from __future__ import annotations

import torch
import torch.nn as nn


def mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2) -> nn.Sequential:
    layers = []
    dim = input_dim
    for _ in range(depth - 1):
        layers.extend(
            [
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ]
        )
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class SharedMAPPO(nn.Module):
    """
    Parameter-shared MAPPO baseline with centralized critic.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = mlp(obs_dim, hidden_dim, hidden_dim, depth=3)
        fused_dim = hidden_dim * 2
        self.actor = mlp(fused_dim, hidden_dim, action_dim, depth=3)
        self.critic = mlp(hidden_dim, hidden_dim, 1, depth=3)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, agents, obs_dim = obs.shape
        flat = obs.view(batch * agents, obs_dim)
        embed = self.encoder(flat).view(batch, agents, -1)
        context = embed.mean(dim=1, keepdim=True).expand_as(embed)
        fused = torch.cat([embed, context], dim=-1)
        logits = self.actor(fused)
        global_value = self.critic(embed.mean(dim=1)).squeeze(-1)
        values = global_value.unsqueeze(-1).expand(batch, agents)
        return logits, values
