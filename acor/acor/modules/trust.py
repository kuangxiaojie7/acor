from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class BehaviorEncoder(nn.Module):
    """
    Encodes short behavior histories into latent embeddings.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sequence:
            Tensor [batch, agents, history, feat_dim]
        """
        batch, agents, history, feat_dim = sequence.shape
        seq_flat = sequence.view(batch * agents, history, feat_dim)
        _, hidden = self.rnn(seq_flat)
        latent = self.fc(hidden[-1])
        return latent.view(batch, agents, -1)


class TrustEvaluator(nn.Module):
    """
    Compute directional trust scores between agents based on local state and neighbor behavior.
    """

    def __init__(self, self_dim: int, behavior_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(self_dim + behavior_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        self_state: torch.Tensor,
        neighbor_behavior: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        self_state:
            Tensor [batch, agents, dim]
        neighbor_behavior:
            Tensor [batch, agents, k, dim]

        Returns
        -------
        trust:
            Tensor [batch, agents, k] in [0, 1]
        """
        batch, agents, k, dim = neighbor_behavior.shape
        expanded_self = self_state.unsqueeze(-2).expand(batch, agents, k, self_state.shape[-1])
        features = torch.cat([expanded_self, neighbor_behavior], dim=-1)
        logits = self.scorer(features).squeeze(-1)
        trust = torch.sigmoid(logits)
        return trust


class TrustMemory:
    """
    Maintains sliding window behavior histories per agent.
    """

    def __init__(self, num_envs: int, num_agents: int, feature_dim: int, history: int, device: torch.device) -> None:
        self.history = history
        self.buffer = torch.zeros(num_envs, num_agents, history, feature_dim, device=device)

    def append(self, features: torch.Tensor, dones: torch.Tensor | None = None) -> None:
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=2)
        if dones is not None:
            reset_mask = dones > 0.5
            if reset_mask.any():
                self.buffer[reset_mask].fill_(0.0)
        self.buffer[:, :, -1].copy_(features)

    def get(self) -> torch.Tensor:
        return self.buffer.clone()
