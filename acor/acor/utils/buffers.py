from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import torch


@dataclass
class Batch:
    obs: torch.Tensor  # [batch, agents, obs_dim]
    pos: torch.Tensor  # [batch, agents, pos_dim]
    hidden: torch.Tensor  # [batch, agents, hidden_dim]
    history: torch.Tensor  # [batch, agents, history, feat]
    actions: torch.Tensor  # [batch, agents]
    log_probs: torch.Tensor  # [batch, agents]
    values: torch.Tensor  # [batch, agents]
    returns: torch.Tensor  # [batch, agents]
    advantages: torch.Tensor  # [batch, agents]


class RolloutBuffer:
    """
    Rollout buffer tailored for PPO-style updates in ACOR.
    """

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        pos_dim: int,
        hidden_dim: int,
        history_shape: tuple[int, int],
        device: torch.device,
    ) -> None:
        shape = (horizon, num_envs, num_agents)
        self.device = device
        self.obs = torch.zeros(*shape, obs_dim, device=device)
        self.pos = torch.zeros(*shape, pos_dim, device=device)
        self.hidden = torch.zeros(*shape, hidden_dim, device=device)
        history_len, history_feat = history_shape
        self.history = torch.zeros(*shape, history_len, history_feat, device=device)
        self.actions = torch.zeros(*shape, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(*shape, device=device)
        self.values = torch.zeros(*shape, device=device)
        self.rewards = torch.zeros(*shape, device=device)
        self.dones = torch.zeros(*shape, device=device)
        self.advantages = torch.zeros(*shape, device=device)
        self.returns = torch.zeros(*shape, device=device)
        self._step = 0

    def add(
        self,
        obs: torch.Tensor,
        pos: torch.Tensor,
        hidden: torch.Tensor,
        history: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        if self._step >= self.obs.shape[0]:
            raise RuntimeError("RolloutBuffer full, call reset().")
        self.obs[self._step].copy_(obs)
        self.pos[self._step].copy_(pos)
        self.hidden[self._step].copy_(hidden)
        self.history[self._step].copy_(history)
        self.actions[self._step].copy_(actions)
        self.log_probs[self._step].copy_(log_probs)
        self.values[self._step].copy_(values)
        self.rewards[self._step].copy_(rewards)
        self.dones[self._step].copy_(dones)
        self._step += 1

    def compute_returns_advantages(
        self,
        last_value: torch.Tensor,
        last_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        advantage = torch.zeros_like(last_value, device=self.device)
        next_value = last_value
        next_non_terminal = 1.0 - last_done

        for t in reversed(range(self.obs.shape[0])):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
            self.advantages[t] = advantage
            self.returns[t] = advantage + self.values[t]
            next_value = self.values[t]
            next_non_terminal = mask

    def get_minibatches(self, batch_size: int) -> Generator[Batch, None, None]:
        num_steps, num_envs, num_agents, _ = self.obs.shape
        total = num_steps * num_envs
        indices = torch.randperm(total, device=self.device)

        def flatten(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(total, num_agents, *tensor.shape[3:])

        obs = flatten(self.obs)
        pos = flatten(self.pos)
        hidden = flatten(self.hidden)
        history = flatten(self.history)
        actions = flatten(self.actions).squeeze(-1) if self.actions.dim() == 5 else flatten(self.actions)
        log_probs = flatten(self.log_probs)
        values = flatten(self.values)
        returns = flatten(self.returns)
        advantages = flatten(self.advantages)

        for start in range(0, total, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            yield Batch(
                obs=obs[idx],
                pos=pos[idx],
                hidden=hidden[idx],
                history=history[idx],
                actions=actions[idx],
                log_probs=log_probs[idx],
                values=values[idx],
                returns=returns[idx],
                advantages=advantages[idx],
            )

    def reset(self) -> None:
        self._step = 0
