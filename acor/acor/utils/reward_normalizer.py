from __future__ import annotations

import math
from typing import Optional

import torch


class RewardNormalizer:
    """
    Tracks running mean/variance of rewards for normalization.
    """

    def __init__(self, clip_value: Optional[float] = None, eps: float = 1e-8) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = eps
        self.clip_value = clip_value
        self.eps = eps

    def update(self, rewards: torch.Tensor) -> None:
        batch_mean = rewards.mean().item()
        batch_var = rewards.var(unbiased=False).item()
        batch_count = rewards.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total_count
        new_var = m2 / max(total_count, self.eps)

        self.mean = new_mean
        self.var = max(new_var, self.eps)
        self.count = total_count

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        std = math.sqrt(self.var + self.eps)
        normalized = (rewards - self.mean) / std
        if self.clip_value is not None:
            normalized = normalized.clamp(-self.clip_value, self.clip_value)
        return normalized
    