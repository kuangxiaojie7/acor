from __future__ import annotations

import math

import torch


class ValueNormalizer:
    """
    Tracks running mean and variance of value targets so they can be normalized
    before computing the critic loss (as done in MAPPO-Light / PPO-Lagged).
    """

    def __init__(self, eps: float = 1e-5) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = eps
        self.eps = eps

    def update(self, values: torch.Tensor) -> None:
        batch_mean = values.mean().item()
        batch_var = values.var(unbiased=False).item()
        batch_count = values.numel()

        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total
        new_var = m2 / max(total, self.eps)

        self.mean = new_mean
        self.var = max(new_var, self.eps)
        self.count = total

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        std = math.sqrt(self.var + self.eps)
        return (values - self.mean) / std
