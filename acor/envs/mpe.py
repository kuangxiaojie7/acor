from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from pettingzoo.mpe import simple_spread_v3, simple_tag_v3


ENV_REGISTRY = {
    "mpe_simple_spread_v3": lambda max_cycles: simple_spread_v3.parallel_env(
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode=None,
    ),
    "mpe_simple_tag_v3": lambda max_cycles: simple_tag_v3.parallel_env(
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode=None,
    ),
}


@dataclass
class StepResult:
    observations: torch.Tensor
    positions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    infos: List[Dict]


class VectorMPE:
    """
    Simple vectorized wrapper over PettingZoo MPE parallel API.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        max_cycles: int,
        device: torch.device,
        seed: int = 1,
    ) -> None:
        if env_name not in ENV_REGISTRY:
            raise ValueError(f"Unsupported environment {env_name}")
        self.device = device
        self.num_envs = num_envs
        self.max_cycles = max_cycles
        self._rng = np.random.default_rng(seed)

        self.envs = []
        for idx in range(num_envs):
            env = ENV_REGISTRY[env_name](max_cycles=max_cycles)
            obs = env.reset(seed=seed + idx)
            if isinstance(obs, tuple):
                obs = obs[0]
            self.envs.append(env)

        ref = self.envs[0]
        self.possible_agents = ref.possible_agents
        self.num_agents = len(self.possible_agents)
        self.obs_dim = max(
            ref.observation_spaces[agent].shape[0] for agent in self.possible_agents
        )
        self.act_dim = ref.action_spaces[self.possible_agents[0]].n

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_batch: List[torch.Tensor] = []
        pos_batch: List[torch.Tensor] = []
        for env in self.envs:
            obs = env.reset(seed=int(self._rng.integers(0, 1_000_000)))
            if isinstance(obs, tuple):
                obs = obs[0]
            obs_tensor, pos_tensor = self._to_tensors(obs)
            obs_batch.append(obs_tensor)
            pos_batch.append(pos_tensor)
        return (
            torch.stack(obs_batch).to(self.device),
            torch.stack(pos_batch).to(self.device),
        )

    def step(self, actions: torch.Tensor) -> StepResult:
        obs_batch: List[torch.Tensor] = []
        pos_batch: List[torch.Tensor] = []
        rewards_batch: List[torch.Tensor] = []
        dones_batch: List[torch.Tensor] = []
        infos: List[Dict] = []

        for env_idx, env in enumerate(self.envs):
            action_dict = {
                agent: int(actions[env_idx, agent_idx].item())
                for agent_idx, agent in enumerate(self.possible_agents)
            }
            obs, reward, terminated, truncated, info = env.step(action_dict)
            if isinstance(obs, tuple):
                obs = obs[0]
            obs_tensor, pos_tensor = self._to_tensors(obs)
            obs_batch.append(obs_tensor)
            pos_batch.append(pos_tensor)
            rewards = torch.tensor(
                [reward.get(agent, 0.0) for agent in self.possible_agents],
                dtype=torch.float32,
            )
            dones = torch.tensor(
                [
                    float(terminated.get(agent, False) or truncated.get(agent, False))
                    for agent in self.possible_agents
                ],
                dtype=torch.float32,
            )

            if torch.all(dones > 0.5):
                reset_obs = env.reset(seed=int(self._rng.integers(0, 1_000_000)))
                if isinstance(reset_obs, tuple):
                    reset_obs = reset_obs[0]
                reset_obs_tensor, reset_pos_tensor = self._to_tensors(reset_obs)
                obs_batch[-1] = reset_obs_tensor
                pos_batch[-1] = reset_pos_tensor

            rewards_batch.append(rewards)
            dones_batch.append(dones)
            infos.append(info)

        return StepResult(
            observations=torch.stack(obs_batch).to(self.device),
            positions=torch.stack(pos_batch).to(self.device),
            rewards=torch.stack(rewards_batch).to(self.device),
            dones=torch.stack(dones_batch).to(self.device),
            infos=infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def _to_tensors(self, obs: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        target_dim = self.obs_dim
        obs_tensors = []
        for agent in self.possible_agents:
            vec = torch.as_tensor(obs[agent], dtype=torch.float32)
            if vec.shape[0] < target_dim:
                pad = torch.zeros(target_dim - vec.shape[0], dtype=vec.dtype)
                vec = torch.cat([vec, pad], dim=0)
            elif vec.shape[0] > target_dim:
                vec = vec[:target_dim]
            obs_tensors.append(vec)
        obs_tensor = torch.stack(obs_tensors)
        positions = obs_tensor[:, 2:4]
        return obs_tensor, positions
