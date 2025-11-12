from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from ..envs import VectorMPE
from ..models import SharedMAPPO
from ..utils.buffers import RolloutBuffer
from ..utils.distributed import init_distributed_mode, is_main_process, local_rank
from ..utils.logger import JSONLogger, format_metrics
from ..utils.value_normalizer import ValueNormalizer


class MAPPOTrainerLight:
    """
    Lightweight MAPPO variant that mirrors MAPPO-Light features:
      * Advantage normalization
      * Value normalization
      * Linear learning-rate decay
      * Reward/return logging identical to ACOR
    """

    def __init__(self, config: Dict, output_dir: Optional[str] = None) -> None:
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        init_distributed_mode(config["hardware"].get("backend", "nccl"))
        self._setup_device()

        if self.output_dir and is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = JSONLogger(output_dir=str(self.output_dir) if self.output_dir else None)
        self.checkpoint_interval = config["experiment"].get("checkpoint_interval", 0)

    def _setup_device(self) -> None:
        hw = self.config["hardware"]
        if hw.get("device", "cuda") == "cuda" and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank()}")
        else:
            self.device = torch.device("cpu")

    def train(self) -> None:
        exp_cfg = self.config["experiment"]
        env = VectorMPE(
            env_name=exp_cfg["env_name"],
            num_envs=exp_cfg["num_envs"],
            max_cycles=exp_cfg.get("max_cycles", 50),
            device=self.device,
            seed=exp_cfg.get("seed", 1),
        )

        obs_dim = env.obs_dim
        action_dim = env.act_dim
        num_agents = env.num_agents

        policy = SharedMAPPO(obs_dim, action_dim, self.config["baseline"]["hidden_dim"]).to(self.device)
        train_cfg = self.config["train"]
        optimizer = torch.optim.Adam(policy.parameters(), lr=train_cfg["lr"], eps=train_cfg.get("adam_eps", 1e-5))

        total_updates = train_cfg["total_updates"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / max(total_updates, 1))

        buffer = RolloutBuffer(
            horizon=train_cfg["rollout_length"],
            num_envs=exp_cfg["num_envs"],
            num_agents=num_agents,
            obs_dim=obs_dim,
            pos_dim=2,
            hidden_dim=1,
            history_shape=(1, 1),
            device=self.device,
        )

        observations, positions = env.reset()
        dummy_history = torch.zeros(exp_cfg["num_envs"], num_agents, 1, 1, device=self.device)
        done_mask = torch.zeros(exp_cfg["num_envs"], num_agents, device=self.device)
        episode_returns = torch.zeros(exp_cfg["num_envs"], device=self.device)
        recent_returns: list[float] = []
        value_normalizer = ValueNormalizer()

        global_step = 0
        start_time = time.time()

        for update_idx in range(total_updates):
            buffer.reset()

            for _ in range(train_cfg["rollout_length"]):
                with torch.no_grad():
                    logits, values = policy(observations)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                step_result = env.step(actions)
                rewards = step_result.rewards
                dones = step_result.dones

                episode_returns += rewards.sum(dim=-1)
                done_envs = dones.sum(dim=-1) > 0
                if done_envs.any():
                    recent_returns.extend(episode_returns[done_envs].tolist())
                    episode_returns[done_envs] = 0.0

                buffer.add(
                    obs=observations.detach(),
                    pos=positions.detach(),
                    hidden=values.unsqueeze(-1).detach(),
                    history=dummy_history,
                    actions=actions.detach(),
                    log_probs=log_probs.detach(),
                    values=values.detach(),
                    rewards=rewards.detach(),
                    dones=dones.detach(),
                )

                observations = step_result.observations
                positions = step_result.positions
                done_mask = dones
                global_step += exp_cfg["num_envs"] * num_agents

            with torch.no_grad():
                _, next_values = policy(observations)

            buffer.compute_returns_advantages(
                next_values,
                done_mask,
                gamma=train_cfg["gamma"],
                gae_lambda=train_cfg["gae_lambda"],
            )

            metrics = self._update(policy, optimizer, value_normalizer, buffer, train_cfg)
            scheduler.step()

            metrics["fps"] = global_step / (time.time() - start_time + 1e-9)
            metrics["global_step"] = global_step
            metrics["update"] = update_idx
            if recent_returns:
                metrics["episode_return"] = float(sum(recent_returns) / len(recent_returns))
                recent_returns.clear()
            else:
                metrics["episode_return"] = 0.0

            if is_main_process() and update_idx % exp_cfg.get("log_interval", 10) == 0:
                self.logger.log(metrics)
                print(format_metrics(update_idx, metrics))

            if self.checkpoint_interval and update_idx % self.checkpoint_interval == 0:
                self._save_checkpoint(policy, optimizer, update_idx)

        env.close()
        self.logger.close()

    def _update(
        self,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        value_normalizer: ValueNormalizer,
        buffer: RolloutBuffer,
        train_cfg: Dict,
    ) -> Dict[str, float]:
        clip_coef = train_cfg["clip_coef"]
        value_coef = train_cfg["value_coef"]
        entropy_coef = train_cfg["entropy_coef"]
        max_grad_norm = train_cfg["max_grad_norm"]

        last_policy_loss = torch.tensor(0.0, device=self.device)
        last_value_loss = torch.tensor(0.0, device=self.device)
        last_entropy = torch.tensor(0.0, device=self.device)

        for _ in range(train_cfg["update_epochs"]):
            for batch in buffer.get_minibatches(train_cfg["minibatch_size"]):
                obs = batch.obs.to(self.device)
                actions = batch.actions.to(self.device)
                old_log_probs = batch.log_probs.to(self.device)
                returns = batch.returns.to(self.device)
                advantages = batch.advantages.to(self.device)

                value_normalizer.update(returns)
                norm_returns = value_normalizer.normalize(returns)

                logits, values = policy(obs)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)
                last_entropy = dist.entropy().mean()

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
                last_policy_loss = -torch.min(surr1, surr2).mean()
                last_value_loss = 0.5 * F.mse_loss(values, norm_returns)

                loss = last_policy_loss + value_coef * last_value_loss - entropy_coef * last_entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        return {
            "policy_loss": float(last_policy_loss.detach()),
            "value_loss": float(last_value_loss.detach()),
            "entropy": float(last_entropy.detach()),
        }

    def _save_checkpoint(self, policy: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
        if not self.output_dir or not is_main_process():
            return
        payload = {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": self.config,
        }
        path = self.output_dir / f"checkpoint_{step:06d}.pt"
        torch.save(payload, path)
