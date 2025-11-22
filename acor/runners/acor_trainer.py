from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from ..envs import VectorMPE
from ..models import ACORPolicy
from ..modules import TrustMemory
from ..utils.buffers import RolloutBuffer
from ..utils.distributed import init_distributed_mode, is_distributed, is_main_process, local_rank
from ..utils.logger import JSONLogger, format_metrics
from ..utils.reward_normalizer import RewardNormalizer


class ACORTrainer:
    """
    Trainer implementing PPO-style optimization for the ACOR framework.
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
        reward_cfg = self.config["train"].get("reward", {})
        self.reward_scale = float(reward_cfg.get("scale", 1.0))
        self.reward_normalize = reward_cfg.get("normalize", True)
        self.reward_clip = reward_cfg.get("clip", None)
        self.reward_normalizer = RewardNormalizer(clip_value=self.reward_clip) if self.reward_normalize else None

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
        pos_dim = 2
        action_dim = env.act_dim
        num_agents = env.num_agents

        policy = ACORPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            position_dim=pos_dim,
            config=self.config,
            device=self.device,
            num_agents=num_agents,
        ).to(self.device)

        if is_distributed():
            policy = torch.nn.parallel.DistributedDataParallel(
                policy,
                device_ids=[local_rank()] if self.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        train_cfg = self.config["train"]
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=train_cfg["lr"],
            eps=train_cfg.get("adam_eps", 1e-8),
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("use_amp", True) and self.device.type == "cuda")

        model_cfg = self.config["model"]
        behavior_input = model_cfg["behavior_input"]
        history_window = model_cfg["history_window"]

        buffer = RolloutBuffer(
            horizon=train_cfg["rollout_length"],
            num_envs=exp_cfg["num_envs"],
            num_agents=num_agents,
            obs_dim=obs_dim,
            pos_dim=pos_dim,
            hidden_dim=model_cfg["feature_dim"] * 2 + model_cfg["behavior_latent"],
            history_shape=(history_window, behavior_input),
            device=self.device,
        )

        trust_memory = TrustMemory(
            num_envs=exp_cfg["num_envs"],
            num_agents=num_agents,
            feature_dim=behavior_input,
            history=history_window,
            device=self.device,
        )

        observations, positions = env.reset()
        done_mask = torch.zeros(exp_cfg["num_envs"], num_agents, device=self.device)
        last_actions = torch.zeros(exp_cfg["num_envs"], num_agents, action_dim, device=self.device)
        episode_returns_raw = torch.zeros(exp_cfg["num_envs"], device=self.device)
        recent_returns: list[float] = []

        total_updates = train_cfg["total_updates"]
        global_step = 0
        start_time = time.time()

        for update_idx in range(total_updates):
            buffer.reset()

            for step in range(train_cfg["rollout_length"]):
                behavior_features = torch.cat([observations, last_actions], dim=-1).detach()
                if behavior_features.shape[-1] != behavior_input:
                    raise ValueError("Behavior feature dimension mismatch; update behavior_input in config.")
                trust_memory.append(behavior_features, done_mask)
                history = trust_memory.get().detach()

                with torch.cuda.amp.autocast(enabled=train_cfg.get("use_amp", True) and self.device.type == "cuda"):
                    logits, values, aux = policy(observations, positions, history, dones=done_mask)

                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                step_result = env.step(actions)
                raw_rewards = step_result.rewards
                dones = step_result.dones

                rewards = raw_rewards
                if self.reward_normalizer is not None:
                    self.reward_normalizer.update(raw_rewards)
                    rewards = self.reward_normalizer.normalize(raw_rewards)
                rewards = rewards * self.reward_scale

                episode_returns_raw += raw_rewards.sum(dim=-1)
                done_envs = dones.sum(dim=-1) > 0
                if done_envs.any():
                    recent_returns.extend(episode_returns_raw[done_envs].tolist())
                    episode_returns_raw[done_envs] = 0.0

                buffer.add(
                    obs=observations.detach(),
                    pos=positions.detach(),
                    hidden=aux["latent"].detach(),
                    history=history.detach(),
                    actions=actions.detach(),
                    log_probs=log_probs.detach(),
                    values=values.detach(),
                    rewards=rewards.detach(),
                    dones=dones.detach(),
                )

                last_actions = F.one_hot(actions, num_classes=action_dim).float()
                last_actions = last_actions * (1.0 - dones.unsqueeze(-1))
                done_mask = dones
                observations = step_result.observations
                positions = step_result.positions
                global_step += exp_cfg["num_envs"] * num_agents

            with torch.no_grad():
                behavior_features = torch.cat([observations, last_actions], dim=-1)
                if behavior_features.shape[-1] != behavior_input:
                    raise ValueError("Behavior feature dimension mismatch; update behavior_input in config.")
                trust_memory.append(behavior_features, done_mask)
                history = trust_memory.get()
                _, next_values, _ = policy(observations, positions, history, dones=done_mask)

            buffer.compute_returns_advantages(
                next_values,
                done_mask,
                gamma=train_cfg["gamma"],
                gae_lambda=train_cfg["gae_lambda"],
            )

            metrics = self._update(policy, optimizer, scaler, buffer, train_cfg)
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
        scaler: torch.cuda.amp.GradScaler,
        buffer: RolloutBuffer,
        train_cfg: Dict,
    ) -> Dict[str, float]:
        clip_coef = train_cfg["clip_coef"]
        value_coef = train_cfg["value_coef"]
        entropy_coef = train_cfg["entropy_coef"]
        max_grad_norm = train_cfg["max_grad_norm"]
        minibatch_size = train_cfg["minibatch_size"]

        optimizer.zero_grad(set_to_none=True)
        last_policy_loss = torch.tensor(0.0, device=self.device)
        last_value_loss = torch.tensor(0.0, device=self.device)
        last_entropy = torch.tensor(0.0, device=self.device)

        for _ in range(train_cfg["update_epochs"]):
            for batch in buffer.get_minibatches(minibatch_size):
                obs = batch.obs.to(self.device)
                pos = batch.pos.to(self.device)
                history = batch.history.to(self.device)
                actions = batch.actions.to(self.device)
                old_log_probs = batch.log_probs.to(self.device)
                returns = batch.returns.to(self.device)
                advantages = batch.advantages.to(self.device)

                with torch.cuda.amp.autocast(enabled=train_cfg.get("use_amp", True) and self.device.type == "cuda"):
                    logits, values, _ = policy(obs, pos, history)

                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions)
                    last_entropy = dist.entropy().mean()

                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
                    last_policy_loss = -torch.min(surr1, surr2).mean()

                    last_value_loss = 0.5 * F.mse_loss(values, returns)
                    loss = last_policy_loss + value_coef * last_value_loss - entropy_coef * last_entropy

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        return {
            "policy_loss": float(last_policy_loss.detach()),
            "value_loss": float(last_value_loss.detach()),
            "entropy": float(last_entropy.detach()),
        }

    def _save_checkpoint(self, policy: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
        if not self.output_dir or not is_main_process():
            return
        module = policy.module if isinstance(policy, torch.nn.parallel.DistributedDataParallel) else policy
        payload = {
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": self.config,
        }
        if self.reward_normalizer is not None:
            payload["reward_norm"] = {
                "mean": self.reward_normalizer.mean,
                "var": self.reward_normalizer.var,
                "count": self.reward_normalizer.count,
            }
        path = self.output_dir / f"checkpoint_{step:06d}.pt"
        torch.save(payload, path)
