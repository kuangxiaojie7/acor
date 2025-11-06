from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..modules import (
    BehaviorEncoder,
    FuzzyTopKGrouping,
    HierarchicalConsensus,
    LeaderPotential,
    TrustEvaluator,
    elect_leaders,
)


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


class ACORPolicy(nn.Module):
    """
    Implements the ACOR policy combining fuzzy grouping, trust evaluation,
    leader election, and hierarchical consensus.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        position_dim: int,
        config: Dict,
        device: torch.device,
        num_agents: int,
    ) -> None:
        super().__init__()
        self.device = device
        self.num_agents = num_agents

        model_cfg = config["model"]
        self.k_neighbors = model_cfg["k_neighbors"]
        self.leader_k = model_cfg["leader_k"]

        self.obs_encoder = mlp(obs_dim, model_cfg["obs_hidden"], model_cfg["obs_embed"], depth=3)
        self.grouping = FuzzyTopKGrouping(self.k_neighbors, temperature=model_cfg.get("group_temp", 1.0))

        behavior_input = model_cfg["behavior_input"]
        behavior_hidden = model_cfg["behavior_hidden"]
        behavior_latent = model_cfg["behavior_latent"]
        self.behavior_encoder = BehaviorEncoder(behavior_input, behavior_hidden, behavior_latent)
        self.trust = TrustEvaluator(
            self_dim=model_cfg["obs_embed"],
            behavior_dim=behavior_latent,
            hidden_dim=model_cfg["trust_hidden"],
        )

        feature_dim = model_cfg["feature_dim"]
        self.feature_proj = mlp(model_cfg["obs_embed"] + behavior_latent, model_cfg["obs_hidden"], feature_dim, depth=2)
        self.leader_potential = LeaderPotential(feature_dim, model_cfg["leader_hidden"])

        self.consensus = HierarchicalConsensus(
            feature_dim=feature_dim,
            message_dim=model_cfg["message_dim"],
            intra_steps=model_cfg["intra_steps"],
            inter_steps=model_cfg["inter_steps"],
        )

        fused_dim = feature_dim * 2 + behavior_latent
        self.actor = mlp(fused_dim, model_cfg["policy_hidden"], action_dim, depth=3)
        self.critic = mlp(fused_dim, model_cfg["value_hidden"], 1, depth=3)

        self.history_window = model_cfg["history_window"]
        self.behavior_feature_dim = behavior_input
        self.position_dim = position_dim

        self.register_buffer("epsilon", torch.tensor(1e-8))

    def forward(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        history: torch.Tensor,
        dones: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        obs:
            Tensor [batch, agents, obs_dim]
        positions:
            Tensor [batch, agents, position_dim]
        """
        batch, agents, _ = obs.shape
        behavior_latent = self.behavior_encoder(history)

        obs_embed = self.obs_encoder(obs)
        neighbor_idx, membership = self.grouping(positions)

        # Gather neighbor behaviors
        behavior_neighbors = torch.gather(
            behavior_latent.unsqueeze(1).expand(batch, agents, agents, behavior_latent.shape[-1]),
            2,
            neighbor_idx.unsqueeze(-1).expand(batch, agents, neighbor_idx.shape[-1], behavior_latent.shape[-1]),
        )

        trust_scores = self.trust(obs_embed, behavior_neighbors)
        edge_weights = membership * trust_scores
        edge_weights = edge_weights / (edge_weights.sum(dim=-1, keepdim=True) + self.epsilon)

        agent_features = torch.cat([obs_embed, behavior_latent], dim=-1)
        agent_features = self.feature_proj(agent_features)

        potentials = self.leader_potential(agent_features)
        leader_mask, leader_indices = elect_leaders(potentials, neighbor_idx)
        leader_agent_mask = leader_mask.clone()

        max_leaders = leader_indices.shape[1]
        leader_pad_mask = leader_indices >= 0
        leader_feat = torch.zeros(
            batch,
            max_leaders,
            agent_features.shape[-1],
            device=obs.device,
            dtype=agent_features.dtype,
        )
        leader_pos = torch.zeros(
            batch,
            max_leaders,
            positions.shape[-1],
            device=positions.device,
            dtype=positions.dtype,
        )

        for b in range(batch):
            valid = leader_pad_mask[b]
            idx = leader_indices[b, valid]
            if idx.numel() == 0:
                continue
            leader_feat[b, valid] = agent_features[b, idx]
            leader_pos[b, valid] = positions[b, idx]

        leader_neighbors = torch.full(
            (batch, max_leaders, self.leader_k),
            -1,
            device=obs.device,
            dtype=torch.long,
        )
        leader_weights = torch.zeros(
            batch,
            max_leaders,
            self.leader_k,
            device=obs.device,
            dtype=edge_weights.dtype,
        )

        for b in range(batch):
            valid = leader_pad_mask[b]
            pad_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            count = pad_idx.numel()
            if count <= 1:
                continue
            pos = leader_pos[b, pad_idx]
            dist = torch.cdist(pos, pos, p=2)
            mask_self = torch.eye(count, device=obs.device, dtype=torch.bool)
            dist = dist.masked_fill(mask_self, float("inf"))
            k = min(self.leader_k, count - 1)
            topk_dist, topk_idx = torch.topk(dist, k=k, dim=-1, largest=False)
            weights = torch.softmax(-topk_dist, dim=-1)
            for i in range(count):
                dst_pad = pad_idx[i].item()
                neighbors_pad = pad_idx[topk_idx[i]]
                leader_neighbors[b, dst_pad, :k] = neighbors_pad
                leader_weights[b, dst_pad, :k] = weights[i]

        agent_latent, leader_latent = self.consensus(
            agent_features,
            neighbor_idx,
            edge_weights,
            leader_feat,
            leader_neighbors,
            leader_weights,
        )

        # Broadcast leader latent to agents via neighbor mapping.
        agent_to_slot = torch.full((batch, agents), -1, device=obs.device, dtype=torch.long)
        for b in range(batch):
            valid = leader_pad_mask[b]
            slot_idx = torch.arange(max_leaders, device=obs.device)[valid]
            agent_ids = leader_indices[b, valid]
            agent_to_slot[b, agent_ids] = slot_idx

        neighbor_slots = torch.gather(
            agent_to_slot.unsqueeze(1).expand(batch, agents, agents),
            2,
            neighbor_idx,
        )
        safe_slots = neighbor_slots.clamp(min=0)
        leader_embed_expanded = leader_latent.unsqueeze(1).expand(batch, agents, max_leaders, leader_latent.shape[-1])
        leader_messages = torch.gather(
            leader_embed_expanded,
            2,
            safe_slots.unsqueeze(-1).expand(batch, agents, neighbor_idx.shape[-1], leader_latent.shape[-1]),
        )
        mask = (neighbor_slots >= 0).to(edge_weights.dtype)
        broadcast = torch.sum(leader_messages * mask.unsqueeze(-1) * edge_weights.unsqueeze(-1), dim=-2)

        fused = torch.cat([agent_latent, broadcast, behavior_latent], dim=-1)
        logits = self.actor(fused)
        values = self.critic(fused).squeeze(-1)

        aux = {
            "edge_weights": edge_weights,
            "leader_mask": leader_agent_mask,
            "potentials": potentials,
            "latent": fused.detach(),
        }
        return logits, values, aux
