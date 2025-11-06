import torch

from acor.models.acor_policy import ACORPolicy


def _build_config():
    return {
        "model": {
            "k_neighbors": 2,
            "leader_k": 1,
            "obs_hidden": 32,
            "obs_embed": 16,
            "feature_dim": 32,
            "message_dim": 32,
            "policy_hidden": 32,
            "value_hidden": 32,
            "leader_hidden": 32,
            "trust_hidden": 32,
            "behavior_hidden": 32,
            "behavior_latent": 16,
            "behavior_input": 10,
            "history_window": 4,
            "intra_steps": 1,
            "inter_steps": 1,
            "group_temp": 1.0,
        }
    }


def test_acor_policy_shapes():
    batch = 3
    agents = 4
    obs_dim = 8
    pos_dim = 2
    action_dim = 5
    config = _build_config()

    policy = ACORPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        position_dim=pos_dim,
        config=config,
        device=torch.device("cpu"),
        num_agents=agents,
    )

    obs = torch.randn(batch, agents, obs_dim)
    pos = torch.randn(batch, agents, pos_dim)
    history = torch.randn(batch, agents, config["model"]["history_window"], config["model"]["behavior_input"])

    logits, values, aux = policy(obs, pos, history)

    assert logits.shape == (batch, agents, action_dim)
    assert values.shape == (batch, agents)
    assert aux["latent"].shape[0] == batch
    assert aux["edge_weights"].shape[0] == batch
