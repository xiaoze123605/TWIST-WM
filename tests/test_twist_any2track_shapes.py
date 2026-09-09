"""CPU-only shape and frozen-base checks for TWIST Any2Track V6."""

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))

from rsl_rl.modules import TwistAny2TrackActorCritic
from rsl_rl.algorithms import PPOAny2Track


def main():
    base_path = (
        ROOT
        / "legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt"
    )
    model = TwistAny2TrackActorCritic(
        num_actions=23,
        num_critic_observations=1322,
        base_actor_jit_path=str(base_path),
        base_obs_dim=1155,
        history_len=79,
        history_frame_dim=74,
        hist_state_dim=51,
        latent_dim=128,
    )
    obs = torch.randn(4, 7001)
    critic_obs = torch.randn(4, 1322)
    actions = model.act_inference(obs)
    base_actions = model.base_action(obs)
    values = model.evaluate(critic_obs)
    prediction = model.predict_world_model(obs, torch.randn(4, 23))

    assert actions.shape == (4, 23)
    assert values.shape == (4, 1)
    assert prediction.shape == (4, 51)
    assert torch.equal(actions, base_actions), "zero-init layer adapters must exactly preserve the base"

    prediction.square().mean().backward()
    encoder_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.history_encoder.parameters()
        if parameter.grad is not None
    )
    adapter_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.adapter.parameters()
        if parameter.grad is not None
    )
    assert encoder_grad > 0.0
    assert adapter_grad == 0.0, "world-model loss must not update policy adapters"

    with torch.no_grad():
        model.adapter[-1].bias[:12].fill_(0.1)
    stand_obs = torch.randn(4, 7001)
    default_leg_pose = stand_obs.new_tensor([
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    ])
    stand_obs[:, 4:8] = 0.0
    stand_obs[:, 8:20] = default_leg_pose
    algorithm = PPOAny2Track(
        env=None,
        actor_critic=model,
        device="cpu",
        adapter_reg_coef=0.05,
        stand_anchor_coef=2.0,
        synthetic_stand_anchor_coef=0.0,
    )
    stand_delta = model.action_delta(stand_obs)
    stand_loss, stand_ratio = algorithm._in_place_leg_anchor_loss(
        stand_obs, stand_delta
    )
    assert stand_loss > 0.0
    assert torch.isclose(stand_ratio, torch.tensor(1.0))
    moving_obs = stand_obs.clone()
    moving_obs[:, 4] = 0.2
    assert not torch.any(algorithm._in_place_leg_mask(moving_obs))
    print("TWIST Any2Track shape/base/gradient checks passed")


if __name__ == "__main__":
    main()
