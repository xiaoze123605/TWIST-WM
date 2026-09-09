"""Export a TWIST Any2Track V6 checkpoint to a 7001-D TorchScript actor."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rsl_rl.modules import TwistAny2TrackActorCritic


BASE_ACTOR = (
    "/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_stu_rl/"
    "0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt"
)


class ActorOnly(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, obs):
        return self.actor_critic.act_inference(obs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--base_actor_jit_path", default=BASE_ACTOR)
    parser.add_argument("--adapter_gain", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.ckpt)
    if args.out is None:
        checkpoint_tag = checkpoint_path.stem.replace("model_", "")
        gain_tag = "" if abs(args.adapter_gain - 1.0) < 1e-8 else f"-gain{args.adapter_gain:g}"
        args.out = str(
            checkpoint_path.parent
            / "traced"
            / f"{checkpoint_path.parent.name}-{checkpoint_tag}-any2track{gain_tag}-jit.pt"
        )

    model = TwistAny2TrackActorCritic(
        num_actions=23,
        num_critic_observations=1322,
        base_actor_jit_path=args.base_actor_jit_path,
        base_obs_dim=1155,
        history_len=79,
        history_frame_dim=74,
        hist_state_dim=51,
        latent_dim=128,
        world_model_hidden_dims=[512, 512, 256, 256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="silu",
        adapter_gain=args.adapter_gain,
    ).to(args.device)

    checkpoint = torch.load(args.ckpt, map_location=args.device)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    inference_state = {
        key: value
        for key, value in state.items()
        if not key.startswith("critic.") and not key.startswith("world_model.")
    }
    incompatible = model.load_state_dict(inference_state, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("critic.")]
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    model.layerwise_actor.adapter_gain = float(args.adapter_gain)
    model.adapter_gain = float(args.adapter_gain)
    model.eval()

    actor = ActorOnly(model).to(args.device).eval()
    example = torch.zeros(1, 7001, device=args.device)
    with torch.no_grad():
        traced = torch.jit.trace(actor, example)
        output = traced(example)
    if output.shape != (1, 23):
        raise RuntimeError(f"Exported actor returned {tuple(output.shape)}, expected (1, 23)")
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    print(f"Saved TWIST Any2Track V6 JIT to {output_path}")
    print(f"input_dim=7001 adapter_gain={args.adapter_gain}")


if __name__ == "__main__":
    main()
