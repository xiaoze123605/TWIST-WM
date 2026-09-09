"""
Export a trained dual-branch TWIST+AnyAdapter actor to TorchScript.

Targets the dual-branch policy (use_dual_branch_adapter=True) trained in
g1_twist_anyadapter_dual/dual_formal_v1. Unlike
export_twist_anyadapter_jit.py (single-branch presets, strict=False load),
this script:

  1. hard-codes the dual training configuration (no presets to get wrong),
  2. loads the checkpoint with strict=True so no adapter key can be
     silently dropped (a single-branch checkpoint fails loudly here),
  3. verifies the traced artifact against eager inference on random inputs
     before saving, including a branch-mode structural check.

Inference knobs (delta scales / branch gains / adapter gain) are plain
attributes, not learned parameters, so they can be overridden at export
time without retraining.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rsl_rl.modules.actor_critic_twist_anyadapter import TwistAnyAdapterActorCritic


# --- Dual training configuration (g1_mimic_distill_anyadapter_config.py) ---
DEFAULT_BASE_ACTOR_JIT_PATH = (
    "/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_stu_rl/"
    "0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt"
)
BASE_OBS_DIM = 1155
NUM_ACTIONS = 23
NUM_CRITIC_OBS = 2635
HISTORY_LEN = 20
HISTORY_FRAME_DIM = 74
HIST_STATE_DIM = 51
LATENT_DIM = 32
ADAPTER_HIDDEN_DIMS = (128, 128)
# ANYADAPTER_STATE_INDICES from the config: 51 dims, indices 31..81.
WM_TARGET_INDICES = list(range(31, 82))
HISTORY_POLICY_GRAD_SCALE = 0.10
DYNAMICS_ACTION_DELTA_SCALE = 0.05
TRACKING_ACTION_DELTA_SCALE = 0.05
DYNAMICS_BRANCH_GAIN = 1.0
TRACKING_BRANCH_GAIN = 1.0
INIT_NOISE_STD = 0.05
DEFAULT_REF_DOF_POS = [
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.4, 0.0, 1.2,
    0.0, -0.4, 0.0, 1.2,
]
EXAMPLE_DIM = BASE_OBS_DIM + HISTORY_LEN * HISTORY_FRAME_DIM  # 2635, no context dim
TRAINING_ONLY_PREFIXES = ("base_actor.", "critic.", "world_model.")


def default_output_path(ckpt_path: str, branch_mode: str, adapter_gain: float) -> str:
    ckpt = Path(ckpt_path)
    checkpoint_tag = ckpt.stem.replace("model_", "")
    run_dir = ckpt.parent
    suffix = branch_mode
    if abs(adapter_gain - 1.0) > 1e-8:
        suffix += f"-gain{adapter_gain:g}"
    return str(run_dir / "traced" / f"{run_dir.name}-{checkpoint_tag}-{suffix}-jit.pt")


def build_actor(args: argparse.Namespace) -> TwistAnyAdapterActorCritic:
    return TwistAnyAdapterActorCritic(
        num_prop=BASE_OBS_DIM,
        num_critic_obs=NUM_CRITIC_OBS,
        num_priv_latent=0,
        num_hist=HISTORY_LEN,
        num_actions=NUM_ACTIONS,
        base_actor_jit_path=args.base_actor_jit_path,
        base_obs_dim=BASE_OBS_DIM,
        history_len=HISTORY_LEN,
        history_frame_dim=HISTORY_FRAME_DIM,
        hist_state_dim=HIST_STATE_DIM,
        latent_dim=LATENT_DIM,
        wm_target_indices=WM_TARGET_INDICES,
        adapter_hidden_dims=ADAPTER_HIDDEN_DIMS,
        init_noise_std=INIT_NOISE_STD,
        action_delta_scale=args.dynamics_delta_scale,
        adapter_gain=args.adapter_gain,
        use_dual_branch_adapter=True,
        dynamics_branch_gain=args.dynamics_branch_gain,
        tracking_branch_gain=args.tracking_branch_gain,
        adapter_branch_mode=args.branch_mode,
        dynamics_action_delta_scale=args.dynamics_delta_scale,
        tracking_action_delta_scale=args.tracking_delta_scale,
        default_ref_dof_pos=DEFAULT_REF_DOF_POS,
        use_tracking_error_adapter_input=True,
        compact_adapter_input=True,
        history_policy_grad_scale=HISTORY_POLICY_GRAD_SCALE,
        adapter_context_dim=0,
        freeze_base=True,
    ).to(args.device)


def check_base_actor_matches(model: TwistAnyAdapterActorCritic, state: dict) -> None:
    """The checkpoint's base_actor must be the same JIT as --base_actor_jit_path."""
    checkpoint_base = {
        key: value for key, value in state.items() if key.startswith("base_actor.")
    }
    for key, current_value in model.base_actor.state_dict().items():
        checkpoint_value = checkpoint_base.get("base_actor." + key)
        if checkpoint_value is not None and not torch.equal(
            current_value.detach().cpu(), checkpoint_value.detach().cpu()
        ):
            raise RuntimeError(
                "Checkpoint base actor does not match --base_actor_jit_path. "
                f"First mismatch: base_actor.{key}"
            )


def load_strict(model: TwistAnyAdapterActorCritic, state: dict) -> dict:
    """Strictly load inference keys; fail loudly on any missing/unexpected key."""
    filtered = {
        k: v
        for k, v in state.items()
        if not k.startswith(TRAINING_ONLY_PREFIXES)
    }
    expected = {
        k
        for k in model.state_dict().keys()
        if not k.startswith(TRAINING_ONLY_PREFIXES)
    }
    missing = sorted(expected - set(filtered.keys()))
    unexpected = sorted(set(filtered.keys()) - expected)
    if missing or unexpected:
        lines = ["Checkpoint does not match the dual-branch actor architecture:"]
        if missing:
            lines.append(f"  missing keys in checkpoint ({len(missing)}):")
            lines += [f"    {k}" for k in missing[:10]]
        if unexpected:
            lines.append(f"  unexpected keys in checkpoint ({len(unexpected)}):")
            lines += [f"    {k}" for k in unexpected[:10]]
            if any(k.startswith("adapter.") for k in unexpected):
                lines.append(
                    "  Note: checkpoint has non-dual adapter keys; it may be a "
                    "single-branch checkpoint. Use export_twist_anyadapter_jit.py "
                    "for those instead."
                )
        raise RuntimeError("\n".join(lines))
    # Set equality was verified above, so strict=False is safe here (strict=True
    # would also require the excluded base_actor./critic./world_model. keys).
    model.load_state_dict(filtered, strict=False)
    return filtered


def verify_traced(
    actor: torch.nn.Module,
    ac: TwistAnyAdapterActorCritic,
    traced: torch.jit.ScriptModule,
    args: argparse.Namespace,
    device: str,
) -> None:
    torch.manual_seed(0)
    samples = [torch.zeros(1, EXAMPLE_DIM, device=device)]
    samples += [torch.randn(args.num_verify_samples - 1, EXAMPLE_DIM, device=device)]

    with torch.no_grad():
        eager = [ac.act_inference(s) for s in samples]
        base = [ac.base_action(s) for s in samples]
        components = [ac.get_adapter_delta_components(s) for s in samples]
    jit = [traced(s) for s in samples]

    # 0) Server-probe compatibility: the reshape guard must reject every
    #    non-2635-D probe (1155 TWIST / 2637 heading / 7001 Any2Track) so the
    #    deploy server detects this file as a plain 2635-D AnyAdapter policy.
    for dim in (BASE_OBS_DIM, BASE_OBS_DIM + 2, 7001):
        try:
            with torch.no_grad():
                traced(torch.zeros(1, dim, device=device))
        except Exception:
            continue
        raise RuntimeError(
            f"Traced JIT accepted a {dim}-D obs probe; the input-dim guard is broken."
        )

    # 1) Trace parity: traced graph must reproduce eager inference exactly.
    max_diff = max((e - j).abs().max().item() for e, j in zip(eager, jit))
    print(f"[verify] traced vs eager max abs diff: {max_diff:.3e}")
    if max_diff > 1e-5:
        raise RuntimeError("Traced JIT does not match eager inference; export aborted.")

    # 2) Branch-mode structure: traced delta must equal the mode's combination
    #    of the eager branch components (dyn_only / err_only / full).
    mode = args.branch_mode
    dyn_gain = args.dynamics_branch_gain
    err_gain = args.tracking_branch_gain
    max_mode_diff = 0.0
    for j, b, (delta_dyn, delta_err) in zip(jit, base, components):
        if mode == "dyn_only":
            expected = dyn_gain * delta_dyn
        elif mode == "err_only":
            expected = err_gain * delta_err
        else:
            expected = dyn_gain * delta_dyn + err_gain * delta_err
        max_mode_diff = max(max_mode_diff, ((j - b) - args.adapter_gain * expected).abs().max().item())
    print(f"[verify] branch-mode '{mode}' structural max abs diff: {max_mode_diff:.3e}")
    if max_mode_diff > 1e-5:
        raise RuntimeError(
            f"Traced delta does not follow adapter_branch_mode={mode!r}; export aborted."
        )

    # 3) Amplitude bound: |a - a_base| <= gain * (dyn_gain*dyn_scale + err_gain*err_scale).
    bound = args.adapter_gain * (
        dyn_gain * args.dynamics_delta_scale + err_gain * args.tracking_delta_scale
    )
    max_delta = max((j - b).abs().max().item() for j, b in zip(jit, base))
    print(f"[verify] max |delta| = {max_delta:.4f} (bound = {bound:.4f})")
    if max_delta > bound + 1e-4:
        raise RuntimeError("Adapter delta exceeds the delta-scale bound; export aborted.")

    # 4) Live-branch report: mean |delta| per branch over the samples.
    mean_dyn = sum(d.abs().mean().item() for d, _ in components) / len(components)
    mean_err = sum(e.abs().mean().item() for _, e in components) / len(components)
    print(f"[verify] mean |delta_dyn| = {mean_dyn:.4f}, mean |delta_err| = {mean_err:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to the dual TWIST adapter checkpoint .pt")
    parser.add_argument("--out", default=None, help="Output JIT actor path. Defaults to <run_dir>/traced/<run>-<ckpt>-<branch_mode>-jit.pt")
    parser.add_argument("--base_actor_jit_path", default=DEFAULT_BASE_ACTOR_JIT_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--branch_mode", choices=["full", "dyn_only", "err_only"], default="full")
    parser.add_argument("--adapter_gain", type=float, default=1.0)
    parser.add_argument("--dynamics_delta_scale", type=float, default=DYNAMICS_ACTION_DELTA_SCALE)
    parser.add_argument("--tracking_delta_scale", type=float, default=TRACKING_ACTION_DELTA_SCALE)
    parser.add_argument("--dynamics_branch_gain", type=float, default=DYNAMICS_BRANCH_GAIN)
    parser.add_argument("--tracking_branch_gain", type=float, default=TRACKING_BRANCH_GAIN)
    parser.add_argument("--num_verify_samples", type=int, default=8)
    args = parser.parse_args()

    if args.out is None:
        args.out = default_output_path(args.ckpt, args.branch_mode, args.adapter_gain)

    model = build_actor(args)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    check_base_actor_matches(model, state)
    loaded = load_strict(model, state)
    model.eval()

    adapter_keys = [k for k in loaded if k.startswith("adapter.")]
    adapter_norm = sum(
        v.detach().float().norm().item() ** 2 for k, v in loaded.items() if k in adapter_keys
    ) ** 0.5
    print(f"[load] strict load OK: {len(loaded)} inference keys "
          f"({len(adapter_keys)} adapter keys, adapter weight L2 norm = {adapter_norm:.4f})")

    class ActorOnly(torch.nn.Module):
        def __init__(self, ac):
            super().__init__()
            self.ac = ac

        def forward(self, obs):
            # Input-dim guard: the deploy server classifies policies by probing
            # 1155/2635/2637/7001-D inputs. The traced graph slices constant
            # columns, so without this reshape it would silently accept
            # 2637/7001-D probes and be misdetected as heading-aware/Any2Track
            # (79-frame history wrapper -> garbage actions). reshape fails
            # cleanly unless the input is an exact multiple of 2635 columns.
            obs = obs.reshape(-1, EXAMPLE_DIM)
            return self.ac.act_inference(obs)

    actor = ActorOnly(model).to(args.device).eval()
    example = torch.zeros(1, EXAMPLE_DIM, device=args.device)
    traced = torch.jit.trace(actor, example)

    verify_traced(actor, model, traced, args, args.device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    traced.save(args.out)
    print(f"Saved dual-branch TWIST+AnyAdapter JIT actor to {args.out}")
    print(
        f"branch_mode={args.branch_mode}, adapter_gain={args.adapter_gain}, "
        f"dynamics_delta_scale={args.dynamics_delta_scale}, "
        f"tracking_delta_scale={args.tracking_delta_scale}, "
        f"dynamics_branch_gain={args.dynamics_branch_gain}, "
        f"tracking_branch_gain={args.tracking_branch_gain}"
    )


if __name__ == "__main__":
    main()
