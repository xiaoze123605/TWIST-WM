"""
Export a trained DTERA actor (TwistDTERAActorCritic) to TorchScript.

DTERA = dual residual branches (dynamics + tracking-error history) with
demand/confidence/risk gating on top of the frozen TWIST student actor:

    a = a_base + adapter_gain * warmup_factor * gate * (dyn_gain*Δdyn + err_gain*Δerr)

Targets checkpoints of task ``g1_stu_anyadapter_dtera`` (config
G1MimicStuAnyAdapterDTERACfg / G1MimicStuAnyAdapterDTERACfgPPO). Unlike
export_twist_anyadapter_dual_jit.py (plain dual branch, 2635-D obs), this
checkpoint's policy observation is 3695-D:

    1155 base obs | 20*74 dynamics history | 20*53 tracking-error history

The training configuration below is hard-coded and cross-checked against the
checkpoint tensor shapes; a mismatch fails loudly at load time.

Gate / gain / scale knobs are plain attributes, not learned parameters, so
they can be overridden at export time without retraining (same semantics as
evaluate_dual_branch.py's DTERA gate-mode presets).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rsl_rl.modules.actor_critic_twist_dtera import TwistDTERAActorCritic


# --- DTERA training configuration (g1_mimic_distill_anyadapter_config.py) ---
DEFAULT_BASE_ACTOR_JIT_PATH = (
    "/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_stu_rl/"
    "0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt"
)
BASE_OBS_DIM = 1155
NUM_ACTIONS = 23
# AnyAdapterHistoryMixin: 1155 base + 20*74 dynamics history + 20*53 tracking history.
POLICY_OBS_DIM = BASE_OBS_DIM + 20 * 74 + 20 * 53  # 3695
# critic.0.weight in the checkpoint is (512, 1318).
NUM_CRITIC_OBS = 1318
HISTORY_LEN = 20
HISTORY_FRAME_DIM = 74
HIST_STATE_DIM = 51
LATENT_DIM = 32
# ANYADAPTER_STATE_INDICES from the config: 51 dims, indices 31..81.
WM_TARGET_INDICES = list(range(31, 82))
TRACKING_HISTORY_LEN = 20
TRACKING_ERROR_FRAME_DIM = 53
TRACKING_LATENT_DIM = 32
ADAPTER_HIDDEN_DIMS = (128, 128)
WORLD_MODEL_HIDDEN_DIMS = (256, 256)
ERROR_PREDICTOR_HIDDEN_DIMS = (128, 128)
RISK_PREDICTOR_HIDDEN_DIMS = (128, 128)
CRITIC_HIDDEN_DIMS = (512, 256, 128)
ACTIVATION = "elu"
USE_CONV_HISTORY = True
DEFAULT_REF_DOF_POS = [
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.4, 0.0, 1.2,
    0.0, -0.4, 0.0, 1.2,
]
BASE_SINGLE_OBS_DIM = 105
BASE_HISTORY_LEN = 10
INIT_NOISE_STD = 0.05
FIX_ACTION_STD = True
HISTORY_POLICY_GRAD_SCALE = 0.0
TRACKING_HISTORY_POLICY_GRAD_SCALE = 0.25
DYNAMICS_ACTION_DELTA_SCALE = 0.03
TRACKING_ACTION_DELTA_SCALE = 0.03
DYNAMICS_BRANCH_GAIN = 1.0
TRACKING_BRANCH_GAIN = 1.0
ADAPTER_GAIN = 1.0
WORLD_MODEL_ENSEMBLE_SIZE = 3
TRACKING_ERROR_SCALES = (0.35, 2.0, 1.0, 0.35, 0.08)
GATE_DEMAND_K = 1.0
GATE_CONFIDENCE_K = 1.0
GATE_RISK_K = 5.0
GATE_MODE = "demand_only"
CONFIDENCE_GATE_STRENGTH = 0.0
USE_INDEPENDENT_BRANCH_GATES = False
TRACKING_DEMAND_MODE = "legacy_exp"
TRACKING_DEMAND_LOW = 0.30
TRACKING_DEMAND_HIGH = 0.80
DYNAMICS_DEMAND_LOW = 0.10
DYNAMICS_DEMAND_HIGH = 0.50
DYNAMICS_GATE_SCALE = 1.0
TRACKING_GATE_SCALE = 1.0
DYNAMICS_CONFIDENCE_GATE_STRENGTH = 0.0
RESIDUAL_WARMUP_ITERATIONS = 1000
FREEZE_RESIDUAL_OUTPUT_BIAS = True
WM_VARIANCE_EMA_DECAY = 0.99
# The base actor is a frozen JIT and the critic is training-only; both are
# excluded from the strict load and re-verified separately.
TRAINING_ONLY_PREFIXES = ("base_actor.", "critic.")
# Obs dims the deploy server probes with (TWIST / AnyAdapter / heading / Any2Track).
REJECTED_PROBE_DIMS = (1155, 2635, 2637, 7001)


def default_output_path(ckpt_path: str, gate_mode: str, adapter_gain: float) -> str:
    ckpt = Path(ckpt_path)
    checkpoint_tag = ckpt.stem.replace("model_", "")
    run_dir = ckpt.parent
    suffix = f"dtera-{gate_mode}"
    if abs(adapter_gain - 1.0) > 1e-8:
        suffix += f"-gain{adapter_gain:g}"
    return str(run_dir / "traced" / f"{run_dir.name}-{checkpoint_tag}-{suffix}-jit.pt")


def build_actor(args: argparse.Namespace) -> TwistDTERAActorCritic:
    return TwistDTERAActorCritic(
        num_prop=POLICY_OBS_DIM,  # legacy positional, unused by this actor
        num_critic_obs=NUM_CRITIC_OBS,
        num_actions=NUM_ACTIONS,
        base_actor_jit_path=args.base_actor_jit_path,
        base_obs_dim=BASE_OBS_DIM,
        history_len=HISTORY_LEN,
        history_frame_dim=HISTORY_FRAME_DIM,
        hist_state_dim=HIST_STATE_DIM,
        latent_dim=LATENT_DIM,
        wm_target_indices=WM_TARGET_INDICES,
        adapter_hidden_dims=ADAPTER_HIDDEN_DIMS,
        critic_hidden_dims=CRITIC_HIDDEN_DIMS,
        world_model_hidden_dims=WORLD_MODEL_HIDDEN_DIMS,
        activation=ACTIVATION,
        init_noise_std=INIT_NOISE_STD,
        fix_action_std=FIX_ACTION_STD,
        adapter_gain=args.adapter_gain,
        dynamics_branch_gain=args.dynamics_branch_gain,
        tracking_branch_gain=args.tracking_branch_gain,
        adapter_branch_mode="full",
        dynamics_action_delta_scale=args.dynamics_delta_scale,
        tracking_action_delta_scale=args.tracking_delta_scale,
        default_ref_dof_pos=DEFAULT_REF_DOF_POS,
        use_tracking_error_adapter_input=True,
        history_policy_grad_scale=HISTORY_POLICY_GRAD_SCALE,
        adapter_context_dim=0,
        use_conv_history=USE_CONV_HISTORY,
        freeze_base=True,
        # --- DTERA-specific ---
        tracking_history_len=TRACKING_HISTORY_LEN,
        tracking_error_frame_dim=TRACKING_ERROR_FRAME_DIM,
        tracking_latent_dim=TRACKING_LATENT_DIM,
        tracking_history_policy_grad_scale=TRACKING_HISTORY_POLICY_GRAD_SCALE,
        world_model_ensemble_size=WORLD_MODEL_ENSEMBLE_SIZE,
        use_tracking_error_history=True,
        use_error_trend_predictor=True,
        use_world_model_ensemble=True,
        use_adaptive_residual_gate=True,
        error_predictor_hidden_dims=ERROR_PREDICTOR_HIDDEN_DIMS,
        risk_predictor_hidden_dims=RISK_PREDICTOR_HIDDEN_DIMS,
        tracking_error_scales=TRACKING_ERROR_SCALES,
        gate_demand_k=GATE_DEMAND_K,
        gate_confidence_k=GATE_CONFIDENCE_K,
        gate_risk_k=GATE_RISK_K,
        gate_mode=args.gate_mode,
        confidence_gate_strength=args.confidence_gate_strength,
        use_independent_branch_gates=args.independent_branch_gates,
        tracking_demand_mode=args.tracking_demand_mode,
        tracking_demand_low=args.tracking_demand_low,
        tracking_demand_high=args.tracking_demand_high,
        dynamics_demand_low=args.dynamics_demand_low,
        dynamics_demand_high=args.dynamics_demand_high,
        dynamics_gate_scale=args.dynamics_gate_scale,
        tracking_gate_scale=args.tracking_gate_scale,
        dynamics_confidence_gate_strength=args.dynamics_confidence_gate_strength,
        residual_warmup_iterations=RESIDUAL_WARMUP_ITERATIONS,
        freeze_residual_output_bias=FREEZE_RESIDUAL_OUTPUT_BIAS,
        wm_variance_ema_decay=WM_VARIANCE_EMA_DECAY,
        base_single_obs_dim=BASE_SINGLE_OBS_DIM,
        base_history_len=BASE_HISTORY_LEN,
        use_dual_branch_adapter=True,
    ).to(args.device)


def check_base_actor_matches(model: TwistDTERAActorCritic, state: dict) -> None:
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


def load_strict(model: TwistDTERAActorCritic, state: dict) -> dict:
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
        lines = ["Checkpoint does not match the DTERA actor architecture:"]
        if missing:
            lines.append(f"  missing keys in checkpoint ({len(missing)}):")
            lines += [f"    {k}" for k in missing[:10]]
        if unexpected:
            lines.append(f"  unexpected keys in checkpoint ({len(unexpected)}):")
            lines += [f"    {k}" for k in unexpected[:10]]
            if any(k.startswith("adapter.") for k in unexpected):
                lines.append(
                    "  Note: checkpoint has non-DTERA adapter keys; it may be a "
                    "plain dual-branch checkpoint. Use "
                    "export_twist_anyadapter_dual_jit.py for those instead."
                )
        raise RuntimeError("\n".join(lines))
    # The excluded base_actor./critic. keys are verified separately below.
    model.load_state_dict(filtered, strict=False)
    return filtered


def verify_traced(
    actor: torch.nn.Module,
    ac: TwistDTERAActorCritic,
    traced: torch.jit.ScriptModule,
    args: argparse.Namespace,
    device: str,
) -> None:
    torch.manual_seed(0)
    samples = [torch.zeros(1, POLICY_OBS_DIM, device=device)]
    samples += [torch.randn(args.num_verify_samples - 1, POLICY_OBS_DIM, device=device)]

    with torch.no_grad():
        eager = [ac.act_inference(s) for s in samples]
        base = [ac.base_action(s) for s in samples]
        diagnostics = [ac.action_diagnostics(s, base_action=b) for s, b in zip(samples, base)]
    jit = [traced(s) for s in samples]

    # 0) Input-dim guard: only exact 3695-column inputs may pass. The deploy
    #    server probes 1155/2635/2637/7001-D; none may be accepted.
    for dim in REJECTED_PROBE_DIMS:
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

    # 2) Amplitude bound: |a - a_base| <= gain * (dyn_gain*dyn_scale + err_gain*err_scale)
    #    (warmup factor and every gate live in [0, 1]).
    bound = args.adapter_gain * (
        args.dynamics_branch_gain * args.dynamics_delta_scale
        + args.tracking_branch_gain * args.tracking_delta_scale
    )
    max_delta = max((j - b).abs().max().item() for j, b in zip(jit, base))
    print(f"[verify] max |delta| = {max_delta:.4f} (bound = {bound:.4f})")
    if max_delta > bound + 1e-4:
        raise RuntimeError("Adapter delta exceeds the delta-scale bound; export aborted.")

    # 3) Live-branch / gate report over the samples.
    mean_dyn = sum(d["delta_dyn"].abs().mean().item() for d in diagnostics) / len(diagnostics)
    mean_err = sum(d["delta_err"].abs().mean().item() for d in diagnostics) / len(diagnostics)
    mean_gate = sum(d["gate"].mean().item() for d in diagnostics) / len(diagnostics)
    mean_applied = sum(d["applied_delta"].abs().mean().item() for d in diagnostics) / len(diagnostics)
    print(f"[verify] mean |delta_dyn| = {mean_dyn:.4f}, mean |delta_err| = {mean_err:.4f}, "
          f"mean gate = {mean_gate:.4f}, mean |applied delta| = {mean_applied:.4f}")
    iteration = int(ac.residual_training_iteration.item())
    factor = ac.residual_warmup_factor()
    print(f"[verify] residual_training_iteration = {iteration}, "
          f"residual warmup factor = {factor:.4f} (warmup iters = {ac.residual_warmup_iterations})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to the DTERA checkpoint .pt")
    parser.add_argument("--out", default=None, help="Output JIT actor path. Defaults to <run_dir>/traced/<run>-<ckpt>-dtera-<gate_mode>-jit.pt")
    parser.add_argument("--base_actor_jit_path", default=DEFAULT_BASE_ACTOR_JIT_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gate_mode", choices=["off", "demand_only", "demand_confidence", "full"],
                        default=GATE_MODE)
    parser.add_argument("--confidence_gate_strength", type=float, default=CONFIDENCE_GATE_STRENGTH)
    parser.add_argument("--independent_branch_gates", action="store_true",
                        default=USE_INDEPENDENT_BRANCH_GATES)
    parser.add_argument("--tracking_demand_mode", choices=["legacy_exp", "smoothstep"],
                        default=TRACKING_DEMAND_MODE)
    parser.add_argument("--tracking_demand_low", type=float, default=TRACKING_DEMAND_LOW)
    parser.add_argument("--tracking_demand_high", type=float, default=TRACKING_DEMAND_HIGH)
    parser.add_argument("--dynamics_demand_low", type=float, default=DYNAMICS_DEMAND_LOW)
    parser.add_argument("--dynamics_demand_high", type=float, default=DYNAMICS_DEMAND_HIGH)
    parser.add_argument("--dynamics_gate_scale", type=float, default=DYNAMICS_GATE_SCALE)
    parser.add_argument("--tracking_gate_scale", type=float, default=TRACKING_GATE_SCALE)
    parser.add_argument("--dynamics_confidence_gate_strength", type=float,
                        default=DYNAMICS_CONFIDENCE_GATE_STRENGTH)
    parser.add_argument("--adapter_gain", type=float, default=ADAPTER_GAIN)
    parser.add_argument("--dynamics_delta_scale", type=float, default=DYNAMICS_ACTION_DELTA_SCALE)
    parser.add_argument("--tracking_delta_scale", type=float, default=TRACKING_ACTION_DELTA_SCALE)
    parser.add_argument("--dynamics_branch_gain", type=float, default=DYNAMICS_BRANCH_GAIN)
    parser.add_argument("--tracking_branch_gain", type=float, default=TRACKING_BRANCH_GAIN)
    parser.add_argument("--num_verify_samples", type=int, default=8)
    args = parser.parse_args()

    if args.out is None:
        args.out = default_output_path(args.ckpt, args.gate_mode, args.adapter_gain)

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
    print(f"[load] gate_mode={model.gate_mode}, independent_branch_gates={model.use_independent_branch_gates}, "
          f"tracking_demand_mode={model.tracking_demand_mode}, "
          f"confidence_gate_strength={model.confidence_gate_strength:g}")

    class ActorOnly(torch.nn.Module):
        def __init__(self, ac):
            super().__init__()
            self.ac = ac

        def forward(self, obs):
            # Input-dim guard: reject every non-3695-D probe so the deploy
            # server cannot misclassify this policy (see REJECTED_PROBE_DIMS).
            obs = obs.reshape(-1, POLICY_OBS_DIM)
            return self.ac.act_inference(obs)

    actor = ActorOnly(model).to(args.device).eval()
    example = torch.zeros(1, POLICY_OBS_DIM, device=args.device)
    traced = torch.jit.trace(actor, example)

    verify_traced(actor, model, traced, args, args.device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    traced.save(args.out)
    print(f"Saved DTERA JIT actor to {args.out}")
    print(
        f"gate_mode={args.gate_mode}, adapter_gain={args.adapter_gain}, "
        f"dynamics_delta_scale={args.dynamics_delta_scale}, "
        f"tracking_delta_scale={args.tracking_delta_scale}, "
        f"dynamics_branch_gain={args.dynamics_branch_gain}, "
        f"tracking_branch_gain={args.tracking_branch_gain}"
    )


if __name__ == "__main__":
    main()
