from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "deploy_real"))

from twist_anyadapter_runtime import AnyAdapterRuntime, AnyAdapterRuntimeConfig


BASE_OBS_DIM = 1155
NUM_ACTIONS = 23
HISTORY_LEN = 20
STATE_INDICES = list(range(31, 82))
DTERA_OBS_DIM = BASE_OBS_DIM + HISTORY_LEN * 74 + HISTORY_LEN * 53


class FixedContractPolicy(torch.nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.obs_dim = int(obs_dim)

    def forward(self, observations):
        observations = observations.reshape(-1, self.obs_dim)
        return observations[:, :NUM_ACTIONS]


class AnyAdapterRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.policy_path = str(Path(cls.temp_dir.name) / "dtera.pt")
        policy = torch.jit.trace(
            FixedContractPolicy(DTERA_OBS_DIM).eval(),
            torch.zeros(1, DTERA_OBS_DIM),
        )
        policy.save(cls.policy_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def make_runtime(self):
        return AnyAdapterRuntime(AnyAdapterRuntimeConfig(
            base_obs_dim=BASE_OBS_DIM,
            num_actions=NUM_ACTIONS,
            history_len=HISTORY_LEN,
            state_indices=STATE_INDICES,
            policy_path=self.policy_path,
            device="cpu",
            fill_history_on_first_observation=True,
            tracking_error_history_len=HISTORY_LEN,
            control_dt=0.02,
        ))

    @staticmethod
    def tracking_inputs(reference_position=0.10):
        reference = np.zeros(8 + NUM_ACTIONS, dtype=np.float32)
        reference[1:3] = [0.20, -0.10]
        reference[4:8] = [0.50, -0.25, 0.10, 0.30]
        reference[8:] = reference_position
        return dict(
            tracking_reference=reference,
            dof_pos=np.full(NUM_ACTIONS, 0.05, dtype=np.float32),
            dof_vel=np.full(NUM_ACTIONS, 0.20, dtype=np.float32),
            root_linear_velocity=np.asarray([0.20, -0.10, 0.05], dtype=np.float32),
            root_yaw_velocity=0.10,
            roll_pitch=np.asarray([0.05, -0.02], dtype=np.float32),
        )

    def test_dtera_observation_contract_and_first_frame_fill(self):
        runtime = self.make_runtime()
        action = runtime.act(
            np.zeros(BASE_OBS_DIM, dtype=np.float32),
            **self.tracking_inputs(),
        )
        self.assertEqual(runtime.policy_obs_dim, DTERA_OBS_DIM)
        self.assertEqual(action.shape, (NUM_ACTIONS,))
        self.assertTrue(np.isfinite(action).all())
        self.assertTrue(np.allclose(
            runtime.tracking_error_history,
            runtime.tracking_error_history[-1],
        ))
        frame = runtime.tracking_error_history[-1]
        self.assertTrue(np.allclose(frame[:NUM_ACTIONS], 0.05))
        self.assertTrue(np.allclose(frame[NUM_ACTIONS:2 * NUM_ACTIONS], 0.0))
        self.assertTrue(np.allclose(frame[46:49], [0.30, -0.15, 0.05]))
        self.assertAlmostEqual(float(frame[49]), 0.20, places=6)
        self.assertTrue(np.allclose(frame[50:52], [0.15, -0.08]))

    def test_reference_velocity_filter_and_reset(self):
        runtime = self.make_runtime()
        base = np.zeros(BASE_OBS_DIM, dtype=np.float32)
        runtime.act(base, **self.tracking_inputs(0.10))
        runtime.act(base, **self.tracking_inputs(0.12))
        # raw reference velocity is 1.0; alpha=.5 blends it with the initial
        # measured/reference velocity of .2, producing .6 and dq error .4.
        self.assertTrue(np.allclose(
            runtime.tracking_error_history[-1][NUM_ACTIONS:2 * NUM_ACTIONS],
            0.40,
            atol=1e-6,
        ))
        runtime.reset()
        self.assertFalse(runtime._tracking_reference_initialized)
        self.assertFalse(runtime._tracking_history_initialized)
        self.assertEqual(float(np.abs(runtime.tracking_error_history).max()), 0.0)

    def test_dtera_requires_physical_tracking_inputs(self):
        runtime = self.make_runtime()
        with self.assertRaisesRegex(ValueError, "DTERA policy requires"):
            runtime.act(np.zeros(BASE_OBS_DIM, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
