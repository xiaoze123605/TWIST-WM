import json
import sys
import unittest
from pathlib import Path

import numpy as np


DEPLOY_DIR = Path(__file__).resolve().parents[1] / "deploy_real"
sys.path.insert(0, str(DEPLOY_DIR))

from deploy_safety import (  # noqa: E402
    MIMIC_OBS_DIM,
    POLICY_OBS_DIM,
    TargetSafetyFilter,
    parse_mimic_msg,
)


class ParseMimicMessageTest(unittest.TestCase):
    def setUp(self):
        self.values = np.linspace(-1.0, 1.0, MIMIC_OBS_DIM, dtype=np.float32)

    def test_legacy_list(self):
        mimic, age, frame_id = parse_mimic_msg(json.dumps(self.values.tolist()))
        np.testing.assert_allclose(mimic, self.values)
        self.assertEqual(age, 0.0)
        self.assertEqual(frame_id, -1)

    def test_timestamped_dict(self):
        raw = json.dumps({
            "timestamp": 100.0,
            "frame_id": 42,
            "action_mimic": self.values.tolist(),
        }).encode()
        mimic, age, frame_id = parse_mimic_msg(raw, now=100.025)
        np.testing.assert_allclose(mimic, self.values)
        self.assertAlmostEqual(age, 0.025)
        self.assertEqual(frame_id, 42)

    def test_rejects_wrong_dimension_and_nan(self):
        with self.assertRaises(ValueError):
            parse_mimic_msg([0.0] * (MIMIC_OBS_DIM - 1))
        bad = self.values.copy()
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            parse_mimic_msg(bad)


class TargetSafetyFilterTest(unittest.TestCase):
    def setUp(self):
        self.default = np.zeros(3, dtype=np.float32)
        self.filter = TargetSafetyFilter(
            self.default,
            np.full(3, -1.0, dtype=np.float32),
            np.full(3, 1.0, dtype=np.float32),
            control_dt=0.02,
            max_target_rate=2.0,
            max_delta_per_step=0.10,
        )

    def test_applies_ramp_limits_and_finite_replacement(self):
        result = self.filter.apply(np.array([2.0, np.nan, -2.0]), action_ramp=0.5)
        np.testing.assert_allclose(result.target, [0.04, 0.0, -0.04])
        self.assertTrue(result.replaced_nonfinite)
        self.assertFalse(result.joint_limit_clipped)
        self.assertTrue(result.rate_limited)

    def test_joint_limit_is_final_invariant(self):
        for _ in range(100):
            result = self.filter.apply(np.array([10.0, 10.0, 10.0]), action_ramp=1.0)
        self.assertTrue(np.all(result.target <= 1.0))
        self.assertTrue(np.all(result.target >= -1.0))
        self.assertTrue(result.joint_limit_clipped)

    def test_policy_observation_contract(self):
        frame_dim = 31 + 3 + 2 + 3 * 23
        # One current 105-D frame plus ten 105-D history frames.  The 31-D
        # mimic reference is already included in frame_dim.
        self.assertEqual(POLICY_OBS_DIM, 11 * frame_dim)


if __name__ == "__main__":
    unittest.main()
