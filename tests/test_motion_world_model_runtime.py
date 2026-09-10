from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from motion_world_model.model import MotionGRU
from motion_world_model.runtime import (
    MotionReferenceRefiner,
    RuntimeCorruptionConfig,
    RuntimeReferenceCorruptor,
    reinsert_wrist_roll,
    remove_wrist_roll,
)
from motion_world_model.utils import HISTORY_LENGTH, REFERENCE_DIM


class MotionWorldModelRuntimeTests(unittest.TestCase):
    def test_wrist_round_trip_keeps_original_values(self) -> None:
        reference = np.arange(33, dtype=np.float32)
        reduced, wrists = remove_wrist_roll(reference)
        self.assertEqual(reduced.shape, (31,))
        np.testing.assert_array_equal(wrists, reference[[27, 32]])
        np.testing.assert_array_equal(reinsert_wrist_roll(reduced, wrists), reference)

    def test_runtime_corruption_is_reproducible(self) -> None:
        first = RuntimeReferenceCorruptor.from_preset("formal", seed=42)
        second = RuntimeReferenceCorruptor.from_preset("formal", seed=42)
        clean = [np.full(REFERENCE_DIM, index / 10, dtype=np.float32) for index in range(50)]
        first_sequence = np.stack([first.corrupt(frame) for frame in clean])
        second_sequence = np.stack([second.corrupt(frame) for frame in clean])
        np.testing.assert_array_equal(first_sequence, second_sequence)
        self.assertTrue(np.all(np.isfinite(first_sequence)))

    def test_runtime_corruption_keeps_yaw_continuous_across_wrap(self) -> None:
        config = RuntimeCorruptionConfig(
            noise_std=0.0,
            hold_probability=0.0,
            delay_max_frames=0,
            lowpass_probability=0.0,
        )
        corruptor = RuntimeReferenceCorruptor(config, seed=1)
        yaws = (3.12, -3.12, -3.10)
        outputs = []
        for yaw in yaws:
            frame = np.zeros(REFERENCE_DIM, dtype=np.float32)
            frame[3] = yaw
            outputs.append(corruptor.corrupt(frame)[3])
        self.assertTrue(np.all(np.abs(np.diff(outputs)) < 0.1))
        self.assertGreater(outputs[-1], np.pi)

    def test_refiner_does_not_pad_warmup_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best.pt"
            model = MotionGRU(hidden_dim=8)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0},
                    "normalization_mean": np.zeros(REFERENCE_DIM, dtype=np.float32),
                    "normalization_std": np.ones(REFERENCE_DIM, dtype=np.float32),
                },
                checkpoint_path,
            )
            refiner = MotionReferenceRefiner(checkpoint_path, device="cpu")
            frame = np.linspace(-0.2, 0.2, REFERENCE_DIM, dtype=np.float32)
            for index in range(HISTORY_LENGTH - 1):
                output = refiner.refine(frame + index * 0.01)
                np.testing.assert_allclose(output, frame + index * 0.01, atol=1e-6)
                self.assertFalse(refiner.ready)
            final = refiner.refine(frame)
            self.assertTrue(refiner.ready)
            self.assertEqual(final.shape, (REFERENCE_DIM,))
            self.assertTrue(np.all(np.isfinite(final)))


if __name__ == "__main__":
    unittest.main()
