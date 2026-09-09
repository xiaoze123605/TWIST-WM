from __future__ import annotations

import ast
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from motion_world_model.baselines import _interpolate_holds, filtering_interpolation, last_frame_hold, linear_extrapolation
from motion_world_model.dataset import CorruptionConfig, MotionWeightedSampler, MotionWindowDataset, _compatible_numpy_module, corrupt_reference, motion_to_reference_50hz
from motion_world_model.model import MotionGRU, motion_prediction_loss
from motion_world_model.utils import FUTURE_LENGTH, HISTORY_LENGTH, REFERENCE_DIM, motion_group_id, stable_split


REPOSITORY = Path(__file__).resolve().parents[1]


class MotionWorldModelTests(unittest.TestCase):
    def test_motion_file_override_is_applied_only_when_explicit(self) -> None:
        source = (REPOSITORY / "legged_gym/legged_gym/gym_utils/helpers.py").read_text()
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "update_cfg_from_args")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "helpers.py", "exec"), namespace)
        env = SimpleNamespace(
            motion=SimpleNamespace(motion_file="configured.yaml"),
            env=SimpleNamespace(teleop_mode=False, num_envs=1, record_video=False),
            terrain=SimpleNamespace(num_rows=1, num_cols=1),
            domain_rand=SimpleNamespace(domain_rand_general=True),
            seed=0,
        )
        args = SimpleNamespace(
            motion_file="override.pkl", teleop_mode=False, num_envs=None, seed=None,
            rows=None, cols=None, record_video=False, no_rand=False,
        )
        namespace["update_cfg_from_args"](env, None, args)
        self.assertEqual(env.motion.motion_file, "override.pkl")
        args.motion_file = None
        namespace["update_cfg_from_args"](env, None, args)
        self.assertEqual(env.motion.motion_file, "override.pkl")

    def test_segmented_motion_group_never_leaks(self) -> None:
        first = motion_group_id("subject/walk_seg01.pkl")
        second = motion_group_id("subject/walk_seg42.pkl")
        self.assertEqual(first, second)
        self.assertEqual(stable_split(first, 42, 0.8, 0.1), stable_split(second, 42, 0.8, 0.1))

    def test_resampling_builds_expected_31d_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linear.pkl"
            fps = 25.0
            frames = 51
            times = np.arange(frames) / fps
            root_pos = np.zeros((frames, 3), dtype=np.float32)
            root_pos[:, 0] = times
            root_pos[:, 2] = 0.8
            root_rot = np.zeros((frames, 4), dtype=np.float32)
            root_rot[:, 3] = 1.0
            dof_pos = np.repeat(times[:, None], 23, axis=1).astype(np.float32)
            with path.open("wb") as handle:
                pickle.dump({"fps": fps, "root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof_pos}, handle)
            reference = motion_to_reference_50hz(path)
            self.assertEqual(reference.shape, (101, REFERENCE_DIM))
            np.testing.assert_allclose(reference[:, 0], 0.8, atol=1e-6)
            np.testing.assert_allclose(reference[30:70, 4], 1.0, atol=1e-5)
            np.testing.assert_allclose(reference[1, 8:], 0.02, atol=1e-6)

    def test_numpy_2_pickle_module_names_are_mapped(self) -> None:
        self.assertEqual(
            _compatible_numpy_module("numpy._core.multiarray"), "numpy.core.multiarray"
        )
        self.assertEqual(_compatible_numpy_module("builtins"), "builtins")

    def test_window_alignment_and_motion_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            sequence_dir = cache / "sequences/train"
            sequence_dir.mkdir(parents=True)
            sequence = np.arange(50 * REFERENCE_DIM, dtype=np.float32).reshape(50, REFERENCE_DIM)
            np.save(sequence_dir / "one.npy", sequence)
            manifest = {"split": "train", "motions": [{
                "motion_id": "one.pkl", "cache_path": "sequences/train/one.npy",
                "frames": 50, "windows": 16,
            }]}
            (cache / "train_manifest.json").write_text(json.dumps(manifest))
            dataset = MotionWindowDataset(cache, "train", CorruptionConfig().only("clean"))
            item = dataset[0]
            self.assertEqual(tuple(item["corrupted_history"].shape), (HISTORY_LENGTH, REFERENCE_DIM))
            self.assertEqual(tuple(item["target"].shape), (FUTURE_LENGTH + 1, REFERENCE_DIM))
            torch.testing.assert_close(item["target"][0], item["clean_history"][-1])
            torch.testing.assert_close(item["target"][-1], torch.from_numpy(sequence[34]))

    def test_motion_weighted_sampler_is_seeded_and_in_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            sequence_dir = cache / "sequences/train"
            sequence_dir.mkdir(parents=True)
            motions = []
            for index, weight in enumerate((1.0, 20.0)):
                sequence = np.zeros((40, REFERENCE_DIM), dtype=np.float32)
                np.save(sequence_dir / f"{index}.npy", sequence)
                motions.append({
                    "motion_id": f"{index}.pkl", "cache_path": f"sequences/train/{index}.npy",
                    "frames": 40, "windows": 6, "weight": weight,
                })
            (cache / "train_manifest.json").write_text(json.dumps({"split": "train", "motions": motions}))
            dataset = MotionWindowDataset(cache, "train", CorruptionConfig().only("clean"))
            first = list(MotionWeightedSampler(dataset, 200, seed=9))
            second = list(MotionWeightedSampler(dataset, 200, seed=9))
            self.assertEqual(first, second)
            self.assertTrue(all(0 <= index < len(dataset) for index in first))
            self.assertGreater(sum(index >= 6 for index in first), 150)

    def test_corruption_is_seeded_and_finite(self) -> None:
        clean = np.linspace(0, 1, HISTORY_LENGTH * REFERENCE_DIM, dtype=np.float32).reshape(HISTORY_LENGTH, REFERENCE_DIM)
        first = corrupt_reference(clean, np.random.default_rng(7), CorruptionConfig())
        second = corrupt_reference(clean, np.random.default_rng(7), CorruptionConfig())
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertFalse(np.array_equal(first, clean))

    def test_model_shape_and_both_losses_backpropagate(self) -> None:
        model = MotionGRU(hidden_dim=16)
        history = torch.randn(4, HISTORY_LENGTH, REFERENCE_DIM)
        target = torch.randn(4, FUTURE_LENGTH + 1, REFERENCE_DIM)
        prediction = model(history)
        self.assertEqual(tuple(prediction.shape), tuple(target.shape))
        loss, parts = motion_prediction_loss(prediction, target)
        loss.backward()
        self.assertGreater(parts["current"].item(), 0)
        self.assertGreater(parts["future"].item(), 0)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_baselines_have_common_shape_and_linear_is_exact(self) -> None:
        slope = torch.linspace(-0.2, 0.2, REFERENCE_DIM)
        time = torch.arange(HISTORY_LENGTH, dtype=torch.float32)
        history = (time[:, None] * slope[None, :])[None, ...]
        expected_time = torch.arange(HISTORY_LENGTH - 1, HISTORY_LENGTH + FUTURE_LENGTH, dtype=torch.float32)
        expected = (expected_time[:, None] * slope[None, :])[None, ...]
        self.assertEqual(last_frame_hold(history).shape, expected.shape)
        torch.testing.assert_close(linear_extrapolation(history), expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(filtering_interpolation(history).shape, expected.shape)

    def test_held_run_interpolation_is_vectorized_and_causal(self) -> None:
        clean = torch.arange(HISTORY_LENGTH, dtype=torch.float32).view(1, -1, 1)
        held = clean.clone()
        held[:, 5:8] = held[:, 4:5]
        held[:, -2:] = held[:, -3:-2]
        repaired = _interpolate_holds(held)
        torch.testing.assert_close(repaired[:, 5:8], clean[:, 5:8])
        torch.testing.assert_close(repaired[:, -2:], held[:, -2:])


if __name__ == "__main__":
    unittest.main()
