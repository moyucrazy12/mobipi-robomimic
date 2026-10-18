"""Small CPU-only tests; no checkpoints, real datasets or simulators are loaded."""
from collections import deque
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from robomimic.scripts import evaluate_rby1_cartesian_checkpoints as evaluator
from robomimic.scripts import report_rby1_cartesian_evaluation as reporter
from robomimic.utils.rby1_evaluation import (
    FIELDS, RGB_KEYS, LOWDIM_KEYS, SUCCESS_THRESHOLD_RAD, atomic_json,
    extract_losses, read_completed_rows, valid_completed_row, write_csv,
    fingerprint,
)


def trial(epoch, demo, seed, success, **updates):
    row = dict.fromkeys(FIELDS, "")
    row.update(checkpoint_epoch=epoch, checkpoint_path=f"/models/model_epoch_{epoch}.pth",
               checkpoint_size=100, checkpoint_mtime_ns=1000, demo_key=demo, split="valid",
               seed=seed, completed=True, success=success, success_timestep=4 if success else None,
               max_steps=16, steps_executed=16, initial_hinge_angle=-1.5,
               best_hinge_angle=-0.01 if success else -0.4,
               final_hinge_angle=-0.01 if success else -0.4,
               failure_classification="success" if success else "partial_closure")
    row.update(updates)
    return row


class EvaluationUtilities(unittest.TestCase):
    def test_atomic_resume_rows_and_invalid_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.csv"
            valid = trial(3, "demo_10", 0, True)
            invalid = trial(3, "demo_19", 0, False, completed=False)
            write_csv(path, [valid, invalid], FIELDS)
            rows, count = read_completed_rows(path)
            self.assertEqual((len(rows), count), (1, 1))
            self.assertTrue(valid_completed_row(rows[0]))
            wrong = dict(rows[0], final_hinge_angle=-1.5)
            self.assertFalse(valid_completed_row(wrong))
            atomic_json(Path(directory) / "metadata.json", {"protocol": {"max_steps": 16}})
            self.assertFalse(list(Path(directory).glob("*.tmp")))
            write_csv(path, [valid, valid], FIELDS)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_completed_rows(path)

    def test_loss_parser_partial_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "logs/log.txt"
            log.parent.mkdir()
            log.write_text('progress\rTrain Epoch 1\n{"Loss": 0.3}\n'
                           'Validation Epoch 1\n{"Loss": 0.2}\n'
                           'Train Epoch 2\n{"Loss": 0.1}\n'
                           'Validation Epoch 2\n{"Loss":')
            losses = extract_losses(directory)
            self.assertEqual(losses, [dict(epoch=1, train_loss=0.3, validation_loss=0.2),
                                     dict(epoch=2, train_loss=0.1, validation_loss=None)])

    def test_report_paired_comparisons_do_not_use_unmatched_trials(self):
        rows = [trial(3, "demo_10", 0, True), trial(3, "demo_19", 0, False),
                trial(5, "demo_10", 0, False), trial(5, "demo_47", 0, True)]
        summary, _, _, _, paired = reporter.summarize(rows)
        self.assertEqual([row["success_rate"] for row in summary], [0.5, 0.5])
        self.assertEqual(paired[0]["matched_rollouts"], 1)
        self.assertEqual(paired[0]["regressions"], 1)
        self.assertEqual(paired[0]["success_rate_delta_b_minus_a"], -1)

    def test_report_cli_synthetic_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            write_csv(evaluation / "rollouts.csv", [trial(3, "demo_10", 0, True),
                       trial(5, "demo_10", 0, False)], FIELDS)
            reporter.main(["--input", str(evaluation), "--output-dir", str(root / "report")])
            report = (root / "report/report.md").read_text()
            self.assertIn("does not assert that training", report)
            self.assertIn("Paired checkpoint comparisons", report)
            with (root / "report/checkpoint_summary.csv").open() as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual([row["successes"] for row in summary], ["1", "0"])
            self.assertTrue((root / "report/training_losses.csv").exists())

    def test_deterministic_mask_selection(self):
        class Mask:
            def asstr(self):
                return np.array(["demo_47", "demo_10", "demo_111", "demo_73", "demo_19"])
        dataset = {"mask/valid": Mask()}
        self.assertEqual(evaluator.select_demos(dataset, "valid", None, 3),
                         ["demo_10", "demo_47", "demo_111"])
        with self.assertRaises(ValueError):
            evaluator.select_demos(dataset, "valid", ["demo_0"], 3)

    def test_resume_skips_completed_without_loading_a_checkpoint(self):
        import h5py
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "synthetic.hdf5"
            run = root / "run"
            (run / "models").mkdir(parents=True)
            output = root / "evaluation"
            with h5py.File(dataset, "w") as f:
                f.create_dataset("mask/valid", data=np.array([b"demo_10"]))
                f.create_group("data").attrs["env_args"] = json.dumps({"env_name": "mock"})
            command = ["--run-dir", str(run), "--epochs", "3", "--dataset", str(dataset),
                       "--num-demos", "1", "--seeds", "0", "--max-steps", "16",
                       "--output-dir", str(output), "--skip-missing-checkpoints"]
            evaluator.main(command)  # Missing checkpoint: only CPU metadata work.
            checkpoint = run / "models/model_epoch_3.pth"
            checkpoint.write_text("This is a stat-only fixture, not a checkpoint.")
            identity = fingerprint(checkpoint)
            row = trial(3, "demo_10", 0, True, checkpoint_path=identity["path"],
                        checkpoint_size=identity["size"], checkpoint_mtime_ns=identity["mtime_ns"])
            write_csv(output / "rollouts.csv", [row], FIELDS)
            # A checkpoint-load attempt would fail on the deliberately invalid fixture.
            evaluator.main(command + ["--resume"])
            rows, _ = read_completed_rows(output / "rollouts.csv")
            self.assertEqual(len(rows), 1)
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                altered = command.copy()
                altered[altered.index("16")] = "17"
                evaluator.main(altered + ["--resume"])
            checkpoint.write_text("changed fixture")
            with self.assertRaisesRegex(ValueError, "Checkpoint changed"):
                evaluator.main(command + ["--resume"])


class MockedRollout(unittest.TestCase):
    def run_loop(self, hinges, fail_step=None):
        class Environment:
            def __init__(self):
                self.sim = SimpleNamespace(data=SimpleNamespace(qpos=np.array([-1.5]), qvel=np.zeros(1)))
                self.step_count = 0
                self.tcp = np.eye(4)
            def step(self, action):
                if self.step_count == fail_step:
                    raise RuntimeError("mock controller failure")
                self.sim.data.qpos[0] = hinges[self.step_count]
                self.step_count += 1
                self.tcp[0, 3] = self.step_count * 0.001
        env = Environment()
        class Adapter:
            tcp_site = "tcp"
            def site_transform(self, site):
                return env.tcp.copy()
            def target_world(self, action):
                result = np.eye(4)
                result[0, 3] = (env.step_count + 1) * 0.001
                return result
            def __call__(self, action):
                return action
        class Policy:
            def __init__(self):
                self.policy = SimpleNamespace(action_queue=deque())
                self.plans = 0
            def __call__(self, observations):
                assert observations[LOWDIM_KEYS[0]].shape == (2, 3)
                if not self.policy.action_queue:
                    self.policy.action_queue.extend([np.zeros(9) for _ in range(8)])
                    self.plans += 1
                return self.policy.action_queue.popleft()
        policy = Policy()
        def read():
            return {**{key: np.zeros((3, 4, 4)) for key in RGB_KEYS},
                    LOWDIM_KEYS[0]: np.zeros(3), LOWDIM_KEYS[1]: np.array([0, 0, 0, 1])}
        fake_torch = SimpleNamespace(no_grad=nullcontext)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            row = evaluator.execute_loop(policy, env, Adapter(), read, 0, len(hinges))
        return row, policy

    def test_native_eight_action_queue_and_success(self):
        row, policy = self.run_loop([-1.5] * 3 + [-0.01] * 13)
        self.assertTrue(row["success"])
        self.assertEqual(row["success_timestep"], 4)
        self.assertEqual((row["number_action_chunks"], policy.plans), (2, 2))
        self.assertEqual(row["mean_tcp_tracking_error_mm"], 0)
        self.assertEqual(row["steps_executed"], 16)

    def test_reopening_is_not_final_success(self):
        row, _ = self.run_loop([-1.5, SUCCESS_THRESHOLD_RAD, -0.2, -0.3])
        self.assertFalse(row["success"])
        self.assertEqual(row["success_timestep"], 2)
        self.assertEqual(row["failure_classification"], "reopened")

    def test_controller_error_is_persistable(self):
        row, _ = self.run_loop([-1.5] * 8, fail_step=2)
        self.assertEqual(row["failure_classification"], "controller_error")
        self.assertEqual(row["steps_executed"], 2)
        combined = trial(3, "demo_10", 0, False)
        combined.update(row)
        self.assertTrue(valid_completed_row(combined))


if __name__ == "__main__":
    unittest.main()
