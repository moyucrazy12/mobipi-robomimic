"""Path-only tests: no trainer, simulator, HDF5 or checkpoint is imported/loaded."""
import argparse
import ast
import contextlib
import copy
import datetime
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from robomimic.utils import rby1_paths as door
from robomimic.scripts import train_rby1

RM = Path(__file__).resolve().parents[1]
MOBIPI_ROOT = RM.parents[1]
PROJECT = MOBIPI_ROOT.parent
COLLECT = MOBIPI_ROOT / "mobipi/collect_demos"
spec = importlib.util.spec_from_file_location("faucet_paths_test", COLLECT / "experiment_paths.py")
faucet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(faucet)


def parser_only(path):
    """Execute only existing argparse construction statements, without module imports."""
    tree = ast.parse(path.read_text())
    candidates = [n.body for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("main", "parser")]
    candidates += [tree.body]
    candidates += [n.body for n in tree.body if isinstance(n, ast.If)]
    for body in candidates:
        start = next((i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                      and isinstance(n.value, ast.Call)
                      and isinstance(n.value.func, ast.Attribute)
                      and n.value.func.attr == "ArgumentParser"), None)
        if start is None:
            continue
        nodes = []
        for n in body[start:]:
            if isinstance(n, ast.Return) or any(isinstance(x, ast.Call)
                                               and isinstance(x.func, ast.Attribute)
                                               and x.func.attr == "parse_args" for x in ast.walk(n)):
                break
            nodes.append(n)
        paths = faucet if path.parent == COLLECT else door
        scope = dict(argparse=argparse, Path=Path, __file__=str(path),
                     artifact_path=paths.artifact_path, new_output=paths.new_output,
                     historical_path=faucet.historical_path, ROBOMIMIC_ROOT=RM,
                     DEFAULT_RUN=faucet.historical_path("train", "rby1_faucet_relative_cartesian_gripper_100d_15epochs", "20261027135336"),
                     TASKS={"TurnOnSinkFaucet": object()})
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
        return scope.get("parser", scope.get("result"))
    raise AssertionError(f"No parser found in {path}")


class PathsTests(unittest.TestCase):
    def test_root_from_both_repositories_and_other_cwd(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RBY1_EXPERIMENT_ROOT", None)
            try:
                for cwd in (MOBIPI_ROOT, RM, Path(temporary)):
                    os.chdir(cwd)
                    self.assertEqual(door.experiment_root(), PROJECT / "experiment_results")
                    self.assertEqual(faucet.experiment_root(), PROJECT / "experiment_results")
                    with patch.dict(os.environ, {"RBY1_EXPERIMENT_ROOT": str(Path(temporary) / "artifacts")}):
                        self.assertEqual(door.experiment_root(), Path(temporary) / "artifacts")
                        self.assertEqual(faucet.experiment_root(), Path(temporary) / "artifacts")
                    with patch.dict(os.environ, {"RBY1_EXPERIMENT_ROOT": "relative_artifacts"}):
                        self.assertEqual(door.experiment_root(), PROJECT / "relative_artifacts")
                        self.assertEqual(faucet.experiment_root(), PROJECT / "relative_artifacts")
            finally:
                os.chdir(previous)

    def test_training_path_overrides_preserve_all_other_values(self):
        config = {"experiment": {"name": "rby1_faucet_example"}, "train": {
            "output_dir": "old", "data": [{"path": "${RBY1_EXPERIMENT_ROOT}/TurnOnFaucet/demos/train.hdf5", "do_eval": False}],
            "action_keys": ["actions_eef_cmd", "actions_gripper"], "batch_size": 16},
            "algo": {"relative_actions": {"enabled": True, "arm": "right"}, "horizon": [2, 8, 16]}}
        before = copy.deepcopy(config)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"RBY1_EXPERIMENT_ROOT": temporary}):
            resolved = door.resolve_training_paths(config)
            self.assertEqual(resolved["train"]["output_dir"], str(Path(temporary) / "TurnOnFaucet/train"))
            self.assertEqual(resolved["train"]["data"][0]["path"], str(Path(temporary) / "TurnOnFaucet/demos/train.hdf5"))
            explicit = door.resolve_training_paths(config, Path(temporary) / "custom_train", Path(temporary) / "custom.hdf5")
            self.assertEqual(explicit["train"]["output_dir"], str(Path(temporary) / "custom_train"))
            self.assertEqual(explicit["train"]["data"][0]["path"], str(Path(temporary) / "custom.hdf5"))
            for result in (resolved, explicit, faucet.resolve_training_paths(config, dataset=Path(temporary) / "prepared.hdf5")):
                result["train"]["output_dir"] = config["train"]["output_dir"]
                result["train"]["data"][0]["path"] = config["train"]["data"][0]["path"]
                self.assertEqual(result, config)
        self.assertEqual(config, before)

    def test_native_training_directory_generation_only_in_temporary_root(self):
        source = RM / "robomimic/utils/train_utils.py"
        node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "get_exp_dir")
        scope = dict(os=os, time=time, datetime=datetime, shutil=shutil, robomimic=SimpleNamespace(__path__=[str(RM / "robomimic")]))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"RBY1_EXPERIMENT_ROOT": temporary}):
            settings = door.resolve_training_paths({"experiment": {"name": "rby1_close_single_door_test"},
                                                   "train": {"data": [{"path": "input.hdf5"}]}})
            config = SimpleNamespace(train=SimpleNamespace(output_dir=settings["train"]["output_dir"]),
                                     experiment=SimpleNamespace(name=settings["experiment"]["name"], save=SimpleNamespace(enabled=True)))
            paths = list(map(Path, scope["get_exp_dir"](config)))
            self.assertEqual({p.name for p in paths}, {"models", "logs", "videos", "vis"})
            for p in paths:
                self.assertTrue(p.is_dir())
                self.assertTrue(p.is_relative_to(Path(temporary) / "CloseSingleDoor/train/rby1_close_single_door_test"))
                self.assertRegex(p.parent.name, r"^\d{14}$")

    def test_native_relative_json_fallback_is_package_relative(self):
        source = RM / "robomimic/utils/train_utils.py"
        node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "get_exp_dir")
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            package = project / "mobipi/external/robomimic/robomimic"
            package.mkdir(parents=True)
            scope = dict(os=os, time=time, datetime=datetime, shutil=shutil, robomimic=SimpleNamespace(__path__=[str(package)]))
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
            config = SimpleNamespace(train=SimpleNamespace(output_dir="../../../../experiment_results/CloseSingleDoor/train"),
                                     experiment=SimpleNamespace(name="rby1_close_single_door_test", save=SimpleNamespace(enabled=True)))
            for path in scope["get_exp_dir"](config):
                self.assertTrue(Path(path).resolve().is_relative_to(project / "experiment_results/CloseSingleDoor/train"))

    def test_real_launcher_print_config_from_both_working_directories(self):
        # This launcher resolves the custom door/faucet paths. Drawer configs
        # continue to use the native trainer and must not be passed to it.
        configs = list((RM / "robomimic/exps").glob("rby1_close_single_door*.json")) + list(COLLECT.glob("faucet_relative_cartesian_dp*.json"))
        self.assertTrue(configs)
        with tempfile.TemporaryDirectory() as temporary:
            env = dict(os.environ, RBY1_EXPERIMENT_ROOT=temporary, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(RM))
            for config in configs:
                raw = json.loads(config.read_text())
                values = []
                for cwd in (MOBIPI_ROOT, RM):
                    result = subprocess.run([sys.executable, "-m", "robomimic.scripts.train_rby1", "--config", str(config), "--print-config"],
                                            cwd=cwd, env=env, text=True, capture_output=True, check=True)
                    value = json.loads(result.stdout)
                    task = "TurnOnFaucet" if config.parent == COLLECT else "CloseSingleDoor"
                    self.assertEqual(value["train"]["output_dir"], str(Path(temporary) / task / "train"))
                    compare = copy.deepcopy(value)
                    compare["train"]["output_dir"] = raw["train"]["output_dir"]
                    compare["train"]["data"][0]["path"] = raw["train"]["data"][0]["path"]
                    self.assertEqual(compare, raw)
                    values.append(value)
                self.assertEqual(values[0], values[1])
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_launcher_delegates_native_main_without_training_in_test(self):
        config = {"algo_name": "diffusion_policy", "experiment": {"name": "rby1_faucet_test"},
                  "train": {"data": [{"path": "input.hdf5", "do_eval": False}], "batch_size": 16}}
        calls = []
        backend = ModuleType("robomimic.scripts.train")
        backend.main = lambda args: calls.append((args, json.loads(Path(args.config).read_text())))
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"RBY1_EXPERIMENT_ROOT": temporary}):
            path = Path(temporary) / "input.json"
            path.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):
                printed = train_rby1.main(["--config", str(path), "--print-config"])
            self.assertEqual(calls, [])
            with patch.dict(sys.modules, {"robomimic.scripts.train": backend}):
                train_rby1.main(["--config", str(path), "--output-dir", str(Path(temporary) / "override"),
                                 "--dataset", str(Path(temporary) / "explicit.hdf5")])
            args, settings = calls[0]
            self.assertFalse(args.debug)
            self.assertIsNone(args.dataset)
            self.assertEqual(settings["train"]["batch_size"], 16)
            self.assertEqual(settings["train"]["output_dir"], str(Path(temporary) / "override"))
            self.assertEqual(settings["train"]["data"][0]["path"], str(Path(temporary) / "explicit.hdf5"))
            self.assertFalse(settings["train"]["data"][0]["do_eval"])
            self.assertFalse(Path(args.config).exists())
            self.assertEqual(json.loads(path.read_text()), config)

    def test_historical_run_does_not_change_with_output_override(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(faucet, "PROJECT_ROOT", Path(temporary) / "workspace"), \
                patch.dict(os.environ, {"RBY1_EXPERIMENT_ROOT": str(Path(temporary) / "outputs")}):
            run = faucet.historical_path("train", "rby1_faucet_relative_cartesian_gripper_100d_15epochs", "20261027135336")
            self.assertEqual(run, faucet.PROJECT_ROOT / "experiment_results/TurnOnFaucet/train/rby1_faucet_relative_cartesian_gripper_100d_15epochs/20261027135336")
            self.assertTrue(faucet.new_output("rollouts", "evaluation").is_relative_to(Path(temporary) / "outputs"))

    def test_manifest_metadata_resolution_preserves_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(door, "PROJECT_ROOT", Path(temporary)):
            root = Path(temporary) / "experiment_results"
            root.mkdir()
            item = dict(original_path="mobipi/old_run", new_path="experiment_results/CloseSingleDoor/train/same_run")
            (root / "migration_manifest.json").write_text(json.dumps(dict(workspace="/old/workspace", items=[item])))
            result = door.migrated_input("/old/workspace/mobipi/old_run/models/model_epoch_9.pth")
            self.assertEqual(result, root / "CloseSingleDoor/train/same_run/models/model_epoch_9.pth")
            explicit = Path(temporary) / "explicit.json"
            explicit.write_text("{}")
            self.assertEqual(door.migrated_input(explicit), explicit)

    def test_actual_cli_defaults_and_explicit_output_precedence(self):
        cases = [
            (COLLECT / "collect_faucet_demos.py", [], "--output", "demos"),
            (COLLECT / "prepare_faucet_dp.py", ["--source", "input.hdf5"], "--output", "demos"),
            (COLLECT / "prepare_faucet_dp.py", ["--source", "input.hdf5"], "--report", "diagnostics"),
            (COLLECT / "report_faucet_evaluation.py", ["--source", "seeds"], "--output", "reports"),
            (COLLECT / "faucet_progress.py", ["tracking.jsonl"], "--output", "diagnostics"),
        ]
        inputs = {
            "eval_rby1_cartesian_dp.py": ["--run", "run"],
            "evaluate_rby1_cartesian_checkpoints.py": ["--run-dir", "run", "--epochs", "9", "--dataset", "input.hdf5"],
            "report_rby1_cartesian_evaluation.py": ["--input", "rollouts"],
            "replay_rby1_cartesian.py": ["--dataset", "input.hdf5"],
            "diagnose_rby1_ab_rollout.py": ["--checkpoint", "explicit.pth"],
            "diagnose_rby1_cartesian_dp.py": ["--run", "run"],
            "sample_rby1_cartesian_offline.py": ["--run-dir", "run", "--dataset", "input.hdf5"],
            "trace_rby1_cartesian_rollout.py": ["--run-dir", "run", "--dataset", "input.hdf5"],
            "prepare_rby1_cartesian_dp_dataset.py": ["--dataset", "input.hdf5", "--quaternion-key", "robot0_base_to_left_eef_quat_site"],
            "prepare_rby1_cartesian_cmd_dataset.py": ["--dataset", "input.hdf5", "--old-dataset", "old.hdf5"],
            "prepare_rby1_115d_dataset.py": ["--phase", "smoke", "--source", "input.hdf5"],
            "validate_rby1_115d_replay.py": [],
            "validate_rby1_relative_cartesian.py": ["--phase", "dataset", "--config", "input.json"],
            "benchmark_rby1_dp_throughput.py": [],
            "create_rby1_115d_config.py": [],
        }
        categories = {"eval_rby1_cartesian_dp.py": "rollouts", "evaluate_rby1_cartesian_checkpoints.py": "rollouts",
                      "report_rby1_cartesian_evaluation.py": "reports", "prepare_rby1_cartesian_dp_dataset.py": "demos",
                      "prepare_rby1_cartesian_cmd_dataset.py": "demos", "prepare_rby1_115d_dataset.py": "demos",
                      "create_rby1_115d_config.py": "configs"}
        output_dir = {"evaluate_rby1_cartesian_checkpoints.py", "report_rby1_cartesian_evaluation.py",
                      "sample_rby1_cartesian_offline.py", "trace_rby1_cartesian_rollout.py", "validate_rby1_relative_cartesian.py"}
        for name, args in inputs.items():
            flag = "--work-dir" if name == "validate_rby1_115d_replay.py" else "--output-dir" if name in output_dir else "--output"
            cases.append((RM / "robomimic/scripts" / name, args, flag, categories.get(name, "diagnostics")))
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"RBY1_EXPERIMENT_ROOT": temporary}):
            for path, args, flag, category in cases:
                with self.subTest(script=path.name):
                    parser = parser_only(path)
                    parsed = parser.parse_args(args)
                    default = Path(getattr(parsed, flag[2:].replace("-", "_")))
                    task = "TurnOnFaucet" if path.parent == COLLECT else "CloseSingleDoor"
                    self.assertTrue(default.is_relative_to(Path(temporary) / task / category), default)
                    explicit = str(Path(temporary) / "explicit")
                    parsed = parser.parse_args([*args, flag, explicit])
                    self.assertEqual(str(getattr(parsed, flag[2:].replace("-", "_"))), explicit)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_stage_dependent_faucet_defaults_and_override(self):
        source = ast.parse((COLLECT / "faucet_pipeline.py").read_text())
        main = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        branch = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == "args.output is None")
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"RBY1_EXPERIMENT_ROOT": temporary}):
            for stage, category in [("prepare", "demos"), ("generate", "demos"), ("replay", "rollouts"), ("execute-source", "diagnostics")]:
                for output in [None, Path(temporary) / "explicit"]:
                    args = SimpleNamespace(stage=stage, output=output)
                    scope = dict(args=args, new_output=faucet.new_output)
                    exec(compile(ast.Module(body=[branch], type_ignores=[]), "faucet_paths", "exec"), scope)
                    if output is None:
                        self.assertTrue(args.output.is_relative_to(Path(temporary) / "TurnOnFaucet" / category))
                    else:
                        self.assertEqual(args.output, output)

    def test_evaluation_check_only_and_pipeline_defaults(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"RBY1_EXPERIMENT_ROOT": temporary}):
            tree = ast.parse((COLLECT / "evaluate_faucet_dp.py").read_text())
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
            assignment = next(n for n in main.body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Attribute) and t.attr == "output" for t in n.targets))
            for check_only in (True, False):
                for output in (None, Path(temporary) / "explicit"):
                    args = SimpleNamespace(output=output, check_only=check_only)
                    scope = dict(args=args, new_output=faucet.new_output)
                    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "eval_paths", "exec"), scope)
                    category = "diagnostics" if check_only else "rollouts"
                    self.assertEqual(args.output, output) if output else self.assertTrue(args.output.is_relative_to(Path(temporary) / "TurnOnFaucet" / category))
            tree = ast.parse((COLLECT / "pipeline.py").read_text())
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
            assignment = next(n for n in main.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "root" for t in n.targets))
            scope = dict(args=SimpleNamespace(output=None, task="TurnOnSinkFaucet"), artifact_path=faucet.artifact_path)
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), "pipeline_paths", "exec"), scope)
            self.assertEqual(scope["root"], Path(temporary) / "TurnOnFaucet/demos/demos_TurnOnSinkFaucet")

    def test_diagnostic_does_not_choose_latest_run(self):
        path = RM / "robomimic/scripts/diagnose_rby1_cartesian_dp.py"
        main = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "diagnose")
        nodes = main.body[:2]  # Path lookup only; no print, HDF5, policy or simulator work.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "20200101000000"
            second = root / "20300101000000"
            (first / "models").mkdir(parents=True)
            (second / "models").mkdir(parents=True)
            scope = dict(Path=Path, args=SimpleNamespace(run=str(root)))
            with self.assertRaisesRegex(ValueError, "timestamped"):
                exec(compile(ast.Module(body=nodes, type_ignores=[]), "run_paths", "exec"), scope)
            scope["args"].run = str(first)
            exec(compile(ast.Module(body=nodes, type_ignores=[]), "run_paths", "exec"), scope)
            self.assertEqual(scope["run"], first)


if __name__ == "__main__":
    unittest.main()
