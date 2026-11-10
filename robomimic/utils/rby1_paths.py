"""Lightweight path resolution for custom RB-Y1 commands only."""
import copy
from datetime import datetime
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
ROBOMIMIC_ROOT = Path(__file__).resolve().parents[2]


def experiment_root():
    value = os.environ.get("RBY1_EXPERIMENT_ROOT")
    path = Path(os.path.expandvars(value)).expanduser() if value else PROJECT_ROOT / "experiment_results"
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def artifact_path(task, category, *parts):
    return experiment_root() / task / category / Path(*parts)


def new_output(task, category, name, suffix=""):
    return artifact_path(task, category, f"{name}_{datetime.now():%Y%m%d_%H%M%S_%f}{suffix}")


def resolve_training_paths(settings, output_dir=None, dataset=None):
    """Bind output/input paths in a copy; all numerical settings remain literal."""
    settings = copy.deepcopy(settings)
    name = settings["experiment"]["name"]
    if name.startswith("rby1_faucet_"):
        task = "TurnOnFaucet"
    elif name.startswith("rby1_close_single_door_"):
        task = "CloseSingleDoor"
    else:
        raise ValueError("Expected a custom RB-Y1 faucet or CloseSingleDoor configuration")
    settings["train"]["output_dir"] = str(
        Path(output_dir).expanduser().resolve() if output_dir is not None else artifact_path(task, "train"))
    for item in settings["train"]["data"]:
        value = item["path"].replace("${RBY1_EXPERIMENT_ROOT}", str(experiment_root()))
        item["path"] = str(Path(os.path.expandvars(value)).expanduser().resolve())
    if dataset is not None:
        if len(settings["train"]["data"]) != 1:
            raise ValueError("--dataset requires a single configured dataset")
        settings["train"]["data"][0]["path"] = str(Path(dataset).expanduser().resolve())
    return settings


def migrated_input(value):
    """Resolve a metadata path through the immutable manifest; never select a model."""
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    manifest = PROJECT_ROOT / "experiment_results/migration_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        for item in data["items"]:
            for workspace in (PROJECT_ROOT, Path(data["workspace"])):
                original = workspace / item["original_path"]
                if path.is_relative_to(original):
                    return PROJECT_ROOT / item["new_path"] / path.relative_to(original)
    return path.resolve()
