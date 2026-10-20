"""CPU-only persistence, provenance and training-log utilities for RB-Y1 evaluation."""
import csv
import json
import math
import os
from pathlib import Path
import re

SCHEMA_VERSION = 1
SUCCESS_THRESHOLD_RAD = -0.07853981634
RGB_KEYS = ("robot0_head_camera_image", "robot0_left_eye_in_hand_image")
LOWDIM_KEYS = ("robot0_base_to_left_eef_pos", "robot0_base_to_left_eef_quat_site")
FIELDS = (
    "checkpoint_epoch", "checkpoint_path", "checkpoint_size", "checkpoint_mtime_ns",
    "demo_key", "split", "seed", "completed", "success", "success_timestep",
    "max_steps", "steps_executed", "final_hinge_angle", "best_hinge_angle",
    "initial_hinge_angle", "mean_tcp_tracking_error_mm", "max_tcp_tracking_error_mm",
    "mean_tcp_orientation_error_deg", "max_tcp_orientation_error_deg",
    "max_tcp_excursion_mm", "final_tcp_position", "final_tcp_orientation",
    "policy_inference_count", "number_action_chunks", "failure_classification",
    "error_stage", "error", "video_path", "wall_seconds",
)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_csv(path, rows, fields):
    """Replace a small table atomically; interruption cannot truncate completed rows."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def as_bool(value):
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


def row_key(row):
    return int(row["checkpoint_epoch"]), row["demo_key"], int(row["seed"])


def valid_completed_row(row):
    try:
        if not set(FIELDS).issubset(row) or not as_bool(row["completed"]):
            return False
        row_key(row)
        as_bool(row["success"])
        int(row["checkpoint_size"])
        int(row["checkpoint_mtime_ns"])
        maximum, steps = int(row["max_steps"]), int(row["steps_executed"])
        if not 0 <= steps <= maximum or maximum < 1:
            return False
        if not row["checkpoint_path"] or not row["failure_classification"]:
            return False
        if not row["error"]:
            if steps != maximum:
                return False
            for key in ("final_hinge_angle", "best_hinge_angle", "initial_hinge_angle"):
                if not math.isfinite(float(row[key])):
                    return False
            reached = float(row["best_hinge_angle"]) >= SUCCESS_THRESHOLD_RAD
            if as_bool(row["success"]) != (float(row["final_hinge_angle"]) >= SUCCESS_THRESHOLD_RAD):
                return False
            if reached != (str(row["success_timestep"]) not in ("", "None")):
                return False
        if str(row["success_timestep"]) not in ("", "None"):
            if not 0 <= int(row["success_timestep"]) <= steps:
                return False
        return True
    except (ValueError, TypeError, KeyError):
        return False


def read_completed_rows(path):
    path = Path(path)
    if not path.exists():
        return [], 0
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Evaluation CSV schema does not match this evaluator")
        rows = list(reader)
    valid = [row for row in rows if valid_completed_row(row)]
    if len({row_key(row) for row in valid}) != len(valid):
        raise ValueError("Duplicate completed epoch/demo/seed rows")
    return valid, len(rows) - len(valid)


def extract_losses(run_dir):
    """Parse completed native JSON epoch blocks; ignore an unfinished current block.

    The trainer prints JSON inside log.txt, so TensorBoard is unnecessary. Also
    accept a logs/losses.json table if another exporter has already produced one.
    """
    run_dir = Path(run_dir)
    table = run_dir / "logs/losses.json"
    if table.exists():
        rows = json.loads(table.read_text())
        if isinstance(rows, list) and all("epoch" in row for row in rows):
            return sorted(rows, key=lambda row: int(row["epoch"]))
    path = run_dir / "logs/log.txt"
    if not path.exists():
        return []
    text = path.read_text(errors="replace")
    decoder = json.JSONDecoder()
    epochs = {}
    for match in re.finditer(r"(?:^|[\r\n])(Train|Validation) Epoch (\d+)\s*(?=\{)", text):
        try:
            block, _ = decoder.raw_decode(text[match.end():])
            loss = float(block["Loss"])
            if not math.isfinite(loss):
                continue
        except (ValueError, KeyError, TypeError):
            continue
        epoch = int(match[2])
        row = epochs.setdefault(epoch, dict(epoch=epoch, train_loss=None, validation_loss=None))
        row["train_loss" if match[1] == "Train" else "validation_loss"] = loss
    return [epochs[epoch] for epoch in sorted(epochs)]
