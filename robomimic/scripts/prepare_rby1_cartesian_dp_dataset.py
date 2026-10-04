"""Copy demonstrations and add validated next-step base-frame Cartesian targets.

The quaternion key is explicit because legacy robosuite EEF body orientations
can differ from grip-site orientations. The source file is always read-only.
"""

import argparse
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from robomimic.utils.rby1_cartesian import quaternion_to_rotation6d, rotation6d_to_matrix


POSITION_KEY = "robot0_base_to_left_eef_pos"


def derive_and_validate(demo, quaternion_key):
    """Return float32 targets and conversion diagnostics for one episode."""
    position = demo[f"obs/{POSITION_KEY}"][:]
    quaternion = demo[f"obs/{quaternion_key}"][:]
    length = demo["actions"].shape[0]
    if length < 1 or position.shape != (length, 3) or quaternion.shape != (length, 4):
        raise ValueError(f"{demo.name}: invalid pose shapes or empty episode")
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError(f"{demo.name}: non-finite source pose")
    if not np.allclose(np.linalg.norm(quaternion, axis=-1), 1, atol=1e-5):
        raise ValueError(f"{demo.name}: source quaternions are not unit length")
    indices = np.minimum(np.arange(length) + 1, length - 1)
    targets = np.concatenate((position[indices], quaternion_to_rotation6d(quaternion[indices])), axis=-1).astype(np.float32)
    matrices = rotation6d_to_matrix(targets[:, 3:])
    position_error = np.linalg.norm(targets[:, :3] - position[indices], axis=-1)
    angle_error = np.rad2deg((Rotation.from_matrix(matrices).inv() * Rotation.from_quat(quaternion[indices])).magnitude())
    orthogonality = np.max(np.abs(matrices.swapaxes(-1, -2) @ matrices - np.eye(3)))
    determinant = np.max(np.abs(np.linalg.det(matrices) - 1))
    if position_error.max() > 1e-6 or angle_error.max() > 1e-4 or orthogonality > 1e-10 or determinant > 1e-10:
        raise ValueError(f"{demo.name}: conversion sanity check failed")
    sample_indices = np.unique(np.linspace(0, length - 1, 5, dtype=int))
    report = {
        "episode": demo.name, "shape": list(targets.shape), "dtype": str(targets.dtype),
        "nan_count": int(np.isnan(targets).sum()), "inf_count": int(np.isinf(targets).sum()),
        "xyz_min": targets[:, :3].min(axis=0).tolist(), "xyz_max": targets[:, :3].max(axis=0).tolist(),
        "rotation6d_min": targets[:, 3:].min(axis=0).tolist(), "rotation6d_max": targets[:, 3:].max(axis=0).tolist(),
        "max_position_error_m": float(position_error.max()), "max_rotation_error_deg": float(angle_error.max()),
        "max_orthogonality_error": float(orthogonality), "max_determinant_error": float(determinant),
        "samples": [{"t": int(t), "source_t": int(indices[t]), "position_error_m": float(position_error[t]),
                     "rotation_error_deg": float(angle_error[t])} for t in sample_indices],
    }
    return targets, report


def prepare(source, output, quaternion_key, selected_demo=None):
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if source == output or output.exists():
        raise ValueError("Output must be a new file distinct from the source")
    # Validate before creating the output, then validate the stored representation.
    with h5py.File(source, "r") as src:
        names = [selected_demo] if selected_demo else list(src["data"])
        for name in names:
            derive_and_validate(src[f"data/{name}"], quaternion_key)
    output.parent.mkdir(parents=True, exist_ok=True)
    if selected_demo is None:
        shutil.copyfile(source, output)
    else:
        with h5py.File(source, "r") as src, h5py.File(output, "x") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            for key in src:
                if key not in ("data", "mask"):
                    src.copy(key, dst)
            data = dst.create_group("data")
            for key, value in src["data"].attrs.items():
                data.attrs[key] = value
            src.copy(src[f"data/{selected_demo}"], data, name=selected_demo)
            data.attrs["total"] = data[selected_demo]["actions"].shape[0]
            if "mask" in src:
                mask = dst.create_group("mask")
                for key, value in src["mask"].attrs.items():
                    mask.attrs[key] = value
                for key, value in src["mask"].items():
                    keep = [item for item in value[:] if item.decode() == selected_demo]
                    mask.create_dataset(key, data=np.asarray(keep, dtype=value.dtype))
    with h5py.File(output, "r+") as dst:
        for name in names:
            demo = dst[f"data/{name}"]
            if "actions_eef" in demo:
                raise ValueError(f"{demo.name}: actions_eef already exists")
            targets, report = derive_and_validate(demo, quaternion_key)
            stored = demo.create_dataset("actions_eef", data=targets)
            stored.attrs["position_key"] = POSITION_KEY
            stored.attrs["quaternion_key"] = quaternion_key
            stored.attrs["quaternion_order"] = "xyzw"
            stored.attrs["frame"] = "robot_base_center"
            stored.attrs["rotation6d"] = "concatenate(R[:,0], R[:,1])"
            stored.attrs["temporal_alignment"] = "pose[min(t+1,T-1)]"
            if not np.array_equal(stored[:], targets):
                raise ValueError("Stored targets differ from validated targets")
            print(json.dumps(report))
    print(f"Derived dataset: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--demo", help="Copy only this demo for the first sanity/replay test")
    parser.add_argument("--quaternion-key", required=True, choices=[
        "robot0_base_to_left_eef_quat", "robot0_base_to_left_eef_quat_site"])
    args = parser.parse_args()
    prepare(args.dataset, args.output, args.quaternion_key, args.demo)


if __name__ == "__main__":
    main()
