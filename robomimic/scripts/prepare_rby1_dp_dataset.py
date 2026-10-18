"""
Add the RB-Y1 Cartesian Diffusion Policy action and train / valid masks to a rendered dataset.

General version of the finalize phase of prepare_rby1_115d_dataset.py, for any
number of demos and any RB-Y1 WholeBody IK action layout (31-D fixed-base
demos, 25-D mobile-base demos such as CloseDrawerLeft). Input is a dataset with
image observations, e.g. from dataset_states_to_obs.py.

For every timestep, the recorded WholeBody action is converted into

    actions_eef_cmd[t] = [x, y, z, R[:, 0], R[:, 1]]   (9-D)

the left TCP target of actions[t] in the robot base frame (same definition as
the 115-demo pipeline: inv(T_world_base[t]) @ T_world_ref_command[t] @ inv(T_tcp_ref)).
Each command is converted back into a full action with RBY1CartesianAdapter and
must reproduce the recorded left-arm command, so the adapter used for policy
rollouts executes exactly the demonstrated commands.

Example:
    python robomimic/scripts/prepare_rby1_dp_dataset.py \\
        --dataset ../../datasets/close_drawer_mg/close_drawer_50_im128.hdf5 \\
        --output ../../datasets/close_drawer_mg/close_drawer_50_im128_cmd.hdf5 \\
        --valid-ratio 0.1
"""
import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.rby1_cartesian import RBY1CartesianAdapter, rotation6d_to_matrix, source_command_as_cartesian

REQUIRED_OBS = ["robot0_head_camera_image", "robot0_left_eye_in_hand_image",
                "robot0_base_to_left_eef_pos", "robot0_base_to_left_eef_quat_site"]


def demo_commands(env, group):
    """9-D base-frame TCP commands of one demo, with round-trip errors through the adapter."""
    states, actions = group["states"][:], group["actions"][:]
    env.reset()
    env.reset_to(dict(model=group.attrs["model_file"], ep_meta=group.attrs["ep_meta"], states=states[0]))
    adapter = RBY1CartesianAdapter(env.env)
    start, end = adapter.controller._whole_body_controller_action_split_indexes["left"]

    # the adapter's base frame must be the one of the policy's TCP observations
    base_tcp = np.linalg.inv(adapter.site_transform(adapter.base_site)) @ adapter.site_transform(adapter.tcp_site)
    obs_error = np.linalg.norm(base_tcp[:3, 3] - group["obs/robot0_base_to_left_eef_pos"][0])
    if obs_error > 1e-6:
        raise RuntimeError(f"adapter base frame differs from robot0_base_to_left_eef_pos by {obs_error} m")

    commands = np.empty((len(actions), 9))
    max_pos, max_ori = 0.0, 0.0
    for t, action in enumerate(actions):
        env.env.sim.set_state_from_flattened(states[t])
        env.env.sim.forward()
        commands[t] = source_command_as_cartesian(adapter, action)
        left = adapter(commands[t])[start:end]
        max_pos = max(max_pos, float(np.linalg.norm(left[:3] - action[start:start + 3])))
        rel = Rotation.from_rotvec(left[3:6]).inv() * Rotation.from_rotvec(action[start + 3:end])
        max_ori = max(max_ori, float(np.degrees(rel.magnitude())))
    return commands, max_pos, max_ori


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Rendered dataset (read only)")
    parser.add_argument("--output", required=True, help="New dataset with actions_eef_cmd and masks")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Fraction of demos for validation")
    parser.add_argument("--split-seed", type=int, default=2026)
    args = parser.parse_args()

    source, output = Path(args.dataset).expanduser().resolve(), Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    shutil.copyfile(source, output)

    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": ["robot0_base_to_left_eef_pos"]})
    env_meta = FileUtils.get_env_metadata_from_dataset(str(output))
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False)

    report = {"dataset": str(output), "demos": {}}
    with h5py.File(output, "r+") as f:
        names = sorted(f["data"], key=lambda n: int(n.split("_")[1]))
        for name in names:
            group = f["data"][name]
            missing = [k for k in REQUIRED_OBS if f"obs/{k}" not in group]
            if missing:
                raise RuntimeError(f"{name}: missing observations {missing}; render the dataset first")
            commands, max_pos, max_ori = demo_commands(env, group)
            # the adapter must reproduce the recorded command exactly
            if max_pos > 1e-6 or max_ori > 1e-4:
                raise RuntimeError(f"{name}: adapter round trip error {max_pos} m / {max_ori} deg")
            if "actions_eef_cmd" in group:
                del group["actions_eef_cmd"]
            # float64 like the 115-demo pipeline: robomimic scales to +-0.999999, and
            # float32 rounding exceeds that margin on rotation dims with a small range
            ds = group.create_dataset("actions_eef_cmd", data=commands)
            ds.attrs.update(frame="robot_base_center", tcp_site="gripper0_left_grip_site",
                            ik_reference_site="gripper0_left_ft_frame", rotation6d="concatenate(R[:,0], R[:,1])",
                            temporal_alignment="original actions[t], no shift", source_action_key="actions",
                            definition="inv(T_world_base[t]) @ T_world_ref_command[t] @ inv(T_tcp_ref)")
            jumps = np.linalg.norm(np.diff(commands[:, :3], axis=0), axis=1)
            rot = Rotation.from_matrix(rotation6d_to_matrix(commands[:, 3:]))
            report["demos"][name] = dict(steps=len(commands), round_trip_m=max_pos, round_trip_deg=max_ori,
                                         max_position_jump_m=float(jumps.max()),
                                         max_orientation_jump_deg=float(np.degrees((rot[:-1].inv() * rot[1:]).magnitude()).max()))
            print(f"{name}: {len(commands)} steps, round trip {max_pos:.1e} m / {max_ori:.1e} deg, "
                  f"max step {jumps.max() * 1000:.1f} mm", flush=True)

        n_valid = max(1, int(round(args.valid_ratio * len(names))))
        permutation = np.random.default_rng(args.split_seed).permutation(names)
        by_index = lambda ns: sorted(ns, key=lambda n: int(n.split("_")[1]))
        valid, train = by_index(permutation[:n_valid].tolist()), by_index(permutation[n_valid:].tolist())
        for key, demos in (("train", train), ("valid", valid)):
            if f"mask/{key}" in f:
                del f[f"mask/{key}"]
            f.create_dataset(f"mask/{key}", data=np.asarray(demos, dtype="S"))
        f["mask"].attrs.update(split_seed=args.split_seed, valid_ratio=args.valid_ratio)

    all_steps = sum(d["steps"] for d in report["demos"].values())
    report.update(train=train, valid=valid, timesteps=all_steps)
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {output}: {len(names)} demos ({len(train)} train / {len(valid)} valid), {all_steps} steps")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
