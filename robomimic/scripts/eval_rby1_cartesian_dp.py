"""Closed-loop fixed-base RB-Y1 evaluation using native Robomimic inference."""

import argparse
import csv
import json
from pathlib import Path

from robomimic.utils.rby1_paths import new_output
import random
import re
import xml.etree.ElementTree as ET

import h5py
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from robomimic.utils import file_utils as FileUtils
from robomimic.utils.rby1_cartesian import RBY1CartesianAdapter, rotation6d_to_matrix


RGB = ["robot0_head_camera_image", "robot0_left_eye_in_hand_image"]
LOWDIM = ["robot0_base_to_left_eef_pos", "robot0_base_to_left_eef_quat_site"]


def offset_robot_base_y(model_xml, root_body, offset):
    """Shift the fixed robot root in world Y while preserving its orientation."""
    root = ET.fromstring(model_xml)
    body = root.find(f"./worldbody/body[@name='{root_body}']")
    if body is None:
        raise ValueError(f"Robot root body {root_body!r} is not a worldbody child")
    if body.find("joint") is not None or body.find("freejoint") is not None:
        raise ValueError("Expected a fixed robot root; saved joints could override placement")
    position = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
    if position.shape != (3,):
        raise ValueError("Invalid robot root position")
    position[1] += offset
    body.set("pos", " ".join(str(value) for value in position))
    return ET.tostring(root, encoding="unicode")


def select_checkpoint(run):
    """Select a periodic checkpoint from unsmoothed terminal validation losses."""
    pairs = [(int(epoch), json.loads(body)["Loss"]) for epoch, body in re.findall(
        r"Validation Epoch (\d+)\s*(\{.*?\})", (run / "logs/log.txt").read_text(), re.S)]
    if not pairs:
        raise ValueError("No raw validation losses found")
    epoch, loss = min(pairs, key=lambda item: item[1])
    checkpoint = run / "models" / f"model_epoch_{epoch}.pth"
    if not checkpoint.is_file():
        raise ValueError(f"Selected checkpoint not saved yet: {checkpoint}")
    return checkpoint, epoch, loss, pairs


def check_checkpoint(policy, checkpoint):
    config = policy.policy.global_config
    assert config.algo_name == "diffusion_policy"
    assert list(config.train.action_keys) in (["actions_eef"], ["actions_eef_cmd"])
    assert checkpoint["shape_metadata"]["ac_dim"] == 9
    obs = config.observation.modalities.obs
    assert list(obs.rgb) == RGB and list(obs.low_dim) == LOWDIM
    assert not obs.depth and not obs.scan and not config.algo.language_conditioned
    assert set(checkpoint["shape_metadata"]["all_shapes"]) == set(RGB + LOWDIM)
    assert config.train.frame_stack == 2
    assert dict(config.algo.horizon) == dict(observation_horizon=2, prediction_horizon=16, action_horizon=8)
    assert policy.action_normalization_stats is not None
    meta = checkpoint["env_metadata"]
    assert meta["env_name"] == "CloseSingleDoor"
    assert meta["env_kwargs"]["robots"] == "RBY1"
    assert meta["env_kwargs"]["control_freq"] == 20
    assert meta["env_kwargs"]["force_robot_placement"] is not None


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(args):
    run = Path(args.run).expanduser().resolve()
    chosen, epoch, loss, pairs = select_checkpoint(run)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else chosen
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    policy, checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint_path), device=torch.device(args.device))
    check_checkpoint(policy, checkpoint)
    if args.initialization == "fresh":
        position, orientation = checkpoint["env_metadata"]["env_kwargs"]["force_robot_placement"]
        position = list(position)
        position[1] += args.base_offset_y
        checkpoint["env_metadata"]["env_kwargs"]["force_robot_placement"] = [position, orientation]
    dataset_path = Path(args.dataset).expanduser()
    with h5py.File(dataset_path, "r") as dataset:
        demos = [name.decode() for name in dataset["mask/valid"][:]]
        # Never claim repeated saved starts are independent scene samples.
        if args.initialization == "validation" and args.episodes > len(demos):
            raise ValueError(f"Only {len(demos)} distinct held-out initial states available")
        assert all(np.all(dataset[f"data/{demo}/actions"][:, -1] == 1) for demo in demos)
        source_meta = json.loads(dataset["data"].attrs["env_args"])
        assert source_meta["env_kwargs"]["controller_configs"] == checkpoint["env_metadata"]["env_kwargs"]["controller_configs"]

    results = []
    summary = {"checkpoint": str(checkpoint_path), "best_epoch": epoch,
               "lowest_raw_validation_loss": loss, "raw_validation_losses": pairs,
               "initialization": args.initialization, "episodes": results,
               "base_offset_world_y_m": args.base_offset_y,
               "original_episode_seeds": "Not stored in per-episode metadata; evaluation seeds recorded below"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    for episode in range(args.episodes):
        seed = args.seed + episode
        seed_all(seed)
        checkpoint["env_metadata"]["env_kwargs"]["seed"] = seed
        if args.render:
            # Use the interactive free camera without altering observation cameras.
            checkpoint["env_metadata"]["env_kwargs"]["render_camera"] = None
            checkpoint["env_metadata"]["env_kwargs"]["renderer"] = "mjviewer"
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=checkpoint, render=args.render, render_offscreen=True)
        raw = env.unwrapped.env
        writer = None
        try:
            if args.initialization == "validation":
                demo = demos[episode]
                with h5py.File(dataset_path, "r") as dataset:
                    group = dataset[f"data/{demo}"]
                    model_xml = group.attrs["model_file"]
                    if args.base_offset_y != 0:
                        model_xml = offset_robot_base_y(
                            model_xml, raw.robots[0].robot_model.root_body, args.base_offset_y)
                    ob = env.reset_to({"model": model_xml,
                                       "ep_meta": group.attrs["ep_meta"], "states": group["states"][0]})
            else:
                demo = None
                ob = env.reset()
            if args.render:
                # Native env.step synchronizes this viewer on subsequent steps.
                raw.viewer.update()
            adapter = RBY1CartesianAdapter(raw)
            assert raw.action_dim == 31 and raw.control_freq == 20
            assert adapter.reference_site == "gripper0_left_ft_frame"
            assert adapter.tcp_site == "gripper0_left_grip_site"
            base0 = adapter.site_transform(adapter.base_site).copy()
            print("Initial robot base world XYZ:", base0[:3, 3].tolist(), flush=True)
            policy.start_episode()
            actual_shapes = {key: list(ob[key].shape) for key in RGB + LOWDIM}
            assert actual_shapes == {**{key: [2, 3, 128, 128] for key in RGB},
                                     LOWDIM[0]: [2, 3], LOWDIM[1]: [2, 4]}
            print("Policy inputs (two frames, RGB CHW float):", actual_shapes, flush=True)
            if args.video and episode == 0:
                camera_names = raw.sim.model.camera_names
                if args.video_camera not in camera_names:
                    print("Available MuJoCo cameras:", camera_names, flush=True)
                    raise ValueError(f"Video camera {args.video_camera!r} does not exist")
                writer = imageio.get_writer(str(output / "episode_0.mp4"), fps=20)
            rows = []
            success = bool(env.is_success()["task"])
            if success:
                raise ValueError("Episode starts already successful")
            last_action = None
            for timestep in range(args.horizon):
                action = policy({key: ob[key] for key in RGB + LOWDIM})
                if action.shape != (9,) or not np.isfinite(action).all():
                    raise ValueError(f"Corrupt policy output at step {timestep}: {action}")
                matrix = rotation6d_to_matrix(action[3:])
                determinant = float(np.linalg.det(matrix))
                orthogonality = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
                if abs(determinant - 1) > 1e-8 or orthogonality > 1e-8:
                    raise ValueError("Invalid decoded rotation")
                target = adapter.target_world(action)
                packed = adapter(action)
                assert packed.shape == (31,) and packed[-1] == 1
                ob, reward, done, info = env.step(packed)
                actual = adapter.site_transform(adapter.tcp_site)
                position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
                angle_error = float(np.rad2deg(Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).magnitude()))
                base_drift = float(np.max(np.abs(adapter.site_transform(adapter.base_site) - base0)))
                if base_drift > 1e-8:
                    raise ValueError(f"Base moved: {base_drift}")
                success = bool(env.is_success()["task"])
                row = {"timestep": timestep, "action": json.dumps(action.tolist()),
                       "predicted_xyz": json.dumps(action[:3].tolist()), "rotation_determinant": determinant,
                       "rotation_orthogonality_error": orthogonality,
                       "actual_tcp_world_xyz": json.dumps(actual[:3, 3].tolist()),
                       "position_error_m": position_error, "orientation_error_deg": angle_error,
                       "target_jump_m": 0.0 if last_action is None else float(np.linalg.norm(action[:3] - last_action[:3])),
                       "queue_remaining": len(policy.policy.action_queue), "base_drift": base_drift,
                       "success": success}
                rows.append(row)
                last_action = action.copy()
                if writer is not None:
                    # Render separately from the unchanged policy observations.
                    writer.append_data(env.render(
                        mode="rgb_array", height=512, width=512, camera_name=args.video_camera))
                if timestep % 100 == 0:
                    print(f"Episode {episode}, step {timestep}: TCP error {position_error:.4f} m / {angle_error:.2f} deg, success={success}", flush=True)
                if success or done:
                    break
            with (output / f"episode_{episode}.csv").open("w", newline="") as stream:
                csv_writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                csv_writer.writeheader()
                csv_writer.writerows(rows)
            result = {"episode": episode, "seed": seed, "demo": demo, "success": success,
                      "length": len(rows), "checkpoint": str(checkpoint_path), "policy_shapes": actual_shapes,
                      "base_world_transform": base0.tolist(), "door_state": raw.door_fxtr.get_door_state(env=raw),
                      "position_error_mean_m": float(np.mean([r["position_error_m"] for r in rows])),
                      "position_error_max_m": max(r["position_error_m"] for r in rows),
                      "orientation_error_mean_deg": float(np.mean([r["orientation_error_deg"] for r in rows])),
                      "orientation_error_max_deg": max(r["orientation_error_deg"] for r in rows),
                      "target_jump_mean_m": float(np.mean([r["target_jump_m"] for r in rows])),
                      "target_jump_max_m": max(r["target_jump_m"] for r in rows)}
            results.append(result)
            successful_lengths = [r["length"] for r in results if r["success"]]
            summary.update(successes=sum(r["success"] for r in results), success_rate=float(np.mean([r["success"] for r in results])),
                           mean_episode_length=float(np.mean([r["length"] for r in results])),
                           mean_successful_episode_length=float(np.mean(successful_lengths)) if successful_lengths else None)
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(result), flush=True)
        finally:
            if writer is not None:
                writer.close()
            raw.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", help="Explicit later comparison; defaults to lowest raw validation loss")
    parser.add_argument("--dataset", default="~/Downloads/demo_im128_head_wrist_cartesian.hdf5")
    parser.add_argument("--output", default=str(new_output("CloseSingleDoor", "rollouts", "rby1_cartesian_dp")), help="New directory; existing results are never overwritten")
    parser.add_argument("--initialization", choices=("validation", "fresh"), default="validation")
    parser.add_argument("--base-offset-y", type=float, default=-0.2,
                        help="World-Y base offset in meters; default moves 20 cm backward, 0 restores original start")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=2600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--render", action="store_true",
                        help="Open an interactive MuJoCo viewer during evaluation")
    parser.add_argument("--video-camera", default="freeview",
                        help="MuJoCo camera for saved video only; does not affect policy observations")
    args = parser.parse_args()
    if args.episodes < 1 or args.horizon < 1:
        parser.error("Episodes and horizon must be positive")
    if not np.isfinite(args.base_offset_y):
        parser.error("Base offset must be finite")
    evaluate(args)


if __name__ == "__main__":
    main()
