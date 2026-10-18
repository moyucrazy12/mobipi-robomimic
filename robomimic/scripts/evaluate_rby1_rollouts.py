"""
Roll out RB-Y1 Cartesian Diffusion Policy checkpoints in simulation and report success rates.

The policy predicts the 9-D left TCP target in the robot base frame
(actions_eef_cmd); RBY1CartesianAdapter turns it into the full WholeBody IK
action, holding the other parts where they are after the reset. The environment
comes from the checkpoint's metadata (the training dataset's env_args, with its
cameras), so the task (e.g. CloseDrawerLeft), controller and scene settings
match the demonstrations. Every checkpoint is evaluated on the same sequence of
scenes (--seed).

Example:
    python robomimic/scripts/evaluate_rby1_rollouts.py \\
        --run-dir experiment_results/rby1_close_drawer_cartesian_cmd_50d/20261013122853 \\
        --epochs 10 20 30 --episodes 20 --output-dir experiment_results/eval_close_drawer_50d

    # unseen kitchens (Mobi-pi test layouts), with videos
    python robomimic/scripts/evaluate_rby1_rollouts.py --checkpoint path/to/model_epoch_10.pth \\
        --layouts 1 4 7 8 9 --episodes 10 --videos --output-dir experiment_results/eval_test_layouts
"""
import argparse
import json
import time
from pathlib import Path

import imageio
import numpy as np
import torch

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.rby1_cartesian import RBY1CartesianAdapter

VIDEO_CAMERAS = ["robot0_head_camera", "robot0_left_eye_in_hand"]


def make_env(ckpt_dict, config, args):
    """Environment of the checkpoint's training data, with optional scene overrides."""
    env_meta = ckpt_dict["env_metadata"]
    env_meta["env_kwargs"]["seed"] = args.seed
    if args.layouts:
        env_meta["env_kwargs"]["layout_ids"] = args.layouts
    if args.styles:
        env_meta["env_kwargs"]["style_ids"] = args.styles
    shape_meta = ckpt_dict["shape_metadata"]
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=shape_meta["use_images"],
    )
    return EnvUtils.wrap_env_from_config(env, config=config)


def run_episode(env, policy, args, video_writer=None):
    obs = env.reset()
    policy.start_episode()
    base_env = env.env.env  # FrameStackWrapper -> EnvRobosuite -> robosuite env
    adapter = RBY1CartesianAdapter(base_env)
    meta = base_env.get_ep_meta()
    success, steps = False, 0
    for steps in range(1, args.horizon + 1):
        command = policy(ob=obs)
        obs, _, _, _ = env.step(adapter(command))
        if video_writer is not None and steps % args.video_skip == 0:
            video_writer.append_data(np.concatenate(
                [env.render(mode="rgb_array", height=256, width=256, camera_name=c) for c in VIDEO_CAMERAS], axis=1))
        if env.is_success()["task"]:
            success = True
            break
    drawer = getattr(base_env, "drawer", None)
    result = dict(success=success, steps=steps, layout=meta.get("layout_id"), style=meta.get("style_id"),
                  fixture=getattr(drawer, "name", None))
    if drawer is not None:
        # fraction of fully open (RoboCasa's success threshold for closing is 0.05)
        result["final_open"] = float(max(drawer.get_door_state(env=base_env).values()))
    return result


def evaluate_checkpoint(ckpt_path, args, out_dir):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=str(ckpt_path), device=device)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = make_env(ckpt_dict, config, args)

    name = ckpt_path.stem
    episodes, start = [], time.time()
    for ep in range(args.episodes):
        writer = None
        if args.videos:
            writer = imageio.get_writer(str(out_dir / f"{name}_ep{ep:02d}.mp4"), fps=20 // args.video_skip)
        try:
            result = run_episode(env, policy, args, writer)
        finally:
            if writer is not None:
                writer.close()
        episodes.append(result)
        print(f"{name} episode {ep}: {'success' if result['success'] else 'fail'} after {result['steps']} steps "
              f"(layout {result['layout']}, style {result['style']}, {result['fixture']}, "
              f"final opening {result.get('final_open', float('nan')):.3f})", flush=True)

    successes = [e for e in episodes if e["success"]]
    summary = dict(checkpoint=str(ckpt_path), episodes=len(episodes), success_rate=len(successes) / len(episodes),
                   mean_success_steps=float(np.mean([e["steps"] for e in successes])) if successes else None,
                   horizon=args.horizon, seed=args.seed, seconds=time.time() - start, results=episodes)
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", nargs="+", default=[], help="Checkpoint file(s)")
    parser.add_argument("--run-dir", help="Training run directory (with models/)")
    parser.add_argument("--epochs", type=int, nargs="+", help="Epochs to evaluate from --run-dir (default: all saved)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000, help="Max steps per episode (demos take ~600)")
    parser.add_argument("--seed", type=int, default=0, help="Scene / sampling seed, the same for every checkpoint")
    parser.add_argument("--layouts", type=int, nargs="+", help="Override the training layouts")
    parser.add_argument("--styles", type=int, nargs="+", help="Override the training styles")
    parser.add_argument("--videos", action="store_true", help="Save a head + wrist camera video per episode")
    parser.add_argument("--video-skip", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoints = [Path(p).expanduser().resolve() for p in args.checkpoint]
    if args.run_dir:
        models = Path(args.run_dir).expanduser().resolve() / "models"
        if args.epochs:
            checkpoints += [models / f"model_epoch_{e}.pth" for e in args.epochs]
        else:
            checkpoints += sorted(models.glob("model_epoch_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
    if not checkpoints:
        parser.error("give --checkpoint or --run-dir")
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = [evaluate_checkpoint(c, args, out_dir) for c in checkpoints]
    print("\n===== Success rates =====")
    for s in summaries:
        steps = f", {s['mean_success_steps']:.0f} steps on success" if s["mean_success_steps"] else ""
        print(f"{Path(s['checkpoint']).name:24s} {s['success_rate'] * 100:5.1f}% of {s['episodes']}{steps}")
    (out_dir / "summary.json").write_text(json.dumps(
        [{k: v for k, v in s.items() if k != "results"} for s in summaries], indent=2))


if __name__ == "__main__":
    main()
