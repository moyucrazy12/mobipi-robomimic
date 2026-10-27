"""Evaluate 9D command checkpoints on saved RB-Y1 models; never use fresh physics.

This script is intentionally not launched by training or documentation tools.
Run after training, or when GPU evaluation is explicitly authorized.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import random
import time

from robomimic.utils.rby1_evaluation import (
    FIELDS, LOWDIM_KEYS, RGB_KEYS, SCHEMA_VERSION, SUCCESS_THRESHOLD_RAD,
    atomic_json, fingerprint, read_completed_rows, row_key, write_csv,
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", required=True)
    result.add_argument("--epochs", nargs="+", type=int, required=True)
    result.add_argument("--dataset", required=True)
    result.add_argument("--filter-key", default="valid")
    selection = result.add_mutually_exclusive_group()
    selection.add_argument("--demo-keys", nargs="+")
    selection.add_argument("--num-demos", type=int, default=5)
    result.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42, 123, 12345])
    result.add_argument("--max-steps", type=int, default=2000)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--device", default="cuda")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--skip-missing-checkpoints", action="store_true")
    videos = result.add_mutually_exclusive_group()
    videos.add_argument("--save-videos", dest="save_videos", action="store_true")
    videos.add_argument("--no-save-videos", dest="save_videos", action="store_false")
    result.set_defaults(save_videos=False)
    result.add_argument("--video-camera", default="freeview", help="Visualization only")
    return result


def select_demos(dataset, filter_key, explicit, number):
    names = sorted(dataset[f"mask/{filter_key}"].asstr()[:], key=lambda n: int(n.split("_")[-1]))
    if explicit:
        if len(set(explicit)) != len(explicit) or not set(explicit).issubset(names):
            raise ValueError("Requested demos must be unique members of the selected mask")
        return explicit
    if number < 1 or number > len(names):
        raise ValueError(f"num-demos must be between 1 and {len(names)}")
    # Deterministically spread across the selected split, without random scene sampling.
    indices = [0] if number == 1 else [i * (len(names) - 1) // (number - 1) for i in range(number)]
    return [names[index] for index in indices]


def seed_rollout(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_policy(policy, checkpoint, dataset_metadata):
    import torch
    config = policy.policy.global_config
    if config.algo_name != "diffusion_policy" or list(config.train.action_keys) != ["actions_eef_cmd"]:
        raise ValueError("Expected command-target Diffusion Policy")
    if checkpoint["shape_metadata"]["ac_dim"] != 9:
        raise ValueError("Expected nine action dimensions")
    obs = config.observation.modalities.obs
    if list(obs.rgb) != list(RGB_KEYS) or list(obs.low_dim) != list(LOWDIM_KEYS):
        raise ValueError("Checkpoint policy observation keys differ from the baseline")
    shapes = checkpoint["shape_metadata"]["all_shapes"]
    if set(shapes) != set(RGB_KEYS + LOWDIM_KEYS):
        raise ValueError("Unexpected checkpoint observation modalities")
    expected = {**{key: [3, 128, 128] for key in RGB_KEYS}, LOWDIM_KEYS[0]: [3], LOWDIM_KEYS[1]: [4]}
    if {key: list(value) for key, value in shapes.items()} != expected:
        raise ValueError("Expected native 128px RGB and TCP position/quaternion shapes")
    if config.train.frame_stack != 2 or dict(config.algo.horizon) != dict(
            observation_horizon=2, action_horizon=8, prediction_horizon=16):
        raise ValueError("Unexpected frame stack or diffusion horizons")
    if not config.algo.ddim.enabled or config.algo.ddim.num_inference_timesteps != 10:
        raise ValueError("Expected ten-step DDIM")
    if config.train.action_config.actions_eef_cmd.normalization != "min_max":
        raise ValueError("Expected min-max command normalization")
    if policy.action_normalization_stats is None or set(policy.action_normalization_stats) != {"actions_eef_cmd"}:
        raise ValueError("Missing command normalization")
    ema = policy.policy.ema
    if not config.algo.ema.enabled or ema is None or checkpoint["model"].get("ema") is None:
        raise ValueError("Missing EMA weights")
    for key, value in ema.averaged_model.state_dict().items():
        if not torch.equal(value.cpu(), checkpoint["model"]["ema"][key].cpu()):
            raise ValueError("EMA checkpoint restoration mismatch")
    if checkpoint["env_metadata"]["env_name"] != dataset_metadata["env_name"]:
        raise ValueError("Checkpoint task mismatch")
    if checkpoint["env_metadata"]["env_kwargs"]["controller_configs"] != dataset_metadata["env_kwargs"]["controller_configs"]:
        raise ValueError("Checkpoint and dataset controllers differ")


def observation_reader(env, adapter):
    """Use native lowdim sensors and the already validated RGB rendering convention."""
    import numpy as np
    import robosuite.macros as macros
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
    from robomimic.utils import obs_utils as O
    convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

    def read():
        raw = env._get_observations(force_update=True)
        result = {key: np.asarray(raw[key]).copy() for key in LOWDIM_KEYS}
        for key in RGB_KEYS:
            frame = env.sim.render(width=128, height=128, camera_name=key[:-6])[::convention][::-1].copy()
            result[key] = O.process_obs(frame, obs_key=key)
        return result
    return read


def execute_loop(policy, env, adapter, read_observation, hinge_address, max_steps, writer=None, video_camera="freeview",
                 observer=None, progress_fn=None, success_threshold=SUCCESS_THRESHOLD_RAD):
    """Normal native queued inference; diagnostics compare post-step TCP to executed target.

    Task progress is the door hinge angle at hinge_address by default. Other tasks
    pass progress_fn() (higher is closer to done) and its success_threshold;
    results then carry *_progress keys instead of *_hinge_angle keys. An observer
    returning False ends the rollout after that step.
    """
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    hinge_mode = progress_fn is None
    if hinge_mode:
        progress_fn = lambda: float(env.sim.data.qpos[hinge_address])
    initial_tcp = adapter.site_transform(adapter.tcp_site)
    initial_progress = float(progress_fn())
    current = read_observation()
    history = deque([current, current], maxlen=2)
    position_errors, orientation_errors = [], []
    best = initial_progress
    success_step = 0 if initial_progress >= success_threshold else None
    chunks = 0
    max_excursion = 0.0
    final_tcp = initial_tcp
    final_progress = initial_progress
    completed_steps = 0
    error_stage, error = "", ""
    for step in range(max_steps):
        try:
            error_stage = "policy"
            regenerated = not policy.policy.action_queue
            previous_tcp = final_tcp.copy()
            previous_progress = final_progress
            if regenerated:
                chunks += 1
            inputs = {key: np.stack([item[key] for item in history]) for key in RGB_KEYS + LOWDIM_KEYS}
            with torch.no_grad():
                action = np.asarray(policy(inputs), dtype=float)
            target = adapter.target_world(action)
            packed = adapter(action)
            error_stage = "controller"
            env.step(packed)
            completed_steps = step + 1
            if not np.isfinite(env.sim.data.qpos).all() or not np.isfinite(env.sim.data.qvel).all():
                raise ValueError("Nonfinite simulator state")
            next_tcp = adapter.site_transform(adapter.tcp_site)
            next_progress = float(progress_fn())
            if not np.isfinite(next_tcp).all() or not np.isfinite(next_progress):
                raise ValueError("Nonfinite TCP or task progress")
            final_tcp, final_progress = next_tcp, next_progress
            position_errors.append(float(np.linalg.norm(final_tcp[:3, 3] - target[:3, 3])) * 1000)
            orientation_errors.append(float(np.rad2deg(Rotation.from_matrix(
                target[:3, :3].T @ final_tcp[:3, :3]).magnitude())))
            max_excursion = max(max_excursion, float(np.linalg.norm(final_tcp[:3, 3] - initial_tcp[:3, 3])) * 1000)
            best = max(best, final_progress)
            if final_progress >= success_threshold and success_step is None:
                success_step = step + 1
            if observer is not None:
                error_stage = "trace"
                keep_going = observer(step=step, regenerated=regenerated, inputs=inputs, action=action,
                                      target=target, packed=packed, actual_before=previous_tcp,
                                      actual_after=final_tcp, progress_before=previous_progress,
                                      progress_after=final_progress,
                                      **(dict(hinge_before=previous_progress, hinge_after=final_progress)
                                         if hinge_mode else {}),
                                      position_error_mm=position_errors[-1],
                                      orientation_error_deg=orientation_errors[-1],
                                      queue_remaining=len(policy.policy.action_queue))
                if keep_going is False:  # the observer ends the rollout early
                    break
            error_stage = "observation"
            history.append(read_observation())
            if writer is not None:
                error_stage = "video"
                writer.append_data(env.sim.render(width=640, height=480, camera_name=video_camera)[::-1])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
    success = not error and final_progress >= success_threshold
    if error:
        category = "controller_error" if error_stage == "controller" else "runtime_error"
    elif success:
        category = "success"
    elif success_step is not None:
        category = "reopened"
    elif max_excursion < 5.0:
        category = "no_meaningful_motion"
    elif best - initial_progress >= 0.1:
        category = "partial_closure"
    else:
        category = "timeout"
    suffix = "_hinge_angle" if hinge_mode else "_progress"
    return dict(
        completed=True, success=bool(success), success_timestep=success_step,
        max_steps=max_steps, steps_executed=completed_steps, **{"final" + suffix: final_progress,
        "best" + suffix: best, "initial" + suffix: initial_progress},
        mean_tcp_tracking_error_mm=float(np.mean(position_errors)) if position_errors else None,
        max_tcp_tracking_error_mm=max(position_errors) if position_errors else None,
        mean_tcp_orientation_error_deg=float(np.mean(orientation_errors)) if orientation_errors else None,
        max_tcp_orientation_error_deg=max(orientation_errors) if orientation_errors else None,
        max_tcp_excursion_mm=max_excursion, final_tcp_position=json.dumps(final_tcp[:3, 3].tolist()),
        final_tcp_orientation=json.dumps(Rotation.from_matrix(final_tcp[:3, :3]).as_quat().tolist()),
        policy_inference_count=chunks, number_action_chunks=chunks,
        failure_classification=category, error_stage=error_stage if error else "", error=error,
    )


def run_rollout(policy, group, metadata, seed, args, video_path):
    import numpy as np
    import imageio.v2 as imageio
    from robomimic.utils.rby1_saved_model import create_saved_env
    from robomimic.utils.rby1_cartesian import RBY1CartesianAdapter
    from robomimic.scripts.replay_rby1_cartesian import door_geometry
    env = writer = None
    try:
        # Seed once before environment initialization; never reseed action chunks.
        seed_rollout(seed)
        env, audit = create_saved_env(metadata, group, render=True)
        adapter = RBY1CartesianAdapter(env)
        hinge_name, hinge_address, _ = door_geometry(env, json.loads(group.attrs["ep_meta"]))
        if "microjoint" not in hinge_name or env.behavior != "close":
            raise ValueError("Expected microwave closing task")
        read = observation_reader(env, adapter)
        initial = read()
        for key in LOWDIM_KEYS:
            if not np.allclose(initial[key], group["obs/" + key][0], atol=1e-6, rtol=0):
                raise ValueError(f"Saved initial lowdim observation mismatch: {key}")
        for key in RGB_KEYS:
            from robomimic.utils import obs_utils as O
            expected = O.process_obs(group["obs/" + key][0], obs_key=key)
            if not np.allclose(initial[key], expected, atol=1.0 / 255 + 1e-7, rtol=0):
                raise ValueError(f"Saved initial RGB observation mismatch: {key}")
        if args.save_videos:
            cameras = list(env.sim.model.camera_names)
            if args.video_camera not in cameras:
                raise ValueError(f"Unknown video camera {args.video_camera!r}; available: {cameras}")
            writer = imageio.get_writer(str(video_path), fps=env.control_freq)
        policy.start_episode()
        # Verify that native queued plans actually invoke the EMA encoder.
        calls = [0]
        hook = policy.policy.ema.averaged_model["policy"]["obs_encoder"].register_forward_hook(
            lambda *unused: calls.__setitem__(0, calls[0] + 1))
        try:
            result = execute_loop(policy, env, adapter, read, hinge_address, args.max_steps, writer, args.video_camera)
        finally:
            hook.remove()
        result["policy_inference_count"] = calls[0]
        if not result["error"] and calls[0] != result["number_action_chunks"]:
            raise ValueError("Native action chunks did not use the EMA observation encoder")
        return result, audit
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()


def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_steps < 1 or any(epoch < 1 for epoch in args.epochs):
        raise ValueError("Epochs and max-steps must be positive")
    if len(set(args.epochs)) != len(args.epochs) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Epochs and seeds must be unique")
    import h5py
    dataset_path = Path(args.dataset).expanduser().resolve()
    run = Path(args.run_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output == run or run in output.parents or output == dataset_path.parent:
        raise ValueError("Evaluation output must be outside the training run and source dataset directory")
    with h5py.File(dataset_path, "r") as dataset:
        demos = select_demos(dataset, args.filter_key, args.demo_keys, args.num_demos)
        metadata = json.loads(dataset["data"].attrs["env_args"])
    protocol = dict(schema_version=SCHEMA_VERSION, run_dir=str(run), dataset=fingerprint(dataset_path),
                    split=args.filter_key, max_steps=args.max_steps, success_threshold_rad=SUCCESS_THRESHOLD_RAD,
                    success_definition="final hinge meets threshold at horizon; success_timestep is first reach, 1-based post-step (0 initially)",
                    seed_strategy="once before saved-environment initialization; normal native queue, no chunk reseeding",
                    observations=list(RGB_KEYS + LOWDIM_KEYS), device=args.device,
                    save_videos=args.save_videos, video_camera=args.video_camera,
                    evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    shared_source_sha256={name: hashlib.sha256(
                        (Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                        for name in ("utils/rby1_cartesian.py", "utils/rby1_saved_model.py",
                                     "algo/diffusion_policy.py")})
    checkpoints = {epoch: run / "models" / f"model_epoch_{epoch}.pth" for epoch in args.epochs}
    missing = [epoch for epoch, path in checkpoints.items() if not path.is_file()]
    if missing and not args.skip_missing_checkpoints:
        raise FileNotFoundError(f"Checkpoints not saved yet: {missing}; no rollouts started")
    if output.exists() and not args.resume:
        raise FileExistsError("Use a new output directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "metadata.json"
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text())["protocol"] != protocol:
            raise ValueError("Resume protocol/input provenance mismatch; use a new output directory")
    rows, invalid = read_completed_rows(output / "rollouts.csv")
    if invalid:
        print(f"Ignoring {invalid} invalid/incomplete rows; they will be rerun")
    manifest = dict(protocol=protocol, requested_epochs=args.epochs, demo_keys=demos, seeds=args.seeds,
                    missing_epochs=missing, planned_rollouts=len(args.epochs) * len(demos) * len(args.seeds))
    atomic_json(manifest_path, manifest)
    completed = {row_key(row): row for row in rows}
    with h5py.File(dataset_path, "r") as dataset:
        for epoch, checkpoint_path in checkpoints.items():
            if epoch in missing:
                print(f"PENDING epoch {epoch}: checkpoint not saved")
                continue
            identity = fingerprint(checkpoint_path)
            pending = []
            for demo in demos:
                for seed in args.seeds:
                    key = epoch, demo, seed
                    previous = completed.get(key)
                    if previous:
                        if previous["split"] != args.filter_key or int(previous["max_steps"]) != args.max_steps:
                            raise ValueError(f"Completed row protocol mismatch for {key}")
                        if (previous["checkpoint_path"] != identity["path"] or
                                int(previous["checkpoint_size"]) != identity["size"] or
                                int(previous["checkpoint_mtime_ns"]) != identity["mtime_ns"]):
                            raise ValueError(f"Checkpoint changed for completed rollout {key}; use a new output directory")
                        print(f"SKIP completed {key}")
                    else:
                        pending.append((demo, seed))
            if not pending:
                continue
            # No heavy imports for --help, missing checkpoints or fully resumed runs.
            import torch
            from robomimic.utils import file_utils as F
            policy, checkpoint = F.policy_from_checkpoint(ckpt_path=str(checkpoint_path), device=torch.device(args.device), verbose=False)
            check_policy(policy, checkpoint, metadata)
            for demo, seed in pending:
                start = time.perf_counter()
                stem = f"epoch_{epoch}_{demo}_seed_{seed}"
                video = output / (stem + ".mp4")
                row = {field: None for field in FIELDS}
                row.update(checkpoint_epoch=epoch, checkpoint_path=identity["path"], checkpoint_size=identity["size"],
                           checkpoint_mtime_ns=identity["mtime_ns"], demo_key=demo, split=args.filter_key, seed=seed,
                           video_path=str(video) if args.save_videos else "")
                try:
                    result, audit = run_rollout(policy, dataset["data/" + demo], metadata, seed, args, video)
                    row.update(result)
                    atomic_json(output / (stem + "_initialization.json"), audit)
                except Exception as exc:
                    row.update(completed=True, success=False, max_steps=args.max_steps, steps_executed=0,
                               failure_classification="runtime_error", error_stage="initialization_or_cleanup",
                               error=f"{type(exc).__name__}: {exc}")
                row["wall_seconds"] = time.perf_counter() - start
                completed[row_key(row)] = row
                rows = list(completed.values())
                write_csv(output / "rollouts.csv", rows, FIELDS)
                print(json.dumps(row, allow_nan=False), flush=True)
            del policy, checkpoint
    print(f"Saved {len(completed)} completed rollouts; pending checkpoints: {missing}")


if __name__ == "__main__":
    main()
