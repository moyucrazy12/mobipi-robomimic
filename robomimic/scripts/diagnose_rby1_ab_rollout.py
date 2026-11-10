"""Bounded saved-start A/B diagnostic without modifying policy or control settings."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from robomimic.utils.rby1_paths import new_output

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from robomimic.scripts.eval_rby1_cartesian_dp import RGB, LOWDIM, check_checkpoint, seed_all
from robomimic.utils import file_utils as F
from robomimic.utils.rby1_cartesian import RBY1CartesianAdapter


def pose(transform):
    return {"xyz_world_m": transform[:3, 3].tolist(),
            "quat_world_xyzw": Rotation.from_matrix(transform[:3, :3]).as_quat().tolist()}


def angle(first, second):
    return float(np.rad2deg(Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).magnitude()))


def left_bodies(raw):
    model = raw.sim.model._model
    joint_name = raw.robots[0].composite_controller.part_controllers["left"].joint_names[0]
    root = int(model.joint(joint_name).bodyid)
    selected = set()
    for body in range(model.nbody):
        parent = body
        while parent:
            if parent == root:
                selected.add(body)
                break
            parent = int(model.body_parentid[parent])
    return selected


def contacts(raw, selected):
    model, data = raw.sim.model._model, raw.sim.data._data
    found = []
    for contact in data.contact[:data.ncon]:
        geoms = [int(contact.geom1), int(contact.geom2)]
        bodies = [int(model.geom_bodyid[g]) for g in geoms]
        if not any(b in selected for b in bodies):
            continue
        body_names = [model.body(b).name for b in bodies]
        geom_names = [model.geom(g).name for g in geoms]
        other = [body_names[i] + " " + geom_names[i] for i in range(2) if bodies[i] not in selected]
        names = " ".join(other).lower()
        categories = []
        for category, keywords in {"microwave": ("microwave",),
                                   "oven": ("oven", "stove", "cooktop"),
                                   "counter": ("counter",)}.items():
            if any(word in names for word in keywords):
                categories.append(category)
        found.append({"body_names": body_names, "geom_names": geom_names,
                      "distance_m": float(contact.dist), "categories": categories,
                      "sim_time_s": float(data.time)})
    return found


def run(args):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    with h5py.File(Path(args.dataset).expanduser(), "r") as dataset:
        assert args.demo in [x.decode() for x in dataset["mask/valid"][:]]
        group = dataset[f"data/{args.demo}"]
        saved = {"model": group.attrs["model_file"], "ep_meta": group.attrs["ep_meta"],
                 "states": group["states"][0]}
        truth = group["actions_eef"][:]
        source_controller = json.loads(dataset["data"].attrs["env_args"])["env_kwargs"]["controller_configs"]
    policy, checkpoint = F.policy_from_checkpoint(
        ckpt_path=args.checkpoint, device=torch.device(args.device))
    check_checkpoint(policy, checkpoint)
    assert checkpoint["env_metadata"]["env_kwargs"]["controller_configs"] == source_controller
    checkpoint["env_metadata"]["env_kwargs"]["seed"] = args.seed
    summary = {"checkpoint": str(Path(args.checkpoint).resolve()), "demo": args.demo,
               "seed": args.seed, "saved_initial_state_sha256": hashlib.sha256(saved["states"].tobytes()).hexdigest(),
               "base_offset_m": 0, "original_controller": source_controller, "action_dim": 31,
               "contact_sampling": "All physics substeps plus post-control-step contacts",
               "distance_measure": "Minimum signed active-contact distance; no separation query when no contact",
               "persistent_collision_definition": "Fixture contact with distance <= 0 in five consecutive control steps",
               "blocked_proxy_definition": "Five consecutive fixture-contact steps, TCP displacement < 2 mm across the window, and position error > 10 mm or orientation error > 5 degrees",
               "first_collision_comparison_note": "Same timestep index, not matched task phase after trajectory divergence",
               "runs": {}}
    initial_observations = None
    for label, limit in (("A_ground_truth", args.gt_steps), ("B_policy", args.policy_max_steps)):
        seed_all(args.seed)
        env, _ = F.env_from_checkpoint(ckpt_dict=checkpoint, render_offscreen=True)
        raw = env.unwrapped.env
        rows = []
        try:
            ob = env.reset_to(saved)
            np.testing.assert_array_equal(raw.sim.get_state().flatten(), saved["states"])
            if initial_observations is None:
                initial_observations = {k: ob[k].copy() for k in RGB + LOWDIM}
            else:
                for key in RGB + LOWDIM:
                    np.testing.assert_allclose(ob[key], initial_observations[key], atol=1e-7, rtol=0)
            adapter = RBY1CartesianAdapter(raw)
            assert raw.action_dim == 31 and raw.control_freq == 20
            base0 = adapter.site_transform(adapter.base_site).copy()
            np.testing.assert_allclose(base0[:3, 3], [.8, -1.45, 0], atol=1e-8)
            selected = left_bodies(raw)
            step_contacts = {}
            # These hooks only observe contacts after delegating the original
            # physics calls. Integration, controller and collision masks stay intact.
            def observe(original):
                def wrapped(*positional, **kwargs):
                    result = original(*positional, **kwargs)
                    for c in contacts(raw, selected):
                        key = tuple(c["geom_names"])
                        if key not in step_contacts or c["distance_m"] < step_contacts[key]["distance_m"]:
                            step_contacts[key] = c
                    return result
                return wrapped
            for method in ("step", "step2"):
                setattr(raw.sim, method, observe(getattr(raw.sim, method)))
            policy.start_episode()
            first_collision = first_persistent = first_blocked = None
            stop_at = None
            for timestep in range(min(limit, len(truth))):
                before = adapter.site_transform(adapter.tcp_site)
                action = truth[timestep].copy() if label == "A_ground_truth" else policy({k: ob[k] for k in RGB + LOWDIM})
                desired = adapter.target_world(action)
                gt_target = adapter.target_world(truth[timestep])
                step_contacts.clear()
                ob, reward, done, info = env.step(adapter(action))
                actual = adapter.site_transform(adapter.tcp_site)
                np.testing.assert_allclose(adapter.site_transform(adapter.base_site), base0, atol=1e-8, rtol=0)
                for c in contacts(raw, selected):
                    key = tuple(c["geom_names"])
                    if key not in step_contacts or c["distance_m"] < step_contacts[key]["distance_m"]:
                        step_contacts[key] = c
                collision = any(c["categories"] and c["distance_m"] <= 0 for c in step_contacts.values())
                minimum = {category: min((c["distance_m"] for c in step_contacts.values() if category in c["categories"]), default=None)
                           for category in ("microwave", "oven", "counter")}
                row = {"timestep": timestep, "action_9d_base": np.asarray(action).tolist(),
                       "desired_tcp": pose(desired), "actual_tcp": pose(actual), "tcp_before_action": pose(before),
                       "ground_truth_target": pose(gt_target),
                       "tracking_position_error_m": float(np.linalg.norm(desired[:3, 3] - actual[:3, 3])),
                       "tracking_orientation_error_deg": angle(desired, actual),
                       "target_minus_demo_xyz_world_m": (desired[:3, 3] - gt_target[:3, 3]).tolist(),
                       "target_demo_position_difference_m": float(np.linalg.norm(desired[:3, 3] - gt_target[:3, 3])),
                       "target_demo_orientation_difference_deg": angle(desired, gt_target),
                       "contacts_left_arm_gripper": list(step_contacts.values()),
                       "minimum_signed_contact_distance_m": minimum, "fixture_collision": collision,
                       "task_state": raw.door_fxtr.get_door_state(env=raw), "task_success": bool(env.is_success()["task"]),
                       "reward": float(reward), "done": bool(done),
                       "queue_remaining": len(policy.policy.action_queue) if label == "B_policy" else None,
                       "blocked_proxy": False}
                rows.append(row)
                if collision and first_collision is None:
                    first_collision = timestep
                if len(rows) >= 5 and all(r["fixture_collision"] for r in rows[-5:]):
                    if first_persistent is None:
                        first_persistent = timestep - 4
                        if label == "B_policy": stop_at = timestep + 10
                    movement = np.linalg.norm(actual[:3, 3] - np.asarray(rows[-5]["actual_tcp"]["xyz_world_m"]))
                    if movement < .002 and (row["tracking_position_error_m"] > .01 or row["tracking_orientation_error_deg"] > 5):
                        row["blocked_proxy"] = True
                        if first_blocked is None: first_blocked = timestep - 4
                if timestep % 25 == 0 or collision:
                    print(f'{label} step {timestep}: error={row["tracking_position_error_m"]*1000:.2f} mm, fixture_collision={collision}', flush=True)
                if done or row["task_success"] or (stop_at is not None and timestep >= stop_at):
                    break
            result = {"steps": len(rows), "initial_state_exact": True, "initial_observations_match": True,
                      "base_world_transform": base0.tolist(),
                      "first_fixture_collision_timestep": first_collision,
                      "first_persistent_collision_timestep": first_persistent,
                      "first_blocked_motion_proxy_timestep": first_blocked,
                      "position_error_mean_m": float(np.mean([r["tracking_position_error_m"] for r in rows])),
                      "position_error_max_m": max(r["tracking_position_error_m"] for r in rows),
                      "orientation_error_mean_deg": float(np.mean([r["tracking_orientation_error_deg"] for r in rows])),
                      "stop_reason": "persistent_collision_plus_10_steps" if stop_at is not None and rows[-1]["timestep"] >= stop_at else
                                     "success_or_done" if rows[-1]["task_success"] or rows[-1]["done"] else "step_cap",
                      "final_task_state": rows[-1]["task_state"]}
            if first_collision is not None:
                result["first_collision_comparison"] = rows[first_collision]
            (output / f"{label}.json").write_text(json.dumps(rows, indent=2))
            with (output / f"{label}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
                writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)
            summary["runs"][label] = result
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(result, indent=2), flush=True)
        finally:
            raw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="~/Downloads/demo_im128_head_wrist_cartesian.hdf5")
    parser.add_argument("--demo", default="demo_5")
    parser.add_argument("--output", default=str(new_output("CloseSingleDoor", "diagnostics", "rby1_ab_rollout")))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gt-steps", type=int, default=200)
    parser.add_argument("--policy-max-steps", type=int, default=600)
    args = parser.parse_args()
    if not 100 <= args.gt_steps <= 200 or not 1 <= args.policy_max_steps <= 600:
        parser.error("Ground-truth window must be 100-200 steps; policy cap must be 1-600")
    run(args)
