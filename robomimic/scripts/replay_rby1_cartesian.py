"""Fixed-base ground-truth replay through RB-Y1 Cartesian IK."""

import argparse
import copy
import csv
import json
from pathlib import Path

from robomimic.utils.rby1_paths import new_output

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from robomimic.utils.rby1_cartesian import RBY1CartesianAdapter, make_transform, source_command_as_cartesian


def controller_snapshot(env):
    """Record the instantiated controller, including both packed action maps."""
    robot = env.robots[0]
    composite = robot.composite_controller
    ik = composite.joint_action_policy
    return {
        "robot_action_dim": robot.action_dim, "env_action_dim": env.action_dim,
        "action_spec_shape": list(env.action_spec[0].shape),
        "controller_config": robot.composite_controller_config,
        "specific_config": composite.composite_controller_specific_config,
        "joint_controller_slices": composite._action_split_indexes,
        "environment_action_slices": composite._whole_body_controller_action_split_indexes,
        "ik_action_slices": ik.action_split_indexes(), "ik_control_dim": ik.control_dim,
        "ik_joint_names": ik.joint_names, "ik_site_names": ik.site_names,
        "ik_input_action_repr": ik.input_action_repr, "ik_input_rotation_repr": ik.input_rotation_repr,
        "parts": {name: {"type": part.name, "control_dim": part.control_dim,
                         "joint_names": getattr(part, "joint_names", []),
                         "input_type": getattr(part, "input_type", None)}
                  for name, part in composite.part_controllers.items()},
        "control_freq": env.control_freq, "control_timestep": env.control_timestep,
        "model_timestep": env.model_timestep,
    }


def diagnostic_env(metadata, controller_config, xml, episode_metadata):
    import robosuite

    from robomimic.utils.rby1_saved_model import with_saved_base

    kwargs = with_saved_base(copy.deepcopy(metadata["env_kwargs"]), xml)
    kwargs.update(controller_configs=copy.deepcopy(controller_config), has_renderer=False,
                  has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
    env = robosuite.make(metadata["env_name"], **kwargs)
    env.set_ep_meta(episode_metadata)
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(xml))
    return env


def restore_sample(env, state, q0, gripper_actions):
    """Clear simulator/controller history before every independent trial.

    HDF5 states contain time/qpos/qvel/act, not controller history or solver
    warm-start buffers. Reset these consistently for source and adapter trials.
    """
    env.sim.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    env.timestep = 0
    env.cur_time = float(state[0])
    env.done = False
    composite = env.robots[0].composite_controller
    for part in composite.part_controllers.values():
        part.update(force=True)
        part.reset_goal()
    composite.joint_action_policy.q0 = q0.copy()
    for arm, value in gripper_actions.items():
        env.robots[0].gripper[arm].current_action = value.copy()


def door_geometry(env, episode_metadata):
    """Find the task door hinge and its descendants for robot/door contact."""
    model = env.sim.model._model
    fixture = episode_metadata["fixture_refs"]["door_fxtr"]
    import mujoco

    candidates = [j for j in range(model.njnt) if fixture in model.joint(j).name
                  and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous door hinge candidates: {[model.joint(j).name for j in candidates]}")
    joint = candidates[0]
    root = int(model.jnt_bodyid[joint])
    descendants = set()
    for body in range(model.nbody):
        ancestor = body
        while ancestor:
            if ancestor == root:
                descendants.add(body)
                break
            ancestor = int(model.body_parentid[ancestor])
    return model.joint(joint).name, int(model.jnt_qposadr[joint]), descendants


def door_contact(env, door_bodies):
    model, data = env.sim.model._model, env.sim.data._data
    count = 0
    for contact in data.contact[:data.ncon]:
        bodies = [int(model.geom_bodyid[int(g)]) for g in (contact.geom1, contact.geom2)]
        names = [model.body(b).name for b in bodies]
        robot = [name.startswith(("robot0_", "gripper0_")) for name in names]
        if (bodies[0] in door_bodies and robot[1]) or (bodies[1] in door_bodies and robot[0]):
            count += 1
    return count


def robot_scene_contacts(env):
    """Report robot/environment contacts; door-touch contacts may be intentional."""
    model, data = env.sim.model._model, env.sim.data._data
    contacts = []
    for contact in data.contact[:data.ncon]:
        geoms = [model.geom(int(g)) for g in (contact.geom1, contact.geom2)]
        names = [model.body(int(g.bodyid)).name for g in geoms]
        robot = [name.startswith(("robot0_", "gripper0_")) for name in names]
        if sum(robot) == 1:
            contacts.append({"bodies": names, "geoms": [g.name for g in geoms],
                             "distance_m": float(contact.dist)})
    return contacts


def summarize_trials(rows):
    summary = {}
    for configuration in sorted({r["configuration"] for r in rows}):
        summary[configuration] = {}
        for method in sorted({r["method"] for r in rows if r["configuration"] == configuration}):
            summary[configuration][method] = {}
            for repeats in (1, 2, 4):
                by_region = {}
                for region in ("all", "free_space", "contact"):
                    selected = [r for r in rows if r["configuration"] == configuration and r["method"] == method
                                and r["repeats"] == repeats and (region == "all" or r["region"] == region)]
                    if not selected:
                        continue
                    by_region[region] = {"count": len(selected)}
                    for key in ("position_error_m", "orientation_error_deg"):
                        values = np.asarray([r[key] for r in selected])
                        by_region[region][key] = {"mean": float(values.mean()), "median": float(np.median(values)),
                                                  "max": float(values.max())}
                summary[configuration][method][str(repeats)] = by_region
    return summary


def teacher_forced(args, metadata, replay_config, xml, episode_metadata, states,
                   source_actions, cartesian_actions, world_positions, world_quaternions):
    """Compare original and Cartesian commands from exactly the same saved states."""
    from robomimic.utils.rby1_cartesian import make_transform, quaternion_to_rotation6d

    source_config = metadata["env_kwargs"]["controller_configs"]
    rows, snapshots, transform_errors = [], {}, []
    samples, repeat_samples, regions = None, None, {}
    report = {"source_controller_metadata": source_config,
              "reset_protocol": "sim.reset + saved time/qpos/qvel/act; force controller update/reset goals; fixed initial q0 and gripper history",
              "limitation": "Dataset does not save controller history or MuJoCo solver warm-start buffers",
              "region_definition": "robot/door-panel contact at recorded state t or t+1"}
    configurations = [("source_exact", source_config)]
    if args.controller:
        configurations.append(("replay_left_only", replay_config))
    for label, config in configurations:
        env = diagnostic_env(metadata, config, xml, episode_metadata)
        try:
            robot = env.robots[0]
            env.sim.reset()
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()
            initial_adapter = RBY1CartesianAdapter(env)
            q0 = robot.composite_controller.joint_action_policy.q0.copy()
            gripper_actions = {arm: robot.gripper[arm].current_action.copy() for arm in robot.arms}
            snapshots[label] = controller_snapshot(env)
            snapshots[label]["q0"] = q0.tolist()
            hinge_name, hinge_index, door_bodies = door_geometry(env, episode_metadata)
            report["hinge_name"] = hinge_name
            if samples is None:
                # Scanning FK/contact is cheap; no trajectory is stepped here.
                contacts, hinge_angles = [], []
                for state in states:
                    env.sim.set_state_from_flattened(state)
                    env.sim.forward()
                    contacts.append(door_contact(env, door_bodies))
                    hinge_angles.append(float(env.sim.data.qpos[hinge_index]))
                report["recorded_contact_timesteps"] = int(np.count_nonzero(contacts))
                changes = np.flatnonzero(np.diff(np.asarray(contacts) > 0)) + 1
                extra = set()
                # Include first/last contact boundaries and fastest door motion.
                boundaries = list(changes[:3]) + list(changes[-3:])
                if np.any(contacts):
                    boundaries += [int(np.flatnonzero(contacts)[0]), int(np.flatnonzero(contacts)[-1])]
                boundaries += np.argsort(np.abs(np.diff(hinge_angles)))[-3:].tolist()
                for t in boundaries:
                    extra.update(range(max(0, t - 2), min(len(states) - 1, t + 3)))
                samples = sorted(set(np.linspace(0, len(states) - 2, args.samples, dtype=int)) | extra)
                repeat_samples = set(samples[::max(1, len(samples) // 12)]) | extra
                regions = {t: "contact" if contacts[t] or contacts[t + 1] else "free_space" for t in samples}
                report.update(samples=samples, repeat_samples=sorted(repeat_samples),
                              contact_boundaries=changes.tolist(), recorded_hinge_range_rad=[min(hinge_angles), max(hinge_angles)])
            for t in samples:
                target = make_transform(world_positions[t + 1], Rotation.from_quat(world_quaternions[t + 1]).as_matrix())
                methods = ("source_action", "cartesian_adapter") if label == "source_exact" else ("cartesian_adapter",)
                if args.command_roundtrip_check and label == "source_exact":
                    methods += ("source_command_via_adapter",)
                for method in methods:
                    repetitions = (1, 2, 4) if t in repeat_samples and method != "source_command_via_adapter" else (1,)
                    for repeats in repetitions:
                        restore_sample(env, states[t], q0, gripper_actions)
                        adapter = RBY1CartesianAdapter(env)
                        # Capture holds from this sample but use identical initial
                        # nullspace reference in both source and adapter trials.
                        robot.composite_controller.joint_action_policy.q0 = q0.copy()
                        before_contact = door_contact(env, door_bodies)
                        hinge_before = float(env.sim.data.qpos[hinge_index])
                        base_before = adapter.site_transform(adapter.base_site)
                        adapter_action = cartesian_actions[t]
                        if method == "source_command_via_adapter":
                            # Additional control: round-trip the ORIGINAL IK
                            # command through the same 9D TCP adapter. This does
                            # not change actions_eef or the policy action format.
                            adapter_action = source_command_as_cartesian(adapter, source_actions[t])
                            # Preserve the recorded nuisance commands in this
                            # round-trip control, so only pose conversion and
                            # packing are compared with the source baseline.
                            slices = robot.composite_controller._whole_body_controller_action_split_indexes
                            adapter.holds = {part: source_actions[t, slice(*slices[part])].copy()
                                             for part in adapter.holds}
                        if method == "cartesian_adapter":
                            transformed = adapter.target_world(cartesian_actions[t])
                            transform_errors.append(float(np.max(np.abs(transformed - target))))
                            if not np.allclose(transformed, target, atol=2e-6):
                                raise ValueError(f"Target frame mismatch at {t}")
                        for _ in range(repeats):
                            packed = source_actions[t] if method == "source_action" else adapter(adapter_action)
                            if packed.shape != (env.action_dim,):
                                raise ValueError(f"Action dimension mismatch: {packed.shape} versus {env.action_dim}")
                            env.step(packed)
                        actual = adapter.site_transform(adapter.tcp_site)
                        finite = bool(np.isfinite(env.sim.data.qpos).all() and np.isfinite(env.sim.data.qvel).all())
                        if not finite:
                            raise ValueError(f"Non-finite diagnostic at {label}/{method}/{t}")
                        rows.append({"configuration": label, "method": method, "timestep": int(t), "target_timestep": int(t + 1),
                                     "repeats": repeats, "region": regions[t], "contact_before": before_contact,
                                     "contact_after": door_contact(env, door_bodies), "hinge_before_rad": hinge_before,
                                     "hinge_after_rad": float(env.sim.data.qpos[hinge_index]),
                                     "position_error_m": float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
                                     "orientation_error_deg": float(np.rad2deg(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude())),
                                     "base_transform_drift": float(np.max(np.abs(adapter.site_transform(adapter.base_site) - base_before))),
                                     "packed_source_max_difference": (float(np.max(np.abs(packed - source_actions[t])))
                                                                      if method == "source_command_via_adapter" else None),
                                     "action_dim": int(packed.size), "finite": finite})
                print(f"{label}: sample {t}, region={regions[t]}", flush=True)
        finally:
            env.close()
    report.update(controllers=snapshots, target_transform_max_element_error=max(transform_errors),
                  summary=summarize_trials(rows),
                  repeated_sample_summary=summarize_trials([row for row in rows if row["timestep"] in repeat_samples]),
                  trials=rows)
    output = Path(args.report).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    output.write_text(json.dumps(report, indent=2, default=encode) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["summary"], indent=2))
    print(f"Diagnostic report: {output}; per-sample table: {output.with_suffix('.csv')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--controller", help="Optional established config override with left-only IK refs; otherwise use unchanged dataset config")
    parser.add_argument("--steps", type=int, help="Optional step limit")
    parser.add_argument("--method", choices=("cartesian", "source", "commanded-cartesian"), default="cartesian",
                        help="Replay measured next-state poses, original commands, or commands encoded as 9D TCP targets")
    parser.add_argument("--action-key", default="actions_eef",
                        choices=("actions_eef", "actions_eef_cmd"), help="Stored 9D targets for --method cartesian")
    parser.add_argument("--recorded-holds", action="store_true",
                        help="Diagnostic: preserve recorded non-left commands to isolate left-pose conversion")
    parser.add_argument("--render", action="store_true", help="Open the interactive MuJoCo viewer")
    parser.add_argument("--output", default=str(new_output("CloseSingleDoor", "diagnostics", "rby1_replay")), help="New directory for replay summary and per-step errors/contacts")
    parser.add_argument("--check-source-poses", action="store_true", help="Check transform chain against five saved simulator states before replay")
    parser.add_argument("--teacher-forced", action="store_true", help="Independent saved-state source/Cartesian diagnostics; no open-loop replay")
    parser.add_argument("--samples", type=int, default=64, help="Evenly spaced teacher-forced samples plus contact/motion boundaries")
    parser.add_argument("--report", default=str(new_output("CloseSingleDoor", "diagnostics", "rby1_teacher_forced", ".json")))
    parser.add_argument("--command-roundtrip-check", action="store_true", help="Additionally round-trip the original IK command through the Cartesian adapter")
    args = parser.parse_args()
    if args.teacher_forced and args.action_key != "actions_eef":
        raise ValueError("The achieved-state teacher-forced diagnostic requires --action-key actions_eef")
    import robosuite
    import robocasa  # noqa: F401: register kitchen environments
    from robosuite.controllers import load_composite_controller_config

    with h5py.File(Path(args.dataset).expanduser(), "r") as dataset:
        metadata = json.loads(dataset["data"].attrs["env_args"])
        demo = dataset[f"data/{args.demo}"]
        source_actions = demo["actions"][:]
        world_positions = demo["obs/robot0_left_eef_pos"][:]
        world_quaternions = demo["obs/robot0_left_eef_quat_site"][:]
        states = demo["states"][:]
        initial_state = states[0]
        source_position = demo["obs/robot0_base_to_left_eef_pos"][:]
        source_quaternion = demo["obs/robot0_base_to_left_eef_quat_site"][:]
        xml = demo.attrs["model_file"]
        episode_metadata = json.loads(demo.attrs["ep_meta"])
        if args.method == "cartesian" or args.teacher_forced:
            stored = demo[args.action_key]
            if args.action_key == "actions_eef":
                if stored.attrs["quaternion_key"] != "robot0_base_to_left_eef_quat_site":
                    raise ValueError("Adapter requires actual grip-site orientation targets")
            elif stored.attrs["tcp_site"] != "gripper0_left_grip_site":
                raise ValueError("Unexpected command TCP site")
            actions = stored[:]
        else:
            actions = source_actions  # only the length is needed for command replay
    kwargs = copy.deepcopy(metadata["env_kwargs"])
    # Prefer the exact recorded configuration. An explicit override reproduces
    # the previous 25D replay for diagnostics, without modifying any asset file.
    config = copy.deepcopy(metadata["env_kwargs"]["controller_configs"])
    if args.controller:
        config_path = Path(args.controller).expanduser()
        config = load_composite_controller_config(controller=str(config_path))
        specific = config["composite_controller_specific_configs"]
        references = dict(zip(specific["actuation_part_names"], specific["ref_name"]))
        specific["actuation_part_names"] = ["left"]
        specific["ref_name"] = [references["left"]]
    if args.teacher_forced:
        if args.samples < 2:
            raise ValueError("At least two samples required")
        teacher_forced(args, metadata, config, xml, episode_metadata, states, source_actions,
                       actions, world_positions, world_quaternions)
        return
    output = Path(args.output).expanduser() if args.output else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    kwargs.update(controller_configs=config, has_renderer=args.render, has_offscreen_renderer=False,
                  use_camera_obs=False, ignore_done=True)
    if args.render:
        kwargs.update(renderer="mjviewer", render_camera=None)
    env = robosuite.make(metadata["env_name"], **kwargs)
    try:
        env.set_ep_meta(episode_metadata)
        env.reset()
        env.reset_from_xml_string(env.edit_model_xml(xml))
        env.sim.reset()
        env.sim.set_state_from_flattened(initial_state)
        env.sim.forward()
        adapter = RBY1CartesianAdapter(env)
        if args.check_source_poses:
            for t in np.unique(np.linspace(0, len(states) - 1, 5, dtype=int)):
                env.sim.set_state_from_flattened(states[t])
                env.sim.forward()
                predicted_tcp = adapter.site_transform(adapter.base_site) @ make_transform(
                    source_position[t], Rotation.from_quat(source_quaternion[t]).as_matrix())
                predicted_ref = predicted_tcp @ adapter.tcp_to_reference
                actual_ref = adapter.site_transform(adapter.reference_site)
                pos_error = np.linalg.norm(predicted_ref[:3, 3] - actual_ref[:3, 3])
                rot_error = np.rad2deg(Rotation.from_matrix(predicted_ref[:3, :3] @ actual_ref[:3, :3].T).magnitude())
                print(f"source state {t}: reference error {pos_error:.9g} m / {rot_error:.9g} deg", flush=True)
                if pos_error > 1e-6 or rot_error > 1e-4:
                    raise ValueError("Source pose does not agree with controller-reference transform")
            env.sim.set_state_from_flattened(initial_state)
            env.sim.forward()
            adapter = RBY1CartesianAdapter(env)
        if args.render:
            env.viewer.update()
        base_initial = adapter.site_transform(adapter.base_site)
        right_initial = adapter.site_transform(adapter.robot.gripper["right"].important_sites["grip_site"])
        right_controller = adapter.controller.part_controllers["right"]
        right_q_initial = np.asarray(env.sim.data.qpos[right_controller.qpos_index]).copy()
        errors, base_drift, right_drift, right_joint_drift = [], [], [], []
        rows = []
        actual_positions = []
        hinge_name, hinge_index, _ = door_geometry(env, episode_metadata)
        count = min(len(actions), args.steps) if args.steps is not None else len(actions)
        if count < 1:
            raise ValueError("At least one replay step is required")
        for t in range(count):
            action = actions[t]
            next_t = min(t + 1, len(world_positions) - 1)
            target = make_transform(world_positions[next_t], Rotation.from_quat(world_quaternions[next_t]).as_matrix())
            if args.method == "commanded-cartesian":
                # Retain the same fixed holds as policy evaluation; only the
                # left target comes from the original recorded IK command.
                action = source_command_as_cartesian(adapter, source_actions[t])
            if args.recorded_holds:
                slices = adapter.controller._whole_body_controller_action_split_indexes
                adapter.holds = {part: source_actions[t, slice(*slices[part])].copy()
                                 for part in adapter.holds}
            packed = source_actions[t] if args.method == "source" else adapter(action)
            if packed.shape != (env.action_dim,):
                raise ValueError("Replay action dimension does not match unchanged controller")
            env.step(packed)
            if not np.isfinite(env.sim.data.qpos).all() or not np.isfinite(env.sim.data.qvel).all():
                raise ValueError(f"Non-finite simulator state at step {t}")
            actual = adapter.site_transform(adapter.tcp_site)
            actual_positions.append(actual[:3, 3])
            errors.append([np.linalg.norm(actual[:3, 3] - target[:3, 3]),
                           np.rad2deg(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude())])
            base_drift.append(np.max(np.abs(adapter.site_transform(adapter.base_site) - base_initial)))
            right = adapter.site_transform(adapter.robot.gripper["right"].important_sites["grip_site"])
            right_drift.append(np.linalg.norm(right[:3, 3] - right_initial[:3, 3]))
            right_joint_drift.append(np.max(np.abs(env.sim.data.qpos[right_controller.qpos_index] - right_q_initial)))
            contacts = robot_scene_contacts(env)
            rows.append({"timestep": t, "target_xyz_world": json.dumps(target[:3, 3].tolist()),
                         "actual_xyz_world": json.dumps(actual[:3, 3].tolist()),
                         "position_error_m": float(errors[-1][0]), "orientation_error_deg": float(errors[-1][1]),
                         "base_drift": float(base_drift[-1]), "scene_contact_count": len(contacts),
                         "scene_contacts": json.dumps(contacts), "success": bool(env._check_success())})
            if t % 250 == 0:
                print(f"step {t}/{count}: position={errors[-1][0]:.6f} m, rotation={errors[-1][1]:.4f} deg", flush=True)
        errors = np.asarray(errors)
        summary = {"demo": args.demo, "method": args.method, "steps": count,
                          "action_key": args.action_key if args.method == "cartesian" else "actions",
                          "recorded_holds": args.recorded_holds,
                          "dataset": str(Path(args.dataset).expanduser().resolve()),
                          "error_reference": "Recorded next-state TCP pose for all replay methods",
                          "base_world_transform": base_initial.tolist(),
                          "position_mean_m": float(errors[:, 0].mean()),
                          "position_max_m": float(errors[:, 0].max()), "orientation_mean_deg": float(errors[:, 1].mean()),
                          "orientation_max_deg": float(errors[:, 1].max()), "base_transform_max_drift": float(max(base_drift)),
                          "right_tcp_max_drift_m": float(max(right_drift)), "right_joint_max_drift_rad": float(max(right_joint_drift)),
                          "left_tcp_xyz_span_m": np.ptp(actual_positions, axis=0).tolist(),
                          "action_dim": int(packed.size), "finite": True, "tcp_site": adapter.tcp_site,
                          "reference_site": adapter.reference_site, "base_site": adapter.base_site,
                          "final_task_state": {"door_joint": hinge_name,
                                               "door_angle_rad": float(env.sim.data.qpos[hinge_index]),
                                               "success": bool(env._check_success())},
                          "tcp_to_reference": adapter.tcp_to_reference.tolist(), "success": bool(env._check_success()),
                          "scene_contact_steps": sum(r["scene_contact_count"] > 0 for r in rows),
                          "first_scene_contact_step": next((r["timestep"] for r in rows if r["scene_contact_count"]), None),
                          "limitation": ("Open-loop diagnostic; measured next-state TCP targets differ from original controller commands. Not a training gate."
                                         if args.method == "cartesian" and args.action_key == "actions_eef" else
                                         "Full command replay; path errors compare actual TCP with recorded next achieved TCP, not with the command target.")}
        if output is not None:
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            with (output / "trajectory.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
