"""RB-Y1 Cartesian pose conversions, using explicit xyzw quaternions."""

import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_to_rotation6d(quaternion):
    """Encode R as [R[:, 0], R[:, 1]] (column vectors concatenated).

    Accepts (..., 4) xyzw quaternions and returns (..., 6).
    """
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1] != 4 or not np.isfinite(quaternion).all():
        raise ValueError("Expected finite xyzw quaternions with final dimension 4")
    if np.any(np.linalg.norm(quaternion, axis=-1) < 1e-12):
        raise ValueError("Zero quaternion")
    matrix = Rotation.from_quat(quaternion.reshape(-1, 4)).as_matrix()
    encoded = np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)
    return encoded.reshape(quaternion.shape[:-1] + (6,))


def rotation6d_to_matrix(rotation6d):
    """Decode concatenated columns using Gram-Schmidt; reject degenerate inputs."""
    rotation6d = np.asarray(rotation6d, dtype=np.float64)
    if rotation6d.shape[-1] != 6 or not np.isfinite(rotation6d).all():
        raise ValueError("Expected finite rotation6d with final dimension 6")
    first, second = rotation6d[..., :3], rotation6d[..., 3:]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("Degenerate first rotation column")
    first = first / norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("Collinear rotation columns")
    second = second / norm
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def make_transform(position, rotation):
    """Build a homogeneous transform for a single pose."""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def source_command_as_cartesian(adapter, source_action):
    """Encode the same-timestep recorded world IK command as a base TCP pose.

    T_base_tcp_cmd = inv(T_world_base) @ T_world_ref_cmd @ inv(T_tcp_ref).
    Resolve the left slot from the instantiated original controller, including
    its otherwise unused reference slots. No achieved TCP target is used.
    """
    source_action = np.asarray(source_action, dtype=np.float64)
    if source_action.shape != (adapter.env.action_dim,) or not np.isfinite(source_action).all():
        raise ValueError("Expected one finite original environment action")
    start, end = adapter.controller.joint_action_policy.action_split_indexes()["left"]
    if (start, end) != tuple(adapter.controller._whole_body_controller_action_split_indexes["left"]):
        raise ValueError("IK and packed left-action slices disagree")
    reference = source_action[start:end]
    world_reference = make_transform(reference[:3], Rotation.from_rotvec(reference[3:]).as_matrix())
    world_tcp = world_reference @ np.linalg.inv(adapter.tcp_to_reference)
    base_tcp = np.linalg.inv(adapter.site_transform(adapter.base_site)) @ world_tcp
    return np.concatenate((base_tcp[:3, 3], quaternion_to_rotation6d(
        Rotation.from_matrix(base_tcp[:3, :3]).as_quat())))


class RBY1CartesianAdapter:
    """Convert one left TCP target into the existing WholeBody action interface.

    Construct after every reset. Holds are captured from the restored state.
    Policy chunk scheduling stays with Robomimic's RolloutPolicy.
    """

    def __init__(self, env):
        from robosuite.controllers.composite.composite_controller import WholeBody

        self.env = env
        self.robot = env.robots[0]
        self.controller = self.robot.composite_controller
        if not isinstance(self.controller, WholeBody):
            raise ValueError("Expected WholeBody IK")
        config = self.controller.composite_controller_specific_config
        parts = config["actuation_part_names"]
        if "left" not in parts:
            raise ValueError("IK actuation parts must include left")
        ik = self.controller.joint_action_policy
        if ik.input_rotation_repr != "axis_angle" or ik.input_action_repr != "absolute":
            raise ValueError("Expected absolute world-frame position + axis-angle IK targets")
        # The recorded configuration has two reference sites but only left-arm
        # IK joints. Resolve references from the solver's actual input slices,
        # rather than zipping unequal config arrays or changing the controller.
        self.refs = {}
        for part in parts:
            start, end = ik.action_split_indexes()[part]
            if end - start != 6 or start % 6:
                raise ValueError(f"Unexpected IK pose slice for {part}: {(start, end)}")
            self.refs[part] = ik.site_names[start // 6]
        self.tcp_site = self.robot.gripper["left"].important_sites["grip_site"]
        self.reference_site = self.refs["left"]
        self.base_site = self.robot.robot_model.base.correct_naming("center")
        self.tcp_to_reference = np.linalg.inv(self.site_transform(self.tcp_site)) @ self.site_transform(self.reference_site)
        ik.q0 = np.array([float(env.sim.data.get_joint_qpos(name)) for name in ik.joint_names])
        self.holds = {}
        for part, controller in self.controller.part_controllers.items():
            if part == "left" or part in self.controller.grippers:
                continue
            if part in self.refs:
                self.holds[part] = self.pose6(self.site_transform(self.refs[part]))
            elif controller.name == "JOINT_POSITION":
                if controller.input_type == "absolute":
                    self.holds[part] = np.asarray(env.sim.data.qpos[controller.qpos_index]).copy()
                else:
                    self.holds[part] = np.zeros(controller.control_dim)
            elif controller.name in ("JOINT_VELOCITY", "JOINT_TORQUE"):
                self.holds[part] = np.zeros(controller.control_dim)
            else:
                raise ValueError(f"Unsupported hold controller: {part}: {controller.name}")

    def site_transform(self, site):
        return make_transform(self.env.sim.data.get_site_xpos(site), self.env.sim.data.get_site_xmat(site))

    @staticmethod
    def pose6(transform):
        return np.concatenate((transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec()))

    def target_world(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (9,) or not np.isfinite(action).all():
            raise ValueError("Expected one finite 9D Cartesian action")
        return self.site_transform(self.base_site) @ make_transform(action[:3], rotation6d_to_matrix(action[3:]))

    def __call__(self, action):
        target = self.target_world(action) @ self.tcp_to_reference
        commands = {part: value.copy() for part, value in self.holds.items()}
        commands["left"] = self.pose6(target)
        for arm in self.robot.arms:
            if self.robot.gripper[arm].dof:
                commands[f"{arm}_gripper"] = np.ones(self.robot.gripper[arm].dof)
        packed = np.asarray(self.robot.create_action_vector(commands))
        if packed.shape != (self.env.action_dim,) or not np.isfinite(packed).all():
            raise ValueError("Invalid packed environment action")
        return packed
