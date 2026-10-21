"""Arm-neutral SE(3) commands anchored to the measured TCP at a replan.

Every target in a trajectory uses ONE anchor. Translation is expressed in the
anchor TCP axes; rotations use concatenate(R[:, 0], R[:, 1]), never Euler deltas.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from robomimic.utils.rby1_cartesian import rotation6d_to_matrix


def tcp_observation_keys(arm, robot_prefix="robot0"):
    if arm not in ("left", "right"):
        raise ValueError("arm must be left or right")
    prefix = f"{robot_prefix}_base_to_{arm}_eef"
    return prefix + "_pos", prefix + "_quat_site"


def validate_transform(transform):
    value = np.asarray(transform, dtype=np.float64)
    if value.shape[-2:] != (4, 4) or not np.isfinite(value).all():
        raise ValueError("Expected finite SE(3) matrices (...,4,4)")
    rotation = value[..., :3, :3]
    if (not np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError("Invalid rigid transform")
    return value


def tcp_pose(position, quaternion_xyzw):
    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if (position.shape[-1:] != (3,) or quaternion.shape != position.shape[:-1] + (4,)
            or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
            or np.any(np.linalg.norm(quaternion, axis=-1) < 1e-12)):
        raise ValueError("Expected matching finite xyz/xyzw arrays")
    result = np.broadcast_to(np.eye(4), position.shape[:-1] + (4, 4)).copy()
    result[..., :3, :3] = Rotation.from_quat(quaternion.reshape(-1, 4)).as_matrix().reshape(position.shape[:-1] + (3, 3))
    result[..., :3, 3] = position
    return result


def get_tcp_pose(obs, arm, robot_prefix="robot0"):
    position, quaternion = tcp_observation_keys(arm, robot_prefix)
    return tcp_pose(obs[position], obs[quaternion])


def pose9_to_transform(action):
    action = np.asarray(action, dtype=np.float64)
    if action.shape[-1:] != (9,) or not np.isfinite(action).all():
        raise ValueError("Expected finite Cartesian commands (...,9)")
    result = np.broadcast_to(np.eye(4), action.shape[:-1] + (4, 4)).copy()
    result[..., :3, :3] = rotation6d_to_matrix(action[..., 3:])
    result[..., :3, 3] = action[..., :3]
    return result


def transform_to_pose9(transform):
    transform = validate_transform(transform)
    return np.concatenate((transform[..., :3, 3], transform[..., :3, 0], transform[..., :3, 1]), axis=-1)


def encode_relative_target(anchor, target):
    """Return pose9(inv(anchor) @ target); targets may be an entire trajectory."""
    return transform_to_pose9(np.linalg.inv(validate_transform(anchor)) @ validate_transform(target))


def decode_relative_target(anchor, relative_target):
    """Return anchor @ relative[k] for every k, with no chained integration."""
    return validate_transform(validate_transform(anchor) @ pose9_to_transform(relative_target))
