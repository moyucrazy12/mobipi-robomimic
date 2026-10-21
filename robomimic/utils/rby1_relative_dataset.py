"""Dynamic sequence-relative command labels; the absolute HDF5 stays read-only."""
import numpy as np

from robomimic.utils.dataset import SequenceDataset
from robomimic.utils.rby1_relative_cartesian import (
    tcp_observation_keys, tcp_pose, pose9_to_transform, encode_relative_target,
)


class RelativeCartesianSequenceDataset(SequenceDataset):
    """Anchor at the LAST of the first two observation frames, i.e. sample t.

    At sample t, SequenceDataset returns clip([t-1,...,t+14]) with length16.
    DP conditions on obs[0:2] = [t-1,t] and executes prediction[1:9]. All16
    labels, including the past target at slot0, use the SAME measured pose[t].
    The source key remains actions_eef_cmd for native shape discovery; only its
    in-memory labels/statistics become relative when this class is selected.
    """
    def __init__(self, *args, relative_arm="left", **kwargs):
        self.relative_arm = relative_arm
        self.position_key, self.quaternion_key = tcp_observation_keys(relative_arm)
        if (kwargs.get("frame_stack") != 2 or kwargs.get("seq_length") != 15
                or not kwargs.get("pad_frame_stack", True) or not kwargs.get("pad_seq_length", True)
                or kwargs.get("hdf5_cache_mode") == "all"
                or tuple(kwargs.get("action_keys", ())) != ("actions_eef_cmd",)
                or kwargs.get("hdf5_normalize_obs", False)
                or kwargs.get("goal_mode") is not None):
            raise ValueError("Relative baseline requires frame2/seq15, both padding flags, one command key, raw TCP observations and low_dim/no cache")
        if not {self.position_key, self.quaternion_key}.issubset(kwargs["obs_keys"]):
            raise ValueError("Selected arm TCP position and grip-site quaternion must be policy observations")
        super().__init__(*args, **kwargs)

    def anchor_at(self, demo_id, index):
        position = self.get_dataset_for_ep(demo_id, "obs/" + self.position_key)[index]
        quaternion = self.get_dataset_for_ep(demo_id, "obs/" + self.quaternion_key)[index]
        return tcp_pose(position, quaternion)

    def get_dataset_sequence_from_demo(self, demo_id, index_in_demo, keys,
                                       num_frames_to_stack=0, seq_length=1):
        data = super().get_dataset_sequence_from_demo(
            demo_id, index_in_demo, keys, num_frames_to_stack, seq_length)
        if "actions_eef_cmd" in data:
            data["actions_eef_cmd"] = encode_relative_target(
                self.anchor_at(demo_id, index_in_demo),
                pose9_to_transform(data["actions_eef_cmd"])).astype(np.float32)
        return data

    def get_action_traj(self, ep):
        """Statistics over ALL actual padded training windows, not per-step deltas.

        Includes every slot in every prediction horizon under its own sample
        anchor. The parent aggregator computes the relative-only normalizer.
        Validation receives these training statistics through existing train.py.
        """
        commands = self.hdf5_file[f"data/{ep}/actions_eef_cmd"][:].astype(np.float32)
        n = len(commands)
        positions = self.hdf5_file[f"data/{ep}/obs/{self.position_key}"][:]
        quaternions = self.hdf5_file[f"data/{ep}/obs/{self.quaternion_key}"][:]
        anchors = tcp_pose(positions, quaternions)
        indices = np.clip(np.arange(n)[:, None] + np.arange(-1, 15)[None, :], 0, n - 1)
        relative = encode_relative_target(anchors[:, None], pose9_to_transform(commands)[indices])
        return {"actions_eef_cmd": relative.reshape(-1, 9).astype(np.float32)}
