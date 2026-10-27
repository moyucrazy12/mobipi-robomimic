"""Same Diffusion Policy networks and queue; only the Cartesian action frame differs."""
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet
from robomimic.utils import tensor_utils as T
from robomimic.utils.rby1_relative_cartesian import (
    tcp_observation_keys, tcp_pose, decode_relative_target, transform_to_pose9,
)


class RelativeDiffusionPolicyUNet(DiffusionPolicyUNet):
    def _create_networks(self):
        relative = self.algo_config.relative_actions
        self.relative_arm = relative.arm
        self.position_key, self.quaternion_key = tcp_observation_keys(self.relative_arm)
        if (dict(self.algo_config.horizon) != dict(observation_horizon=2, prediction_horizon=16, action_horizon=8)
                or self.global_config.train.frame_stack != 2
                or self.global_config.train.hdf5_normalize_obs
                or list(self.global_config.train.action_keys) != ["actions_eef_cmd"]
                or self.global_config.train.action_config["actions_eef_cmd"].get("format") is not None
                or not {self.position_key, self.quaternion_key}.issubset(self.obs_shapes)):
            raise ValueError("Relative baseline requires 2/16/8, raw selected-arm TCP observations and native 9D command format")
        self.relative_anchor = None
        super()._create_networks()

    def reset(self):
        super().reset()
        self.relative_anchor = None

    def get_action(self, obs_dict, goal_dict=None):
        if len(self.action_queue) == 0:
            # obs_dict has [B=1,T=2,D]. Capture only when a new native plan is made.
            positions = T.to_numpy(obs_dict[self.position_key])
            quaternions = T.to_numpy(obs_dict[self.quaternion_key])
            if positions.shape != (1, 2, 3) or quaternions.shape != (1, 2, 4):
                raise ValueError("Relative queued rollout supports one environment with two observation frames")
            self.relative_anchor = tcp_pose(positions[0, -1], quaternions[0, -1])
        return super().get_action(obs_dict, goal_dict)

    def decode_action_for_environment(self, action):
        """Called by RolloutPolicy AFTER its one native unnormalization.

        Queue entries remain normalized relative9. External RolloutPolicy returns
        absolute base-frame9, so the unchanged Cartesian adapter is reused.
        """
        if self.relative_anchor is None:
            raise RuntimeError("No measured replan anchor; call through the native rollout policy")
        return transform_to_pose9(decode_relative_target(self.relative_anchor, action))
