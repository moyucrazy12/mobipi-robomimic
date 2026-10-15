"""Check that native diffusion inference uses one consistent network version."""

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from robomimic.algo.diffusion_policy import DiffusionPolicyUNet


class Encoder(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, obs, goal=None):
        self.calls += 1
        return torch.full((obs["state"].shape[0], 1), self.value)


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.conditions = []

    def forward(self, sample, timestep, global_cond):
        self.conditions.append(global_cond.clone())
        return torch.ones_like(sample) * global_cond.mean()


class Scheduler:
    def set_timesteps(self, count):
        self.timesteps = range(count)

    def step(self, model_output, timestep, sample):
        return SimpleNamespace(prev_sample=model_output)


def networks(value):
    return nn.ModuleDict({"policy": nn.ModuleDict({
        "obs_encoder": Encoder(value), "noise_pred_net": Denoiser(),
    })}).eval()


class DiffusionInferenceTests(unittest.TestCase):
    def check_selection(self, use_ema):
        raw, averaged = networks(1.0), networks(2.0)
        # Exercise the real inference method and time-distributed observation
        # handling with small deterministic modules, without training or data.
        policy = SimpleNamespace(
            nets=raw, ema=SimpleNamespace(averaged_model=averaged) if use_ema else None,
            algo_config=SimpleNamespace(
                horizon=SimpleNamespace(observation_horizon=2, prediction_horizon=16, action_horizon=8),
                ddpm=SimpleNamespace(enabled=False),
                ddim=SimpleNamespace(enabled=True, num_inference_timesteps=2),
            ),
            obs_shapes={"state": (3,)}, ac_dim=3, device=torch.device("cpu"),
            noise_scheduler=Scheduler(),
        )
        action = DiffusionPolicyUNet._get_action_trajectory(
            policy, obs_dict={"state": torch.zeros(1, 2, 3)})
        selected, unused = (averaged, raw) if use_ema else (raw, averaged)
        self.assertEqual(selected["policy"]["obs_encoder"].calls, 1)
        self.assertEqual(len(selected["policy"]["noise_pred_net"].conditions), 2)
        self.assertEqual(unused["policy"]["obs_encoder"].calls, 0)
        self.assertEqual(len(unused["policy"]["noise_pred_net"].conditions), 0)
        expected = 2.0 if use_ema else 1.0
        for condition in selected["policy"]["noise_pred_net"].conditions:
            torch.testing.assert_close(condition, torch.full((1, 2), expected))
        torch.testing.assert_close(action, torch.full((1, 8, 3), expected))

    def test_ema_encoder_and_denoiser_are_used_together(self):
        self.check_selection(True)

    def test_online_encoder_and_denoiser_are_used_without_ema(self):
        self.check_selection(False)


if __name__ == "__main__":
    unittest.main()
