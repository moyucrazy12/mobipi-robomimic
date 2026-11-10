"""CPU training and model-state reload with synthetic vector observations."""
from pathlib import Path
import tempfile
import unittest

import torch

from robomimic.algo import algo_factory
from robomimic.config import config_factory
from robomimic.utils import obs_utils


class TrainingSmokeTests(unittest.TestCase):
    def test_absolute_and_relative_gripper_update_reload_and_continue(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        for relative in (False, True):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                torch.manual_seed(0)
                config = config_factory("diffusion_policy")
                config.algo.relative_actions.enabled = relative
                config.algo.relative_actions.arm = "right"
                config.train.frame_stack = 2
                config.train.hdf5_normalize_obs = False
                keys = ["actions_eef_cmd", "actions_gripper"] if relative else ["actions_eef_cmd"]
                config.train.action_keys = keys
                config.train.action_config = {key: {"normalization": "min_max"} for key in keys}
                shapes = {"robot0_base_to_right_eef_pos": (3,), "robot0_base_to_right_eef_quat_site": (4,)}
                config.observation.modalities.obs.low_dim = list(shapes)
                config.observation.modalities.obs.rgb = []
                config.observation.modalities.obs.depth = []
                config.observation.modalities.obs.scan = []
                obs_utils.initialize_obs_utils_with_config(config)
                dimension = 10 if relative else 9

                def make_model():
                    return algo_factory("diffusion_policy", config, shapes, dimension, torch.device("cpu"))

                model = make_model()
                model.set_train()
                batch = {"obs": {key: torch.zeros(2, 16, *shape) for key, shape in shapes.items()},
                         "actions": torch.rand(2, 16, dimension) * 2 - 1}
                batch["obs"]["robot0_base_to_right_eef_quat_site"][..., 3] = 1
                processed = model.process_batch_for_training(batch)
                before = {key: value.clone() for key, value in model.nets.state_dict().items()}
                result = model.train_on_batch(processed, epoch=0)
                self.assertTrue(torch.isfinite(result["losses"]["l2_loss"]))
                self.assertTrue(any(not torch.equal(before[key], value)
                                    for key, value in model.nets.state_dict().items()))
                path = Path(temporary) / "model_state.pt"
                torch.save(model.serialize(), path)
                restored = make_model()
                restored.deserialize(torch.load(path, map_location="cpu", weights_only=True))
                for key, value in model.nets.state_dict().items():
                    torch.testing.assert_close(restored.nets.state_dict()[key], value)
                restored.set_train()
                result = restored.train_on_batch(restored.process_batch_for_training(batch), epoch=1)
                self.assertTrue(torch.isfinite(result["losses"]["l2_loss"]))


if __name__ == "__main__":
    unittest.main()
