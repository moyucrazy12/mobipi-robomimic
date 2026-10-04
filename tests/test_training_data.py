"""Regression tests for dataset lists, optional language, and normalization."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from robomimic.config import config_factory
from robomimic.macros import LANG_EMB_KEY
from robomimic.utils import obs_utils, train_utils
from robomimic.utils.dataset import SequenceDataset


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="robomimic_data_test_")
        self.addCleanup(self.directory.cleanup)
        self.paths = []
        for dataset_id in range(2):
            path = Path(self.directory.name) / f"dataset_{dataset_id}.hdf5"
            self.paths.append(str(path))
            with h5py.File(path, "w") as dataset:
                for demo_id in range(2):
                    demo = dataset.create_group(f"data/demo_{demo_id}")
                    demo.attrs["num_samples"] = 4
                    demo.attrs["ep_meta"] = json.dumps({"lang": "example task"})
                    demo.create_dataset("obs/position", data=np.zeros((4, 3), dtype=np.float32))
                    values = np.arange(4, dtype=np.float32) + 10 * demo_id + dataset_id
                    demo.create_dataset("actions", data=np.column_stack((values, values)))
                dataset.create_dataset("mask/train", data=np.array([b"demo_0"]))
                dataset.create_dataset("mask/valid", data=np.array([b"demo_1"]))
        self.config = config_factory("diffusion_policy")
        self.config.train.data = [{"path": path} for path in self.paths]
        self.config.experiment.validate = True
        self.config.train.hdf5_filter_key = "train"
        self.config.train.hdf5_validation_filter_key = "valid"
        self.config.train.hdf5_cache_mode = "low_dim"
        self.config.train.hdf5_load_next_obs = False
        self.config.train.dataset_keys = []
        self.config.train.action_keys = ["actions"]
        self.config.train.action_config = {"actions": {"normalization": "min_max"}}
        self.config.observation.modalities.obs.low_dim = ["position"]
        obs_utils.initialize_obs_utils_with_config(self.config)

    def register_close(self, dataset):
        for child in getattr(dataset, "datasets", [dataset]):
            self.addCleanup(child.close_and_delete_hdf5_handle)

    def load_split(self):
        train, valid = train_utils.load_data_for_training(self.config, ["position"])
        self.register_close(train)
        self.register_close(valid)
        return train, valid

    def test_dataset_list_masks_are_checked_per_file_and_language_disabled(self):
        with patch("robomimic.utils.dataset.LangUtils.LangEncoder", side_effect=AssertionError("unexpected CLIP load")):
            train, valid = self.load_split()
            self.assertEqual(len(train.datasets), 2)
            self.assertEqual(len(valid.datasets), 2)
            self.assertTrue(all(child.demos == ["demo_0"] for child in train.datasets))
            self.assertTrue(all(child.demos == ["demo_1"] for child in valid.datasets))
            self.assertNotIn(LANG_EMB_KEY, train[0]["obs"])
            self.assertNotIn(LANG_EMB_KEY, valid[0]["obs"])

    def test_overlap_in_second_dataset_is_rejected(self):
        with h5py.File(self.paths[1], "r+") as dataset:
            dataset["mask/valid"][0] = b"demo_0"
        with self.assertRaisesRegex(AssertionError, "overlap"):
            self.load_split()

    def test_validation_uses_training_scale_for_single_and_multiple_datasets(self):
        for count in (1, 2):
            with self.subTest(dataset_count=count):
                self.config.train.data = [{"path": path} for path in self.paths[:count]]
                train, valid = self.load_split()
                independently_normalized = valid[0]["actions"].copy()
                stats = train.get_action_normalization_stats()
                valid.set_action_normalization_stats(stats)
                raw_action = np.array([10, 10], dtype=np.float32)
                expected = (raw_action - stats["actions"]["offset"][0]) / stats["actions"]["scale"][0]
                np.testing.assert_allclose(valid[0]["actions"][0], expected)
                self.assertFalse(np.allclose(valid[0]["actions"], independently_normalized))
                # Held-out data may exceed training bounds; it must keep its scale.
                self.assertTrue(np.all(valid[0]["actions"][0] > 1))

    def test_language_enabled_and_direct_constructor_default_are_preserved(self):
        for kwargs in ({}, {"load_language": True}):
            with self.subTest(kwargs=kwargs):
                with patch("robomimic.utils.dataset.LangUtils.LangEncoder") as encoder:
                    encoder.return_value.get_lang_emb.side_effect = lambda languages: torch.zeros((len(languages), 768))
                    with patch("robomimic.utils.dataset.TorchUtils.get_torch_device", return_value=torch.device("cpu")):
                        dataset = SequenceDataset(
                            hdf5_path=self.paths[0], obs_keys=["position"], action_keys=["actions"],
                            dataset_keys=[], action_config={"actions": {"normalization": "min_max"}},
                            load_next_obs=False, **kwargs,
                        )
                    self.register_close(dataset)
                    encoder.assert_called_once()
                    self.assertIn(LANG_EMB_KEY, dataset[0]["obs"])


if __name__ == "__main__":
    unittest.main()
