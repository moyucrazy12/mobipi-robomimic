"""CPU tests for common-anchor geometry, padded dataset indexing and native queues."""
from collections import deque
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from robomimic.algo.algo import RolloutPolicy, algo_name_to_factory_func
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet
from robomimic.algo.relative_diffusion_policy import RelativeDiffusionPolicyUNet
from robomimic.utils import obs_utils as O
from robomimic.utils.dataset import SequenceDataset
from robomimic.utils.rby1_relative_dataset import RelativeCartesianSequenceDataset
from robomimic.utils.rby1_relative_cartesian import (
    tcp_pose, tcp_observation_keys, get_tcp_pose, transform_to_pose9,
    pose9_to_transform, encode_relative_target, decode_relative_target,
)


class RelativeGeometryTests(unittest.TestCase):
    def test_random_se3_and_rotation6d_roundtrip(self):
        rng = np.random.default_rng(132)
        anchors = tcp_pose(rng.normal(size=(50,3)), Rotation.random(50, random_state=rng).as_quat())
        targets = tcp_pose(rng.normal(size=(50,3)), Rotation.random(50, random_state=rng).as_quat())
        rel = encode_relative_target(anchors, targets)
        np.testing.assert_allclose(pose9_to_transform(rel), np.linalg.inv(anchors) @ targets, atol=2e-14)
        np.testing.assert_allclose(decode_relative_target(anchors, rel), targets, atol=2e-14)
        np.testing.assert_allclose(pose9_to_transform(transform_to_pose9(targets)), targets, atol=2e-14)

    def test_whole_trajectory_one_anchor_not_chained(self):
        anchor = tcp_pose([4,5,6], Rotation.from_euler('z',90,degrees=True).as_quat())
        targets = np.stack([tcp_pose([4,5+k,6], [0,0,0,1]) for k in [1,2,3]])
        relative = encode_relative_target(anchor, targets)
        np.testing.assert_allclose(relative[:,:3], [[1,0,0],[2,0,0],[3,0,0]], atol=1e-14)
        decoded = decode_relative_target(anchor, relative)
        np.testing.assert_allclose(decoded, targets, atol=1e-14)
        chained = anchor.copy()
        for action in relative: chained = chained @ pose9_to_transform(action)
        self.assertFalse(np.allclose(chained, targets[-1]))

    def test_both_arm_observation_interfaces(self):
        for arm in ['left','right']:
            p,q = tcp_observation_keys(arm)
            np.testing.assert_allclose(get_tcp_pose({p:np.zeros(3),q:[0,0,0,1]},arm), np.eye(4))
        with self.assertRaises(ValueError):tcp_observation_keys('invalid')

    def test_invalid_rotations_and_nans_rejected(self):
        for invalid in [np.zeros(9), np.full(9,np.nan)]:
            with self.assertRaises(ValueError):pose9_to_transform(invalid)
        bad=np.eye(4);bad[0,0]=2
        with self.assertRaises(ValueError):encode_relative_target(bad,np.eye(4))
        with self.assertRaises(ValueError):tcp_pose([0,0,0],[0,0,0,0])


class RelativeDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'test.hdf5'
        self.positions=np.array([[3+t*.1,4,5] for t in range(5)])
        self.quaternions=Rotation.from_euler('z',[0,20,40,60,80],degrees=True).as_quat()
        targets=tcp_pose(self.positions+np.array([.2,0,0]), self.quaternions)
        self.commands=transform_to_pose9(targets).astype(np.float32)
        with h5py.File(self.path,'w') as f:
            d=f.create_group('data/demo_0');d.attrs['num_samples']=5
            d.create_dataset('actions_eef_cmd',data=self.commands)
            for arm in ['left','right']:
                p,q=tcp_observation_keys(arm)
                d.create_dataset('obs/'+p,data=self.positions);d.create_dataset('obs/'+q,data=self.quaternions)
        O.initialize_obs_utils_with_obs_specs({'obs':{'low_dim':list(tcp_observation_keys('left'))+list(tcp_observation_keys('right'))}})

    def dataset(self, arm='left', relative=True):
        kwargs=dict(hdf5_path=str(self.path),obs_keys=list(tcp_observation_keys(arm)),action_keys=['actions_eef_cmd'],
            dataset_keys=[],action_config={'actions_eef_cmd':{'normalization':'min_max'}},frame_stack=2,seq_length=15,
            pad_frame_stack=True,pad_seq_length=True,load_next_obs=False,hdf5_cache_mode='low_dim',load_language=False)
        cls=RelativeCartesianSequenceDataset if relative else SequenceDataset
        return cls(**kwargs,**({'relative_arm':arm} if relative else {}))

    def test_sample_anchor_latest_observation_and_boundary_padding(self):
        for arm in ['left','right']:
            d=self.dataset(arm)
            try:
                for t in [0,1,3,4]:
                    sample=d[t];physical=O.unnormalize_dict({'actions_eef_cmd':sample['actions'].copy()},d.get_action_normalization_stats())['actions_eef_cmd']
                    indices=np.clip(t+np.arange(-1,15),0,4)
                    anchor=tcp_pose(self.positions[t],self.quaternions[t])
                    np.testing.assert_allclose(decode_relative_target(anchor,physical),pose9_to_transform(self.commands[indices]),atol=5e-7)
                    p,q=tcp_observation_keys(arm)
                    np.testing.assert_allclose(sample['obs'][p][1], self.positions[t])
                    np.testing.assert_allclose(sample['obs'][p][0], self.positions[max(0,t-1)])
            finally:d.close_and_delete_hdf5_handle()

    def test_relative_normalizer_covers_exact_conditioned_windows(self):
        d=self.dataset();a=self.dataset(relative=False)
        try:
            physical=d.get_action_traj('demo_0')['actions_eef_cmd']
            stats=d.get_action_stats()['actions_eef_cmd']
            np.testing.assert_allclose(stats['min'][0],physical.min(0))
            np.testing.assert_allclose(stats['max'][0],physical.max(0))
            self.assertEqual(len(physical),5*16)
            self.assertFalse(np.allclose(d.get_action_normalization_stats()['actions_eef_cmd']['offset'],
                                         a.get_action_normalization_stats()['actions_eef_cmd']['offset']))
            for t in range(5):
                sample=d[t]['actions']
                self.assertTrue(np.isfinite(sample).all())
                self.assertLessEqual(float(np.abs(sample).max()),1.000001)
            with h5py.File(self.path,'r') as source:
                np.testing.assert_array_equal(self.commands,source['data/demo_0/actions_eef_cmd'][:])
        finally:d.close_and_delete_hdf5_handle();a.close_and_delete_hdf5_handle()

    def test_native_diffusion_batch_anchor_is_second_observation(self):
        d = self.dataset()
        try:
            sample = d[3]
            batch = {"obs": {k: torch.tensor(v[None]) for k,v in sample["obs"].items()},
                     "actions": torch.tensor(sample["actions"][None])}
            model = SimpleNamespace(
                algo_config=SimpleNamespace(horizon=SimpleNamespace(
                    observation_horizon=2, action_horizon=8, prediction_horizon=16)),
                action_check_done=True, device=torch.device("cpu"))
            processed = DiffusionPolicyUNet.process_batch_for_training(model, batch)
            key = tcp_observation_keys("left")[0]
            self.assertEqual(tuple(processed["actions"].shape), (1,16,9))
            np.testing.assert_allclose(processed["obs"][key][0,-1].numpy(), self.positions[3], atol=1e-6)
            self.assertEqual(tuple(processed["obs"][key].shape), (1,2,3))
        finally:
            d.close_and_delete_hdf5_handle()


class RelativeQueueTests(unittest.TestCase):
    def test_creation_accepts_native_dictionary_action_config(self):
        from robomimic.config import config_factory
        config = config_factory("diffusion_policy")
        config.algo.relative_actions.enabled = True
        config.train.frame_stack = 2
        config.train.action_keys = ["actions_eef_cmd"]
        config.train.action_config = {"actions_eef_cmd": {"normalization": "min_max"}}
        model = object.__new__(RelativeDiffusionPolicyUNet)
        model.global_config = config
        model.algo_config = config.algo
        model.obs_shapes = dict.fromkeys(tcp_observation_keys("left"), (3,))
        with patch.object(DiffusionPolicyUNet, "_create_networks") as create:
            model._create_networks()
            create.assert_called_once()
        self.assertIsNone(model.relative_anchor)

    def test_native_queue_anchor_held_for_eight_actions_and_refreshed(self):
        # Exercise real relative subclass and real RolloutPolicy unnormalization,
        # with only the expensive denoiser replaced by a deterministic trajectory.
        model=object.__new__(RelativeDiffusionPolicyUNet)
        model.relative_arm='right';model.position_key,model.quaternion_key=tcp_observation_keys('right')
        model.algo_config=SimpleNamespace(horizon=SimpleNamespace(observation_horizon=2,action_horizon=8))
        model.device=torch.device('cpu');model.relative_anchor=None;model.action_queue=deque(maxlen=8)
        model.global_config=SimpleNamespace(train=SimpleNamespace(action_keys=['actions_eef_cmd'],action_config={'actions_eef_cmd':{}}))
        model.set_eval=lambda:None
        relative=transform_to_pose9(tcp_pose([.25,0,0],[0,0,0,1])).astype(np.float32)
        model._get_action_trajectory=lambda **kwargs:torch.tensor(np.tile(relative,(1,8,1)))
        wrapper=RolloutPolicy(model,action_normalization_stats={'actions_eef_cmd':{'scale':np.ones((1,9)),'offset':np.zeros((1,9))}})
        wrapper.start_episode()
        def obs(x):return {model.position_key:np.array([[x,0,0],[x,0,0]]),model.quaternion_key:np.array([[0,0,0,1],[0,0,0,1]])}
        for i in range(8):
            result=wrapper(obs(1+i))
            self.assertAlmostEqual(result[0],1.25)
        result=wrapper(obs(9))
        self.assertAlmostEqual(result[0],9.25)
        self.assertEqual(len(model.action_queue),7)
        wrapper.start_episode();self.assertIsNone(model.relative_anchor)

    def test_absolute_factory_unchanged_unless_enabled(self):
        class Config:
            unet=SimpleNamespace(enabled=True)
            def get(self,key,default):return {'enabled':True}
        # Real configs have dict.get; use a small compatible object.
        class AbsoluteConfig(Config):
            def get(self,key,default):return default
        self.assertIs(algo_name_to_factory_func("diffusion_policy")(AbsoluteConfig())[0],DiffusionPolicyUNet)
        self.assertIs(algo_name_to_factory_func("diffusion_policy")(Config())[0],RelativeDiffusionPolicyUNet)


if __name__=='__main__':unittest.main()
