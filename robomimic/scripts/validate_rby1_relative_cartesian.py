"""Validate dynamic relative labels, one saved-demo replay, or a debug checkpoint.

No phase launches training. Run standard train.py --debug separately after the
read-only dataset audit and saved-model replay gates have passed.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from robomimic.utils.rby1_evaluation import atomic_json, fingerprint
from robomimic.utils.rby1_relative_cartesian import (
    tcp_pose, tcp_observation_keys, pose9_to_transform, encode_relative_target,
    decode_relative_target, transform_to_pose9, get_tcp_pose,
)


def load_config(path):
    from robomimic.config import config_factory
    from robomimic.utils import obs_utils as O
    raw=json.loads(Path(path).read_text());config=config_factory('diffusion_policy')
    with config.values_unlocked():config.update(raw)
    config.lock();O.initialize_obs_utils_with_config(config)
    assert config.algo.relative_actions.enabled and config.algo.relative_actions.arm=='left'
    return config


def audit_dataset(config, output):
    from torch.utils.data import DataLoader
    from robomimic.utils import train_utils as T,obs_utils as O
    from robomimic.utils.dataset import action_stats_to_normalization_stats
    path=Path(config.train.data[0]["path"]).expanduser().resolve();before=fingerprint(path)
    keys=list(config.observation.modalities.obs.low_dim)+list(config.observation.modalities.obs.rgb)
    train,valid=T.load_data_for_training(config,keys)
    assert type(train).__name__==type(valid).__name__=='RelativeCartesianSequenceDataset'
    assert len(train.demos)==100 and len(valid.demos)==15
    stats=train.get_action_stats()
    normalization=action_stats_to_normalization_stats(stats,train.action_config)
    train.set_action_normalization_stats(normalization);valid.set_action_normalization_stats(normalization)
    maximum=0.;commands=0;windows=0
    with h5py.File(path,'r') as f:
        for name,g in f['data'].items():
            targets=pose9_to_transform(g['actions_eef_cmd'][:])
            anchors=get_tcp_pose({k:g['obs/'+k][:] for k in tcp_observation_keys('left')},'left')
            n=len(targets);indices=np.clip(np.arange(n)[:,None]+np.arange(-1,15)[None,:],0,n-1)
            relative=encode_relative_target(anchors[:,None],targets[indices])
            decoded=decode_relative_target(anchors[:,None],relative)
            maximum=max(maximum,float(np.max(np.abs(decoded-targets[indices]))))
            assert np.isfinite(relative).all()
            commands+=n;windows+=n*16
    assert maximum<1e-12 and commands==171258
    sample_errors=[]
    for d in [train,valid]:
        for index in [0,len(d)//2,len(d)-1]:
            sample=d[index]
            assert sample['actions'].shape==(16,9) and np.isfinite(sample['actions']).all()
            demo=d._index_to_demo_id[index];t=index-d._demo_id_to_start_indices[demo]
            with h5py.File(path,'r') as f:
                g=f['data/'+demo];ids=np.clip(t+np.arange(-1,15),0,len(g['actions'])-1)
                expected=pose9_to_transform(g['actions_eef_cmd'][:].astype(np.float32)[ids])
                anchor=get_tcp_pose({k:sample['obs'][k][1] for k in tcp_observation_keys('left')},'left')
                physical=O.unnormalize_dict({'actions_eef_cmd':sample['actions'].copy()},normalization)['actions_eef_cmd']
                sample_errors.append(float(np.max(np.abs(decode_relative_target(anchor,physical)-expected))))
    counts=dict(train_sequences=len(train),valid_sequences=len(valid),
                train_batches=len(DataLoader(train,batch_size=16,drop_last=True)),
                valid_batches=len(DataLoader(valid,batch_size=16,drop_last=True)))
    assert counts==dict(train_sequences=145925,valid_sequences=25333,train_batches=9120,valid_batches=1583)
    assert max(sample_errors)<2e-6
    assert fingerprint(path)==before
    action_stats=stats['actions_eef_cmd']
    report=dict(passed=True,dataset=str(path),counts=counts,commands=commands,conditioned_targets_checked=windows,
        anchor='sample t; second observation [max(0,t-1),t]; all clipped targets[t-1:t+15] share pose[t]',
        max_float64_transform_reconstruction_error=maximum,max_native_float32_normalized_sample_transform_error=max(sample_errors),
        relative_min=action_stats['min'][0].tolist(),relative_max=action_stats['max'][0].tolist(),
        normalization_source='100 training demos, all16 slots of every padded window relative to its own sample anchor; validation reuses train statistics',
        normalization={k:{part:values.tolist() for part,values in v.items()} for k,v in normalization.items()},dataset_unchanged=True)
    atomic_json(output/'dataset.json',report)
    print(json.dumps(report),flush=True)
    train.close_and_delete_hdf5_handle();valid.close_and_delete_hdf5_handle()


def replay(config, output, demo):
    assert json.loads((output/'dataset.json').read_text())['passed'],'Run dataset audit first'
    from robomimic.utils.rby1_saved_model import create_saved_env
    from robomimic.utils.rby1_cartesian import RBY1CartesianAdapter
    from robomimic.scripts.replay_rby1_cartesian import door_geometry
    path=Path(config.train.data[0]["path"]).expanduser().resolve();before=fingerprint(path)
    with h5py.File(path,'r') as f:
        g=f['data/'+demo];metadata=json.loads(f['data'].attrs['env_args'])
        env,audit=create_saved_env(metadata,g,render=False)
        try:
            adapter=RBY1CartesianAdapter(env);_,address,_=door_geometry(env,json.loads(g.attrs['ep_meta']))
            arm_frames={}
            for arm in ['left','right']:
                grip=f'gripper0_{arm}_grip_site';ref=f'gripper0_{arm}_ft_frame';hand=f'robot0_{arm}_hand'
                grip_tf=adapter.site_transform(grip);ref_tf=adapter.site_transform(ref)
                arm_frames[arm]=dict(tcp_site=grip,ik_reference_site=ref,hand_body=hand,
                    ft_to_tcp=(np.linalg.inv(ref_tf)@grip_tf).tolist(),
                    hand_to_tcp_rotation=(np.asarray(env.sim.data.get_body_xmat(hand)).T@grip_tf[:3,:3]).tolist())
            compatible=bool(np.allclose(arm_frames['left']['ft_to_tcp'],arm_frames['right']['ft_to_tcp'],atol=1e-10))
            truth=g['actions_eef_cmd'][:];n=len(truth);relative=None;decoded=None
            max_transform=0.;max_packed=0.;path_errors=[];first_success=None
            for t in range(n):
                if t%8==0:
                    raw=env._get_observations(force_update=True)
                    anchor=get_tcp_pose(raw,'left')
                    measured=np.linalg.inv(adapter.site_transform(adapter.base_site))@adapter.site_transform(adapter.tcp_site)
                    assert np.allclose(anchor,measured,atol=1e-6)
                    ids=np.clip(t+np.arange(-1,15),0,n-1)
                    relative=encode_relative_target(anchor,pose9_to_transform(truth[ids]))
                    decoded=decode_relative_target(anchor,relative)
                    max_transform=max(max_transform,float(np.max(np.abs(decoded-pose9_to_transform(truth[ids])))))
                    assert max_transform<1e-12
                absolute=transform_to_pose9(decoded[1+t%8])
                packed=adapter(absolute);reference=adapter(truth[t])
                pos=float(np.max(np.abs(packed[6:9]-reference[6:9])))
                orientation=float((Rotation.from_rotvec(packed[9:12]).inv()*Rotation.from_rotvec(reference[9:12])).magnitude())
                max_packed=max(max_packed,pos,orientation)
                assert max_packed<1e-12
                env.step(packed)
                tcp=np.linalg.inv(adapter.site_transform(adapter.base_site))@adapter.site_transform(adapter.tcp_site)
                if t+1<n:path_errors.append(float(np.linalg.norm(tcp[:3,3]-g['obs/robot0_base_to_left_eef_pos'][t+1])))
                hinge=float(env.sim.data.qpos[address])
                if hinge>=-0.07853981634 and first_success is None:first_success=t+1
                assert np.isfinite(env.sim.data.qpos).all() and np.isfinite(env.sim.data.qvel).all()
                if t%300==0:print('REPLAY',demo,t,'hinge',hinge,flush=True)
            report=dict(passed=hinge>=-0.07853981634,demo=demo,steps=n,first_success_step=first_success,
                final_hinge_rad=hinge,max_transform_reconstruction_error=max_transform,
                max_packed_left_position_m_or_rotation_rad_difference=max_packed,
                mean_recorded_path_error_mm=float(np.mean(path_errors)*1000),max_recorded_path_error_mm=float(max(path_errors)*1000),
                anchor_refresh_steps=list(range(0,n,8)),arm_frames=arm_frames,
                left_right_local_tcp_conventions_compatible=compatible,initialization=audit)
            atomic_json(output/'replay.json',report)
            assert report['passed'],'STOP: relative representation replay did not close the door'
            print(json.dumps(report),flush=True)
        finally:env.close()
    assert fingerprint(path)==before


def reload_debug(config, output, checkpoint):
    assert json.loads((output/'replay.json').read_text())['passed']
    import torch
    from robomimic.utils import file_utils as F,obs_utils as O
    policy,ckpt=F.policy_from_checkpoint(ckpt_path=str(checkpoint),device=torch.device('cuda'),verbose=False)
    assert type(policy.policy).__name__=='RelativeDiffusionPolicyUNet' and policy.policy.ema is not None
    assert policy.policy.global_config.algo.relative_actions.enabled
    assert ckpt['shape_metadata']['ac_dim']==9
    saved=policy.action_normalization_stats['actions_eef_cmd']
    expected=json.loads((output/'dataset.json').read_text())['normalization']['actions_eef_cmd']
    for key in ['offset','scale']:np.testing.assert_allclose(saved[key],expected[key],rtol=1e-6,atol=1e-7)
    with h5py.File(Path(config.train.data[0]["path"]).expanduser(),'r') as f:
        g=f['data/demo_10'];keys=list(config.observation.modalities.obs.low_dim)+list(config.observation.modalities.obs.rgb)
        obs={k:np.stack([O.process_obs(g['obs/'+k][0],obs_key=k)]*2) for k in keys}
    policy.start_episode();torch.manual_seed(0)
    # Return both the native unnormalized relative8 and native wrapper absolute8.
    normalized=[];original=policy.policy.get_action
    def capture(*args,**kwargs):
        action=original(*args,**kwargs);normalized.append(action.detach().cpu().numpy()[0]);return action
    policy.policy.get_action=capture
    counts=[0]
    hook=policy.policy.ema.averaged_model['policy']['obs_encoder'].register_forward_hook(lambda *a:counts.__setitem__(0,counts[0]+1))
    full_predictions=[]
    original_step=policy.policy.noise_scheduler.step
    def capture_step(*args,**kwargs):
        result=original_step(*args,**kwargs)
        full_predictions.append(result.prev_sample.detach().cpu().numpy()[0].copy())
        return result
    policy.policy.noise_scheduler.step=capture_step
    try:
        with torch.no_grad():absolute=np.stack([policy(obs) for _ in range(8)])
    finally:
        hook.remove();policy.policy.get_action=original
        policy.policy.noise_scheduler.step=original_step
    relative=O.unnormalize_dict({'actions_eef_cmd':np.stack(normalized)},policy.action_normalization_stats)['actions_eef_cmd']
    full_relative=O.unnormalize_dict({'actions_eef_cmd':full_predictions[-1]},policy.action_normalization_stats)['actions_eef_cmd']
    full_absolute=decode_relative_target(policy.policy.relative_anchor,full_relative)
    assert full_relative.shape==(16,9) and np.isfinite(full_absolute).all()
    np.testing.assert_allclose(relative,full_relative[1:9],atol=1e-7)
    expected_absolute=decode_relative_target(policy.policy.relative_anchor,relative)
    error=float(np.max(np.abs(pose9_to_transform(absolute)-expected_absolute)))
    assert np.isfinite(absolute).all() and counts[0]==1 and not policy.policy.action_queue and error<1e-12
    report=dict(passed=True,checkpoint=str(checkpoint),algorithm=type(policy.policy).__name__,ema_plans=counts[0],
        relative_chunk_shape=list(relative.shape),absolute_chunk_shape=list(absolute.shape),finite=True,
        full_relative_trajectory_shape=list(full_relative.shape),full_decoded_trajectory_finite=True,
        decode_error=error,first_relative_action=relative[0].tolist(),first_absolute_action=absolute[0].tolist(),
        relative_normalizer_restored=True)
    atomic_json(output/'debug_reload.json',report);print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['dataset','replay','reload'],required=True)
    parser.add_argument('--config',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--demo',default='demo_10');parser.add_argument('--checkpoint')
    args=parser.parse_args();config=load_config(args.config);output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    if args.phase=='dataset':audit_dataset(config,output)
    elif args.phase=='replay':replay(config,output,args.demo)
    else:
        if not args.checkpoint:parser.error('--checkpoint is required for reload')
        reload_debug(config,output,Path(args.checkpoint).expanduser().resolve())


if __name__=='__main__':main()
