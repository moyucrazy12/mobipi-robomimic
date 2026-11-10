"""Diagnose exactly one command-policy rollout, its recorded-observation predictions and GT control.

Observers capture native sampling/queue outputs without replacing the queue,
normalizer, adapter, controller, observations or diffusion schedule.
"""
import argparse
from collections import deque
import json
from pathlib import Path

from robomimic.utils.rby1_paths import new_output
from types import SimpleNamespace
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from robomimic.utils import file_utils as F, obs_utils as O
from robomimic.utils.rby1_cartesian import RBY1CartesianAdapter, source_command_as_cartesian, rotation6d_to_matrix
from robomimic.utils.rby1_saved_model import create_saved_env
from robomimic.utils.rby1_evaluation import RGB_KEYS, LOWDIM_KEYS, atomic_json, fingerprint
from robomimic.scripts.replay_rby1_cartesian import door_geometry
from robomimic.scripts.evaluate_rby1_cartesian_checkpoints import (
    check_policy, execute_loop, observation_reader, seed_rollout,
)


def angle(a, b):
    return float(np.rad2deg((Rotation.from_matrix(rotation6d_to_matrix(a[3:])).inv() *
                            Rotation.from_matrix(rotation6d_to_matrix(b[3:]))).magnitude()))


def stats(a):
    return dict(shape=list(a.shape), min=a.min(0).tolist(), max=a.max(0).tolist(),
                mean=a.mean(0).tolist(), max_norm=float(np.linalg.norm(a, axis=-1).max()))


class NativeTrace:
    def __init__(self, policy, env, adapter, address, output, ranges):
        self.policy, self.env, self.adapter = policy, env, adapter
        self.address, self.output, self.ranges = address, output, ranges
        self.chunks, self.steps, self.rng, self.executed = [], [], [], []
        self.previous = None
        self.scale_errors, self.conversion_errors, self.orientation_roundtrip = [], [], []
        self.original_trajectory = policy.policy._get_action_trajectory
        self.original_action = policy.policy.get_action
        self.original_scheduler = policy.policy.noise_scheduler.step

        def scheduler(*args, **kwargs):
            result = self.original_scheduler(*args, **kwargs)
            self.full = result.prev_sample.detach().cpu().numpy()[0].copy()
            return result

        def trajectory(*args, **kwargs):
            self.rng.append((torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()))
            result = self.original_trajectory(*args, **kwargs)
            queue = result.detach().cpu().numpy()[0]
            assert queue.shape == (8, 9) and self.full.shape == (16, 9)
            assert np.array_equal(queue, self.full[1:9]), 'Native queue slice mismatch'
            physical = O.unnormalize_dict({'actions_eef_cmd': self.full.copy()},
                                          policy.action_normalization_stats)['actions_eef_cmd']
            index = len(self.chunks)
            self.chunks.append(dict(chunk_index=index, environment_timestep=len(self.steps),
                raw_policy_output_shape=list(result.shape), full_prediction_shape=[1, 16, 9],
                normalized_statistics=stats(self.full), unnormalized_statistics=stats(physical),
                normalized_predictions_16=self.full.tolist(), unnormalized_predictions_16=physical.tolist(),
                first_predicted_target=physical[0].tolist(), executed_targets_8=physical[1:9].tolist(),
                position_target_min=physical[:, :3].min(0).tolist(), position_target_max=physical[:, :3].max(0).tolist(),
                max_executed_position_jump_mm=float(np.linalg.norm(np.diff(physical[1:9, :3], axis=0), axis=1).max()*1000),
                max_executed_orientation_jump_deg=max(angle(a,b) for a,b in zip(physical[1:8],physical[2:9]))))
            return result

        def action(*args, **kwargs):
            result = self.original_action(*args, **kwargs)
            self.normalized_executed = result.detach().cpu().numpy()[0].copy()
            return result
        policy.policy.noise_scheduler.step = scheduler
        policy.policy._get_action_trajectory = trajectory
        policy.policy.get_action = action

    def close(self):
        self.policy.policy.noise_scheduler.step = self.original_scheduler
        self.policy.policy._get_action_trajectory = self.original_trajectory
        self.policy.policy.get_action = self.original_action

    def observe(self, **s):
        action = s['action']; chunk = self.chunks[-1]
        expected = O.unnormalize_dict({'actions_eef_cmd': self.normalized_executed.copy()},
                                     self.policy.action_normalization_stats)['actions_eef_cmd']
        self.scale_errors.append(float(np.max(np.abs(action - expected))))
        assert self.scale_errors[-1] < 1e-10, 'Action was not unnormalized exactly once'
        back = source_command_as_cartesian(self.adapter, s['packed'])
        self.conversion_errors.append(float(np.linalg.norm(back[:3]-action[:3])))
        self.orientation_roundtrip.append(angle(back,action))
        base = self.adapter.site_transform(self.adapter.base_site)
        before = np.linalg.inv(base) @ s['actual_before']; after = np.linalg.inv(base) @ s['actual_after']
        actual_pose = np.r_[before[:3,3], before[:3,0], before[:3,1]]
        jump_mm = float(np.linalg.norm(action[:3]-self.previous[:3])*1000) if self.previous is not None else None
        jump_deg = angle(self.previous,action) if self.previous is not None else None
        outside = (action < self.ranges[0]-1e-7) | (action > self.ranges[1]+1e-7)
        row = dict(timestep=s['step'], chunk_index=chunk['chunk_index'], queue_regenerated=bool(s['regenerated']),
            queue_remaining=s['queue_remaining'], actual_tcp_before_base_xyz=before[:3,3].tolist(),
            actual_tcp_before_base_quat=Rotation.from_matrix(before[:3,:3]).as_quat().tolist(),
            actual_tcp_after_base_xyz=after[:3,3].tolist(),
            actual_tcp_after_base_quat=Rotation.from_matrix(after[:3,:3]).as_quat().tolist(),
            policy_lowdim_observation={key:s['inputs'][key].tolist() for key in LOWDIM_KEYS},
            normalized_executed_action=self.normalized_executed.tolist(), executed_action9=action.tolist(),
            executed_action31=s['packed'].tolist(),
            hinge_before=s['hinge_before'], hinge_after=s['hinge_after'],
            tracking_error_mm=s['position_error_mm'], tracking_error_deg=s['orientation_error_deg'],
            current_tcp_to_command_mm=float(np.linalg.norm(action[:3]-actual_pose[:3])*1000),
            current_tcp_to_command_deg=angle(actual_pose,action),
            consecutive_command_jump_mm=jump_mm, consecutive_command_jump_deg=jump_deg,
            outside_dataset_components=np.flatnonzero(outside).tolist())
        if s['regenerated']:
            chunk.update(actual_tcp_position_base=before[:3,3].tolist(), actual_tcp_quat_base=row['actual_tcp_before_base_quat'],
                         lowdim_observation=row['policy_lowdim_observation'], hinge_angle=s['hinge_before'],
                         delta_to_first_executed_xyz=(action[:3]-before[:3,3]).tolist(),
                         delta_to_first_executed_mm=row['current_tcp_to_command_mm'],
                         delta_to_first_executed_deg=row['current_tcp_to_command_deg'])
        self.steps.append(row); self.executed.append(action.copy()); self.previous=action.copy()
        with (self.output/'policy_steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if s['step'] % 200 == 0:
            atomic_json(self.output/'chunks.json', self.chunks)
            print('POLICY',s['step'],'hinge',s['hinge_after'],'tracking mm',s['position_error_mm'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--dataset',required=True)
    parser.add_argument("--output-dir",default=str(new_output("CloseSingleDoor", "diagnostics", "rby1_rollout_trace")))
    args=parser.parse_args()
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    dataset=Path(args.dataset).expanduser().resolve()
    checkpoint=Path(args.run_dir).resolve()/'models/model_epoch_15.pth'
    identities={str(p):fingerprint(p) for p in [dataset,checkpoint]}
    with h5py.File(dataset,'r') as f:
        metadata=json.loads(f['data'].attrs['env_args']);g=f['data/demo_10']
        all_min=np.full(9,np.inf);all_max=np.full(9,-np.inf)
        train_min=all_min.copy();train_max=all_max.copy();train_names=set(f['mask/train'].asstr()[:])
        for name,d in f['data'].items():
            a=d['actions_eef_cmd'][:];all_min=np.minimum(all_min,a.min(0));all_max=np.maximum(all_max,a.max(0))
            if name in train_names:train_min=np.minimum(train_min,a.min(0));train_max=np.maximum(train_max,a.max(0))
        ranges=np.stack([all_min,all_max]);truth=g['actions_eef_cmd'][:]
        seed_rollout(0)
        policy,ckpt=F.policy_from_checkpoint(ckpt_path=str(checkpoint),device=torch.device('cuda'),verbose=False)
        check_policy(policy,ckpt,metadata)
        normalizer=policy.action_normalization_stats['actions_eef_cmd']
        # Physical coordinates at -1 and +1; native train extrema map to +/-0.999999.
        limits=O.unnormalize_dict({'actions_eef_cmd':np.stack([-np.ones(9),np.ones(9)])},policy.action_normalization_stats)['actions_eef_cmd']
        atomic_json(output/'normalization.json',dict(dataset_min=all_min.tolist(),dataset_max=all_max.tolist(),
            train_min=train_min.tolist(),train_max=train_max.tolist(),normalized_minus1_plus1_physical=limits.tolist(),
            statistics={k:v.tolist() for k,v in normalizer.items()}))
        seed_rollout(0)
        env,audit=create_saved_env(metadata,g,render=True)
        try:
            adapter=RBY1CartesianAdapter(env);_,address,_=door_geometry(env,json.loads(g.attrs['ep_meta']))
            read=observation_reader(env,adapter);initial=read()
            for key in LOWDIM_KEYS:assert np.allclose(initial[key],g['obs/'+key][0],atol=1e-6,rtol=0)
            for key in RGB_KEYS:assert np.allclose(initial[key],O.process_obs(g['obs/'+key][0],obs_key=key),atol=1/255+1e-7,rtol=0)
            policy.start_episode();trace=NativeTrace(policy,env,adapter,address,output,ranges)
            try:result=execute_loop(policy,env,adapter,read,address,2000,observer=trace.observe)
            finally:trace.close()
            assert not result['error'],result
            atomic_json(output/'policy_result.json',result);atomic_json(output/'initialization.json',audit)
            atomic_json(output/'chunks.json',trace.chunks)
        finally:env.close()
        commands=np.stack(trace.executed)
        full=np.concatenate([np.asarray(c['unnormalized_predictions_16']) for c in trace.chunks])
        event=None
        for row in trace.steps:
            reasons=[]
            if row['consecutive_command_jump_mm'] is not None and row['consecutive_command_jump_mm']>100:reasons.append('command_jump_over_100mm')
            if row['consecutive_command_jump_deg'] is not None and row['consecutive_command_jump_deg']>20:reasons.append('command_jump_over_20deg')
            if row['tracking_error_mm']>100:reasons.append('tracking_over_100mm')
            if row['outside_dataset_components']:reasons.append('outside_training_dataset_range')
            if reasons:event=dict(row,reasons=reasons);break
        notable=[r['timestep'] for r in trace.steps if r['queue_regenerated'] and
                 ((r['consecutive_command_jump_mm'] or 0)>100 or (r['consecutive_command_jump_deg'] or 0)>20)]
        chosen=sorted(set([0,8,16,32,64]+notable[:5]+([event['timestep']//8*8] if event else [])))
        offline=[]
        rotations=Rotation.from_matrix(rotation6d_to_matrix(truth[:,3:]))
        def nearest(a):
            distance=np.linalg.norm(truth[:,:3]-a[:3],axis=1)
            orientation=np.rad2deg((Rotation.from_matrix(rotation6d_to_matrix(a[3:])).inv()*rotations).magnitude())
            i=int(np.argmin(distance/0.05+orientation/30))
            return dict(timestep=i,position_error_mm=float(distance[i]*1000),orientation_error_deg=float(orientation[i]))
        for t in chosen:
            if t>=len(truth):continue
            obs={k:np.stack([O.process_obs(g['obs/'+k][max(0,t-1)],obs_key=k),O.process_obs(g['obs/'+k][t],obs_key=k)]) for k in RGB_KEYS+LOWDIM_KEYS}
            for mode in ['fresh_seed0','matching_rollout_rng']:
                policy.start_episode()
                if mode=='fresh_seed0':seed_rollout(0)
                else:
                    torch.set_rng_state(trace.rng[t//8][0]);torch.cuda.set_rng_state(trace.rng[t//8][1])
                with torch.no_grad():pred=policy(obs)
                row=dict(timestep=t,rng_mode=mode,predicted_command=pred.tolist(),ground_truth_command=truth[t].tolist(),
                    position_error_mm=float(np.linalg.norm(pred[:3]-truth[t,:3])*1000),orientation_error_deg=angle(pred,truth[t]),
                    prediction_norm=float(np.linalg.norm(pred)),
                    displacement_from_current_tcp_mm=float(np.linalg.norm(pred[:3]-obs[LOWDIM_KEYS[0]][-1])*1000),
                    nearest_gt_target=nearest(pred),
                    rollout_prediction_difference_mm=float(np.linalg.norm(pred[:3]-commands[t,:3])*1000),
                    rollout_prediction_difference_deg=angle(pred,commands[t]))
                offline.append(row);print('OFFLINE',t,mode,'err mm',row['position_error_mm'],'move mm',row['displacement_from_current_tcp_mm'],flush=True)
        atomic_json(output/'offline.json',offline)
        if event:event['nearest_gt_target']=nearest(np.asarray(event['executed_action9']))
        scale=dict(dataset_min=all_min.tolist(),dataset_max=all_max.tolist(),train_min=train_min.tolist(),train_max=train_max.tolist(),
            normalizer_physical_limits=limits.tolist(),executed_statistics=stats(commands),full_predictions_statistics=stats(full),
            executed_xyz_outside_percent=float(np.any((commands[:,:3]<all_min[:3]-1e-7)|(commands[:,:3]>all_max[:3]+1e-7),axis=1).mean()*100),
            executed_rot6d_outside_percent=float(np.any((commands[:,3:]<all_min[3:]-1e-7)|(commands[:,3:]>all_max[3:]+1e-7),axis=1).mean()*100),
            full_xyz_outside_percent=float(np.any((full[:,:3]<all_min[:3]-1e-7)|(full[:,:3]>all_max[:3]+1e-7),axis=1).mean()*100),
            full_rot6d_outside_percent=float(np.any((full[:,3:]<all_min[3:]-1e-7)|(full[:,3:]>all_max[3:]+1e-7),axis=1).mean()*100),
            max_once_unnormalization_difference=max(trace.scale_errors),max_adapter_roundtrip_position_m=max(trace.conversion_errors),
            max_adapter_roundtrip_orientation_deg=max(trace.orientation_roundtrip),native_chunks=len(trace.chunks),
            regenerated_timesteps=[r['timestep'] for r in trace.steps if r['queue_regenerated']],first_bad_event=event)
        atomic_json(output/'scale_and_queue.json',scale)
        # Ground-truth open-loop command control, using identical saved start and evaluator metric.
        seed_rollout(0);env,audit=create_saved_env(metadata,g,render=True)
        gt_rows=[]
        class GroundTruth:
            def __init__(self):self.index=0;self.policy=SimpleNamespace(action_queue=deque())
            def __call__(self,inputs):
                action=truth[self.index];self.index+=1;return action
        try:
            adapter=RBY1CartesianAdapter(env);_,address,_=door_geometry(env,json.loads(g.attrs['ep_meta']))
            def observe_gt(**s):
                row=dict(timestep=s['step'],tracking_error_mm=s['position_error_mm'],tracking_error_deg=s['orientation_error_deg'],hinge=s['hinge_after'])
                gt_rows.append(row)
                if s['step']%300==0:print('GT',s['step'],'hinge',s['hinge_after'],'tracking mm',s['position_error_mm'],flush=True)
            gt_result=execute_loop(GroundTruth(),env,adapter,observation_reader(env,adapter),address,len(truth),observer=observe_gt)
            gt_result.update(policy_inference_count=0, number_action_chunks=0,
                             execution_mode='recorded commands, one per timestep; no diffusion inference')
            atomic_json(output/'gt_result.json',gt_result);atomic_json(output/'gt_steps.json',gt_rows)
        finally:env.close()
    assert identities=={str(p):fingerprint(p) for p in [dataset,checkpoint]}
    atomic_json(output/'summary.json',dict(checkpoint=str(checkpoint),dataset=str(dataset),demo='demo_10',seed=0,
        policy_result=result,gt_result=gt_result,scale_and_queue=scale,offline=offline,inputs_unchanged=True,
        metric='post-step actual LEFT grip-site TCP vs same-step executed Cartesian grip command, both world frame; units mm and degrees'))
    print('DONE',output,flush=True)


if __name__=='__main__':main()
