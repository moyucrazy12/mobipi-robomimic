"""Small repeated-latent analysis on recorded observations; no simulator construction."""
import argparse
import csv
import gc
import json
from pathlib import Path

from robomimic.utils.rby1_paths import new_output
import statistics

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from robomimic.utils import file_utils as F, obs_utils as O
from robomimic.utils.rby1_cartesian import rotation6d_to_matrix
from robomimic.utils.rby1_evaluation import RGB_KEYS, LOWDIM_KEYS, atomic_json, fingerprint, write_csv
from robomimic.scripts.evaluate_rby1_cartesian_checkpoints import check_policy, seed_rollout


def distribution(values):
    a=np.asarray(values)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.percentile(a,90)),
                max=float(a.max()),min=float(a.min()),p10=float(np.percentile(a,10)))


def summarize(rows):
    result=[]
    for epoch,t in sorted({(r['epoch'],r['timestep']) for r in rows}):
        group=[r for r in rows if r['epoch']==epoch and r['timestep']==t]
        p=np.array([r['position_error_mm'] for r in group]);a=np.array([r['orientation_error_deg'] for r in group])
        offsets=np.array([r['phase_offset'] for r in group]);nearest=np.array([r['nearest_gt_timestep'] for r in group])
        result.append(dict(epoch=epoch,timestep=t,samples=len(group),position_error_mm=distribution(p),orientation_error_deg=distribution(a),
            position_lt25_fraction=float((p<25).mean()),position_lt50_fraction=float((p<50).mean()),
            position_gt100_fraction=float((p>100).mean()),position_gt200_fraction=float((p>200).mean()),
            orientation_lt5_fraction=float((a<5).mean()),orientation_gt20_fraction=float((a>20).mean()),
            phase_offset=distribution(offsets),absolute_phase_offset=distribution(np.abs(offsets)),
            nearest_gt_timestep=distribution(nearest),
            phase_offset_gt24_fraction=float((offsets>24).mean()),phase_offset_lt_minus16_fraction=float((offsets< -16).mean()),
            nearest_match_position_error_mm=distribution([r['nearest_gt_position_error_mm'] for r in group]),
            nearest_match_orientation_error_deg=distribution([r['nearest_gt_orientation_error_deg'] for r in group]),
            xyz_spread_mm=(np.ptp(np.array([r['predicted_xyz'] for r in group]),axis=0)*1000).tolist(),
            best_seed=min(group,key=lambda r:r['position_error_mm'])['seed'],worst_seed=max(group,key=lambda r:r['position_error_mm'])['seed']))
    return result


def observation_ambiguity(g,truth,out):
    """Lowdim neighbors and downsampled pixel RMSE, not learned visual feature similarity."""
    positions=g['obs/'+LOWDIM_KEYS[0]][:];quats=g['obs/'+LOWDIM_KEYS[1]][:]
    rotations=Rotation.from_quat(quats);command_rot=Rotation.from_matrix(rotation6d_to_matrix(truth[:,3:]))
    rows=[]
    for t in [16,24]:
        distance=np.linalg.norm(positions-positions[t],axis=1)*1000
        orientation=np.rad2deg((rotations[t].inv()*rotations).magnitude())
        candidates=np.flatnonzero((distance<=10)&(orientation<=5))
        # Near-identical two-frame lowdim histories, and a small visual comparison.
        initial={k:g['obs/'+k][t].astype(float).reshape(32,4,32,4,3).mean((1,3)) for k in RGB_KEYS}
        for n in candidates:
            if n==t:continue
            rms={}
            for key in RGB_KEYS:
                image=g['obs/'+key][n].astype(float).reshape(32,4,32,4,3).mean((1,3))
                rms[key]=float(np.sqrt(np.mean((image-initial[key])**2)))
            hist_pos=max(float(np.linalg.norm(positions[max(0,t-i)]-positions[max(0,n-i)])*1000) for i in [0,1])
            hist_angle=max(float(np.rad2deg((rotations[max(0,t-i)].inv()*rotations[max(0,n-i)]).magnitude())) for i in [0,1])
            cmd_angle=float(np.rad2deg((command_rot[t].inv()*command_rot[n]).magnitude()))
            rows.append(dict(source_timestep=t,neighbor_timestep=int(n),tcp_distance_mm=float(distance[n]),tcp_orientation_deg=float(orientation[n]),
                two_frame_max_tcp_distance_mm=hist_pos,two_frame_max_orientation_deg=hist_angle,
                head_rgb_rmse_0to255=rms[RGB_KEYS[0]],wrist_rgb_rmse_0to255=rms[RGB_KEYS[1]],
                command_position_difference_mm=float(np.linalg.norm(truth[t,:3]-truth[n,:3])*1000),command_orientation_difference_deg=cmd_angle))
    atomic_json(out/'observation_neighbors.json',rows)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--dataset',required=True);parser.add_argument("--output-dir",default=str(new_output("CloseSingleDoor", "diagnostics", "rby1_offline_sampling")))
    args=parser.parse_args();out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    run=Path(args.run_dir).resolve();dataset=Path(args.dataset).expanduser().resolve()
    checkpoints={e:run/'models'/f'model_epoch_{e}.pth' for e in [3,5,8,10,12,15]}
    identities={str(p):fingerprint(p) for p in [dataset,*checkpoints.values()]}
    fields=['epoch','timestep','seed','predicted_action9','predicted_xyz','predicted_quat_xyzw','executed_chunk8',
        'position_error_mm','orientation_error_deg','nearest_gt_timestep','phase_offset',
        'nearest_gt_position_error_mm','nearest_gt_orientation_error_deg','position_only_nearest_timestep',
        'position_only_phase_offset','position_only_nearest_error_mm','displacement_from_current_tcp_mm']
    all_rows=[];states=[0,8,16,24,32,64,128,256]
    with h5py.File(dataset,'r') as f:
        g=f['data/demo_10'];truth=g['actions_eef_cmd'][:];metadata=json.loads(f['data'].attrs['env_args'])
        gt_rot=Rotation.from_matrix(rotation6d_to_matrix(truth[:,3:]));gt_quat=gt_rot.as_quat()
        observations={t:{k:np.stack([O.process_obs(g['obs/'+k][max(0,t-1)],obs_modality='rgb' if k in RGB_KEYS else 'low_dim'),
                                    O.process_obs(g['obs/'+k][t],obs_modality='rgb' if k in RGB_KEYS else 'low_dim')]) for k in RGB_KEYS+LOWDIM_KEYS} for t in states}
        def collect(epoch,times,count):
            policy,ckpt=F.policy_from_checkpoint(ckpt_path=str(checkpoints[epoch]),device=torch.device('cuda'),verbose=False)
            check_policy(policy,ckpt,metadata)
            calls=[0];hook=policy.policy.ema.averaged_model['policy']['obs_encoder'].register_forward_hook(lambda *unused:calls.__setitem__(0,calls[0]+1))
            initial_calls=calls[0]
            try:
                for t in times:
                    for seed in range(count):
                        policy.start_episode();seed_rollout(seed)
                        with torch.no_grad():chunk=np.stack([policy(observations[t]) for _ in range(8)])
                        assert chunk.shape==(8,9) and np.isfinite(chunk).all() and not policy.policy.action_queue
                        pred=chunk[0];rot=Rotation.from_matrix(rotation6d_to_matrix(pred[3:]));quat=rot.as_quat()
                        d=np.linalg.norm(truth[:,:3]-pred[:3],axis=1)*1000
                        angles=np.rad2deg(2*np.arccos(np.clip(np.abs(gt_quat@quat),0,1)))
                        # Same 50 mm / 30 deg scale as the previous single-rollout trace.
                        nearest=int(np.argmin(d/50+angles/30));pos_nearest=int(np.argmin(d))
                        row=dict(epoch=epoch,timestep=t,seed=seed,predicted_action9=pred.tolist(),predicted_xyz=pred[:3].tolist(),
                            predicted_quat_xyzw=quat.tolist(),executed_chunk8=chunk.tolist(),position_error_mm=float(d[t]),orientation_error_deg=float(angles[t]),
                            nearest_gt_timestep=nearest,phase_offset=nearest-t,nearest_gt_position_error_mm=float(d[nearest]),
                            nearest_gt_orientation_error_deg=float(angles[nearest]),position_only_nearest_timestep=pos_nearest,
                            position_only_phase_offset=pos_nearest-t,position_only_nearest_error_mm=float(d[pos_nearest]),
                            displacement_from_current_tcp_mm=float(np.linalg.norm(pred[:3]-observations[t][LOWDIM_KEYS[0]][-1])*1000))
                        all_rows.append(row)
                    print('SAMPLED',epoch,t,count,flush=True)
                    atomic_json(out/'summary_progress.json',summarize(all_rows))
                    # CSV stores vector columns as JSON, rather than ambiguous Python repr.
                    csv_rows=[{k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()} for row in all_rows]
                    write_csv(out/'samples.csv',csv_rows,fields)
                assert calls[0]-initial_calls==len(times)*count,'Exactly one EMA plan required per eight-action sample'
            finally:hook.remove()
            del policy,ckpt;gc.collect();torch.cuda.empty_cache()
        collect(15,states,50)
        epoch15=summarize(all_rows)
        gate=any(r['timestep'] in [16,24] and (r['position_gt100_fraction']>=.1 or
                    (r['absolute_phase_offset']['p90']>=24 and r['position_lt50_fraction']<=.8)) for r in epoch15)
        atomic_json(out/'checkpoint_comparison_gate.json',dict(run_comparison=gate,
            rule='t16/t24: >=10% errors >100 mm OR p90 absolute phase offset >=24 with >=20% errors >=50 mm'))
        if gate:
            for epoch in [3,5,8,10,12]:collect(epoch,[16,24],20)
            # Reuse epoch15 seeds0..19: same samples, no repeated inference needed.
        neighbors=observation_ambiguity(g,truth,out)
    summary=summarize(all_rows);atomic_json(out/'summary.json',summary)
    comparison_rows=[r for r in all_rows if r['timestep'] in [16,24] and r['seed']<20]
    atomic_json(out/'checkpoint_comparison.json',summarize(comparison_rows) if gate else [])
    assert identities=={str(p):fingerprint(p) for p in [dataset,*checkpoints.values()]}
    atomic_json(out/'metadata.json',dict(checkpoints={e:str(p) for e,p in checkpoints.items()},dataset=str(dataset),demo='demo_10',
        epoch15_states=states,epoch15_seeds=list(range(50)),comparison_states=[16,24],comparison_seeds=list(range(20)),
        comparison_executed=gate,samples=len(all_rows),inputs_unchanged=True,ema=True,obs_frames='[max(0,t-1),t]',
        nearest_metric='position_error_mm/50 + orientation_error_deg/30; position-only nearest also recorded',
        note='Nearest phase labels may be nonunique for repeated commands; physical pose error must also be considered. No simulator was constructed.'))
    print('DONE',len(all_rows),'samples',out,flush=True)


if __name__=='__main__':main()
