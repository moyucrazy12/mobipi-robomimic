"""Gate config creation on six complete saved-model command replays."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from robomimic.utils.rby1_saved_model import create_saved_env
from robomimic.utils.rby1_cartesian import RBY1CartesianAdapter
from robomimic.scripts.replay_rby1_cartesian import door_geometry
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work-dir',required=True,help='Directory containing dataset_report.json')
args=parser.parse_args()
ROOT=Path(args.work_dir).expanduser().resolve()
# Clear a previous gate before replaying; a failed rerun must not authorize config creation.
(ROOT/'replay_gate.json').unlink(missing_ok=True)
report=json.loads((ROOT/'dataset_report.json').read_text());dataset=Path(report['final'])
rows=[]
with h5py.File(dataset,'r') as f:
    metadata=json.loads(f['data'].attrs['env_args'])
    for split in ['train','valid']:
        names=report[split]
        # Spread the selected demos across each numerically sorted split.
        selected=[names[i] for i in np.linspace(0,len(names)-1,3,dtype=int)]
        for name in selected:
            g=f['data/'+name];env,verification=create_saved_env(metadata,g,render=False)
            try:
                adapter=RBY1CartesianAdapter(env)
                hinge,adr,_=door_geometry(env,json.loads(g.attrs['ep_meta']))
                model=env.sim.model._model;joint=model.joint(hinge).id
                assert 'microjoint' in hinge and env.behavior=='close'
                assert model.jnt_range[joint,0]<-1.5 and abs(model.jnt_range[joint,1])<1e-6
                initial=float(env.sim.data.qpos[adr]);errors=[];history=[];success_t=None
                actions=g['actions_eef_cmd'][:];reference=g['obs/robot0_base_to_left_eef_pos'][:]
                for t,a in enumerate(actions):
                    env.step(adapter(a))
                    tcp=np.linalg.inv(adapter.site_transform(adapter.base_site))@adapter.site_transform(adapter.tcp_site)
                    angle=float(env.sim.data.qpos[adr]);success=angle>=-0.05*np.pi/2
                    if success and success_t is None:success_t=t
                    if t+1<len(reference):errors.append(float(np.linalg.norm(tcp[:3,3]-reference[t+1])))
                    history.append([t,*tcp[:3,3],angle])
                    if t%500==0:print('REPLAY',split,name,t,'hinge',angle,flush=True)
                row=dict(demo=name,split=split,steps=len(actions),success=bool(success),first_success_timestep=success_t,initial_hinge_rad=initial,final_hinge_rad=angle,hinge_name=hinge,hinge_qpos_address=adr,mean_tcp_path_error_m=float(np.mean(errors)),max_tcp_path_error_m=float(np.max(errors)),path_reference='recorded achieved base-relative grip TCP at t+1, excluding unavailable final next observation',verification=verification)
                rows.append(row)
                np.savetxt(ROOT/(name+'_replay.csv'),history,delimiter=',',header='timestep,tcp_x_base,tcp_y_base,tcp_z_base,microwave_hinge_rad',comments='')
                (ROOT/'replay_report.json').write_text(json.dumps(rows,indent=2)+'\n')
                print('RESULT',json.dumps(row),flush=True)
                assert success,'STOP: command replay did not reproduce successful behavior; no config or debug training'
            finally:env.close()
assert len(rows)==6 and all(r['success'] for r in rows)
(ROOT/'replay_gate.json').write_text(json.dumps(dict(passed=True,demos=6,dataset=str(dataset),source_sha256=report["source_sha256"]),indent=2)+'\n')
