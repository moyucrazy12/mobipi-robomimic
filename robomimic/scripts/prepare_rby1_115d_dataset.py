"""Preserve 115 source episodes; render saved states and cache validated commands."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import h5py
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation
import robosuite.macros as macros
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
from robomimic.utils.rby1_cartesian import RBY1CartesianAdapter, source_command_as_cartesian, rotation6d_to_matrix
from robomimic.utils.rby1_saved_model import create_saved_env
CAMERAS=['robot0_head_camera','robot0_left_eye_in_hand']
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--phase',choices=['smoke','render','finalize'],required=True)
parser.add_argument('--source',required=True,help='Read-only original 115-demo HDF5')
parser.add_argument('--rendered',required=True,help='New rendered HDF5 output')
parser.add_argument('--output',required=True,help='New Cartesian-command HDF5 output')
parser.add_argument('--work-dir',required=True,help='Local reports, smoke file and command cache')
args=parser.parse_args()
SOURCE=Path(args.source).expanduser().resolve()
RENDERED=Path(args.rendered).expanduser().resolve()
FINAL=Path(args.output).expanduser().resolve()
ROOT=Path(args.work_dir).expanduser().resolve()
if len({SOURCE,RENDERED,FINAL})!=3:
    raise ValueError('Source and derivative paths must be distinct')
ROOT.mkdir(parents=True,exist_ok=True)

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(name,value):(ROOT/name).write_text(json.dumps(value,indent=2)+'\n')

def original_preserved(src,dst,allow_env_metadata=False):
    def check(name,obj):
        other=dst[name] if name else dst
        for key in obj.attrs:
            if allow_env_metadata and name=='data' and key=='env_args':
                assert other.attrs['source_env_args']==obj.attrs[key];continue
            assert np.array_equal(obj.attrs[key],other.attrs[key]),(name,key)
        if isinstance(obj,h5py.Dataset):
            assert other.shape==obj.shape and other.dtype==obj.dtype,name
            if obj.ndim==0:assert np.array_equal(obj[()],other[()]),name
            else:
                for start in range(0,len(obj),128):assert np.array_equal(obj[start:start+128],other[start:start+128]),name
    check('',src);src.visititems(check)

if args.phase in ['smoke','render']:
    source_hash=digest(SOURCE)
    with h5py.File(SOURCE,'r') as src:
        names=sorted(src['data'],key=lambda n:int(n.split('_')[1]));assert len(names)==115
        metadata=json.loads(src['data'].attrs['env_args'])
        assert 'mask' not in src
        destination=ROOT/'smoke.hdf5' if args.phase=='smoke' else RENDERED
        if destination.exists():raise FileExistsError(destination)
        if args.phase=='smoke':
            with h5py.File(destination,'w') as dst:
                data=dst.create_group('data');src.copy('data/'+names[0],data,name=names[0])
                for k,v in src['data'].attrs.items():data.attrs[k]=v
                data.attrs['total']=len(src['data/'+names[0]+'/actions'])
            names=names[:1]
        else:
            smoke=json.loads((ROOT/'smoke_report.json').read_text())
            assert smoke['passed'] and smoke['source']==str(SOURCE) and smoke['source_sha256']==source_hash
            shutil.copyfile(SOURCE,destination)
        rows=[];start=time.perf_counter()
        with h5py.File(destination,'r+') as dst:
            for name in names:
                g=src['data/'+name];target=dst['data/'+name];n=len(g['actions'])
                states=g['states'][:];actions=g['actions'][:]
                assert len(states)==n and actions.shape==(n,31)
                env,verification=create_saved_env(metadata,g,render=True)
                try:
                    model=env.sim.model._model
                    camera_names=[model.camera(i).name for i in range(model.ncam)]
                    assert all(c in camera_names for c in CAMERAS)
                    adapter=RBY1CartesianAdapter(env)
                    assert adapter.tcp_site=='gripper0_left_grip_site' and adapter.reference_site=='gripper0_left_ft_frame'
                    base_tcp=np.linalg.inv(adapter.site_transform(adapter.base_site))@adapter.site_transform(adapter.tcp_site)
                    assert np.linalg.norm(base_tcp[:3,3]-g['obs/robot0_base_to_left_eef_pos'][0])<1e-6
                    buffers=[np.empty((64,128,128,3),dtype=np.uint8) for _ in CAMERAS]
                    datasets=[target['obs'].create_dataset(c+'_image',shape=(n,128,128,3),dtype=np.uint8,chunks=(1,128,128,3),compression='lzf') for c in CAMERAS]
                    commands=np.empty((n,9),dtype=np.float64)
                    maxima=np.zeros(3);step_start=time.perf_counter()
                    convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
                    for t in range(n):
                        # Restore same-timestep source state; never integrate or regenerate actions.
                        env.sim.set_state_from_flattened(states[t]);env.sim.forward()
                        commands[t]=source_command_as_cartesian(adapter,actions[t])
                        full=adapter(commands[t])
                        maxima[0]=max(maxima[0],np.max(np.abs(full[6:12]-actions[t,6:12])))
                        maxima[1]=max(maxima[1],np.linalg.norm(full[6:9]-actions[t,6:9]))
                        ori=np.rad2deg((Rotation.from_rotvec(full[9:12]).inv()*Rotation.from_rotvec(actions[t,9:12])).magnitude())
                        maxima[2]=max(maxima[2],ori)
                        for index,camera in enumerate(CAMERAS):
                            frame=env.sim.render(width=128,height=128,camera_name=camera)[::convention][::-1].copy()
                            assert frame.shape==(128,128,3) and frame.dtype==np.uint8
                            buffers[index][t%64]=frame
                            if t==0:
                                imageio.imwrite(ROOT/f'{args.phase}_{name}_{camera}.png',frame)
                                # Compare directly to native camera observable + Robomimic flip.
                                sensors,_=env._create_camera_sensors(camera,128,128,False,None,'image')
                                assert np.array_equal(sensors[0]({})[::-1],frame)
                        if t%64==63 or t==n-1:
                            begin=t-t%64
                            for ds,buffer in zip(datasets,buffers):ds[begin:t+1]=buffer[:t%64+1]
                        if t%500==0:print('RENDER',name,t,'/',n,flush=True)
                    assert np.isfinite(commands).all() and maxima[0]<1e-10 and maxima[2]<1e-8
                    assert np.array_equal(target['actions'][:],actions) and np.array_equal(target['states'][:],states)
                    cache=ROOT/'commands';cache.mkdir(exist_ok=True)
                    np.save(cache/(name+'.npy'),commands)
                    row=dict(demo=name,steps=n,rgb_shapes={c+'_image':list(ds.shape) for c,ds in zip(CAMERAS,datasets)},round_trip_packed_left_max=float(maxima[0]),round_trip_reference_position_max_m=float(maxima[1]),round_trip_reference_orientation_max_deg=float(maxima[2]),seconds=time.perf_counter()-step_start,available_camera_names=camera_names,verification=verification)
                    rows.append(row);dst.flush();write(args.phase+'_progress.json',rows)
                    print('DONE',name,'seconds',row['seconds'],'roundtrip',maxima.tolist(),flush=True)
                finally:env.close()
            original_args=dst['data'].attrs['env_args'];new_meta=json.loads(original_args)
            new_meta['env_kwargs'].update(camera_names=CAMERAS,camera_heights=128,camera_widths=128,use_camera_obs=True,has_offscreen_renderer=True)
            dst['data'].attrs['source_env_args']=original_args
            dst['data'].attrs['env_args']=json.dumps(new_meta)
            if args.phase=='render':original_preserved(src,dst,allow_env_metadata=True)
        assert digest(SOURCE)==source_hash
        report=dict(passed=True,source=str(SOURCE),output=str(destination),source_sha256=source_hash,source_unchanged=True,demos=len(rows),timesteps=sum(r['steps'] for r in rows),seconds=time.perf_counter()-start,rows=rows,image_pipeline='native sim.render, RoboSuite IMAGE_CONVENTION, then Robomimic vertical flip; native observable equality checked at t0',metadata_changes='Only camera names/resolution/use_camera_obs/has_offscreen_renderer in derivative env_args; original env_args retained as source_env_args. All original datasets and attrs preserved.')
        write(args.phase+'_report.json',report)
else:
    render_report=json.loads((ROOT/'render_report.json').read_text());assert render_report['passed'] and render_report['demos']==115
    assert render_report['source']==str(SOURCE) and render_report['output']==str(RENDERED)
    assert digest(SOURCE)==render_report['source_sha256']
    if FINAL.exists():raise FileExistsError(FINAL)
    render_hash=digest(RENDERED);shutil.copyfile(RENDERED,FINAL)
    all_commands=[];position_jumps=[];orientation_jumps=[];split_seed=2026
    with h5py.File(RENDERED,'r') as src,h5py.File(FINAL,'r+') as dst:
        names=sorted(src['data'],key=lambda n:int(n.split('_')[1]))
        assert 'mask' not in dst
        permutation=np.random.default_rng(split_seed).permutation(names)
        valid=sorted(permutation[:15].tolist(),key=lambda n:int(n.split('_')[1]));train=sorted(permutation[15:].tolist(),key=lambda n:int(n.split('_')[1]))
        for name in names:
            g=dst['data/'+name];a=np.load(ROOT/'commands'/(name+'.npy'))
            assert a.shape==(len(g['actions']),9)
            ds=g.create_dataset('actions_eef_cmd',data=a)
            assert np.array_equal(ds[:],a)
            ds.attrs.update(frame='robot_base_center',tcp_site='gripper0_left_grip_site',ik_reference_site='gripper0_left_ft_frame',rotation6d='concatenate(R[:,0], R[:,1])',temporal_alignment='original actions[t], no shift',source_action_key='actions',definition='inv(T_world_base[t]) @ T_world_ref_command[t] @ inv(T_tcp_ref)')
            for key in CAMERAS:assert g['obs/'+key+'_image'].shape==(len(a),128,128,3)
            for key in ['robot0_base_to_left_eef_pos','robot0_base_to_left_eef_quat_site','robot0_joint_pos_cos','robot0_joint_pos_sin','robot0_joint_vel']:assert len(g['obs/'+key])==len(a)
            all_commands.append(a);position_jumps.append(np.linalg.norm(np.diff(a[:,:3],axis=0),axis=1))
            r=Rotation.from_matrix(rotation6d_to_matrix(a[:,3:]));orientation_jumps.append(np.rad2deg((r[:-1].inv()*r[1:]).magnitude()))
        mask=dst.create_group('mask');mask.create_dataset('train',data=np.asarray(train,dtype='S'));mask.create_dataset('valid',data=np.asarray(valid,dtype='S'))
        mask.attrs['split_seed']=split_seed;mask.attrs['split_method']='numpy.random.default_rng(seed).permutation(numerically sorted demo names); first 15 validation'
        original_preserved(src,dst)
    assert digest(RENDERED)==render_hash and digest(SOURCE)==render_report['source_sha256']
    a=np.concatenate(all_commands)
    def stats(v):return dict(mean=float(v.mean()),median=float(np.median(v)),p90=float(np.percentile(v,90)),max=float(v.max()))
    report=dict(source=str(SOURCE),rendered=str(RENDERED),final=str(FINAL),demos=115,timesteps=len(a),split_seed=split_seed,train=train,valid=valid,xyz_min=a[:,:3].min(0).tolist(),xyz_max=a[:,:3].max(0).tolist(),rotation6d_min=a[:,3:].min(0).tolist(),rotation6d_max=a[:,3:].max(0).tolist(),position_jump_m=stats(np.concatenate(position_jumps)),orientation_jump_deg=stats(np.concatenate(orientation_jumps)),round_trip_commands=len(a),max_packed_left_difference=max(r['round_trip_packed_left_max'] for r in render_report['rows']),max_reference_position_difference_m=max(r['round_trip_reference_position_max_m'] for r in render_report['rows']),max_reference_orientation_difference_deg=max(r['round_trip_reference_orientation_max_deg'] for r in render_report['rows']),source_sha256=render_report['source_sha256'],rendered_sha256=render_hash,source_and_rendered_unchanged=True,original_content_preserved=True)
    write('dataset_report.json',report);print(json.dumps(report),flush=True)
