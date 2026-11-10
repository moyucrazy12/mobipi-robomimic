"""Copy original RB-Y1 episodes and add same-timestep commanded TCP targets.

The original file and existing achieved-target derivative are read-only.
Conversion and replay share source_command_as_cartesian. Saved states supply
only frame transforms, never the commanded TCP pose.
"""
import argparse
import hashlib
import json
from pathlib import Path

from robomimic.utils.rby1_paths import new_output
import shutil

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from robomimic.utils.rby1_cartesian import RBY1CartesianAdapter, source_command_as_cartesian, rotation6d_to_matrix
from robomimic.scripts.replay_rby1_cartesian import diagnostic_env, controller_snapshot


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def motion(actions):
    xyz = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1)
    rotations = Rotation.from_matrix(rotation6d_to_matrix(actions[:, 3:]))
    angles = np.rad2deg((rotations[1:] * rotations[:-1].inv()).magnitude())
    return xyz, angles


def distribution(values):
    return dict(zip(('mean', 'median', 'p90', 'max'), map(float,
        (values.mean(), np.median(values), np.percentile(values, 90), values.max()))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument("--output", default=str(new_output("CloseSingleDoor", "demos", "rby1_commanded_cartesian", ".hdf5")))
    parser.add_argument('--old-dataset', required=True)
    parser.add_argument("--report", default=str(new_output("CloseSingleDoor", "diagnostics", "rby1_command_conversion", ".json")))
    args = parser.parse_args()
    import robocasa  # noqa: F401
    source, output, old = (Path(x).expanduser().resolve() for x in
                           (args.dataset, args.output, args.old_dataset))
    if output.exists() or output in (source, old):
        raise ValueError('Output must be a new, distinct file')
    report_path = Path(args.report).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    source_hash, old_hash = digest(source), digest(old)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    rows, all_targets, cmd_jumps, old_jumps = [], [], [[], []], [[], []]
    maxima = dict(packed_left_difference=0., position_difference_m=0., orientation_difference_deg=0.,
                  full_packed_difference_with_recorded_holds=0., default_hold_difference=0.)
    report = {'source': str(source), 'dataset': str(output), 'temporal_alignment': 'original actions[t], no shift',
              'dtype': 'float64 (native training converts inputs to float32)', 'round_trip_commands': 0}
    with h5py.File(source, 'r') as src, h5py.File(old, 'r') as achieved, h5py.File(output, 'r+') as dst:
        names = sorted(src['data'], key=lambda n: int(n.split('_')[1]))
        metadata = json.loads(src['data'].attrs['env_args'])
        masks = {k: [x.decode() for x in src['mask/'+k][:]] for k in ('train', 'valid')}
        assert len(names) == 50 and len(masks['train']) == 45 and len(masks['valid']) == 5
        for name in names:
            g = src['data/'+name]; states, original = g['states'][:], g['actions'][:]
            env = diagnostic_env(metadata, metadata['env_kwargs']['controller_configs'],
                                 g.attrs['model_file'], json.loads(g.attrs['ep_meta']))
            try:
                env.sim.reset(); env.sim.set_state_from_flattened(states[0]); env.sim.forward()
                adapter = RBY1CartesianAdapter(env)
                snapshot = controller_snapshot(env)
                assert env.action_dim == 31
                assert tuple(snapshot['ik_action_slices']['left']) == (6, 12)
                assert adapter.tcp_site == 'gripper0_left_grip_site'
                assert adapter.reference_site == 'gripper0_left_ft_frame'
                if not rows:
                    report['controller'] = snapshot
                    report['tcp_to_reference'] = adapter.tcp_to_reference.tolist()
                targets = np.empty((len(original), 9), dtype=np.float64)
                slices = adapter.controller._whole_body_controller_action_split_indexes
                fixed_holds = {p: a.copy() for p, a in adapter.holds.items()}
                for t, action in enumerate(original):
                    # Restoring t is only needed for the world->base transform.
                    env.sim.set_state_from_flattened(states[t]); env.sim.forward()
                    targets[t] = source_command_as_cartesian(adapter, action)
                    adapter.holds = fixed_holds
                    packed = adapter(targets[t])
                    maxima['packed_left_difference'] = max(maxima['packed_left_difference'], float(np.max(np.abs(packed[6:12]-action[6:12]))))
                    maxima['default_hold_difference'] = max(maxima['default_hold_difference'], float(np.max(np.abs(packed-action))))
                    maxima['position_difference_m'] = max(maxima['position_difference_m'], float(np.linalg.norm(packed[6:9]-action[6:9])))
                    orientation = Rotation.from_rotvec(packed[9:12]) * Rotation.from_rotvec(action[9:12]).inv()
                    maxima['orientation_difference_deg'] = max(maxima['orientation_difference_deg'], float(np.rad2deg(orientation.magnitude())))
                    # Isolate representation/packing from fixed nuisance holds.
                    adapter.holds = {p: action[slice(*slices[p])].copy() for p in fixed_holds}
                    packed = adapter(targets[t])
                    maxima['full_packed_difference_with_recorded_holds'] = max(maxima['full_packed_difference_with_recorded_holds'],float(np.max(np.abs(packed-action))))
                if not np.isfinite(targets).all():
                    raise ValueError('Nonfinite command targets')
                stored = dst['data/'+name].create_dataset('actions_eef_cmd', data=targets)
                stored.attrs.update(frame='robot_base_center', tcp_site=adapter.tcp_site,
                    ik_reference_site=adapter.reference_site, quaternion_order='xyzw',
                    rotation6d='concatenate(R[:,0], R[:,1])', temporal_alignment='original actions[t], no shift',
                    source_action_key='actions', definition='inv(T_world_base[t]) @ T_world_ref_command[t] @ inv(T_tcp_ref)')
                for jump, values in zip(cmd_jumps, motion(targets)): jump.append(values)
                for jump, values in zip(old_jumps, motion(achieved['data/'+name+'/actions_eef'][:])): jump.append(values)
                all_targets.append(targets)
                rows.append({'demo':name,'steps':len(targets),'shape':list(targets.shape)})
                print(f'{name}: {len(targets)} command targets; round-trip max {maxima["packed_left_difference"]:.3g}',flush=True)
            finally:
                env.close()
        # Verify every original dataset and attribute, including images and masks.
        def verify(name, obj):
            other = dst[name] if name else dst
            assert set(obj.attrs) == set(other.attrs)
            for key in obj.attrs: assert np.array_equal(obj.attrs[key], other.attrs[key])
            if isinstance(obj, h5py.Dataset):
                assert obj.shape == other.shape and obj.dtype == other.dtype
                if obj.ndim == 0: assert np.array_equal(obj[()], other[()])
                else:
                    for start in range(0,len(obj),128):
                        assert np.array_equal(obj[start:start+128],other[start:start+128]), name
        verify('',src); src.visititems(verify)
        report['original_content_preserved'] = True
        report['masks'] = masks
    targets = np.concatenate(all_targets)
    report.update(demonstrations=len(rows), total_timesteps=len(targets), episodes=rows,
        xyz_min=targets[:,:3].min(0).tolist(), xyz_max=targets[:,:3].max(0).tolist(),
        rotation6d_min=targets[:,3:].min(0).tolist(), rotation6d_max=targets[:,3:].max(0).tolist(),
        round_trip_commands=len(targets), round_trip_maxima=maxima,
        command_jumps=dict(position_m=distribution(np.concatenate(cmd_jumps[0])), orientation_deg=distribution(np.concatenate(cmd_jumps[1]))),
        achieved_jumps=dict(position_m=distribution(np.concatenate(old_jumps[0])), orientation_deg=distribution(np.concatenate(old_jumps[1]))))
    assert maxima['packed_left_difference'] < 1e-10 and maxima['orientation_difference_deg'] < 1e-8
    assert source_hash == digest(source) and old_hash == digest(old)
    report['source_sha256'] = source_hash; report['old_dataset_sha256'] = old_hash
    report['source_and_old_unchanged'] = True
    def encode(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    report_path.write_text(json.dumps(report, indent=2,default=encode)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('controller','episodes','masks')},indent=2))


if __name__ == '__main__':
    main()
