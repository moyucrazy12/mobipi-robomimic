"""Offline Cartesian DP diagnostics without simulation or inference changes."""
import argparse
import csv
import json
from pathlib import Path

from robomimic.utils.rby1_paths import new_output
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import DataLoader

from robomimic.utils import file_utils as F, obs_utils as O, train_utils as T
from robomimic.utils.dataset import action_stats_to_normalization_stats
from robomimic.utils.rby1_cartesian import quaternion_to_rotation6d, rotation6d_to_matrix


def metrics(values):
    values = np.asarray(values)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.percentile(values, 90)), max=float(values.max()))


def orientation_error(first, second):
    return np.rad2deg((Rotation.from_matrix(rotation6d_to_matrix(first)).inv() *
                      Rotation.from_matrix(rotation6d_to_matrix(second))).magnitude())


def jumps(actions):
    return {'xyz_m': metrics(np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1)),
            'orientation_deg': metrics(orientation_error(actions[:-1, 3:], actions[1:, 3:]))}


def door_motion_indices(demo):
    # XML traversal gives qpos addresses without loading MuJoCo. Door motion is
    # a contact proxy, not a saved contact flag (those were not recorded).
    root = ET.fromstring(demo.attrs['model_file'])
    address = 0
    door_address = None
    fixture = json.loads(demo.attrs['ep_meta'])['fixture_refs']['door_fxtr']
    for element in root.find('worldbody').iter():
        if element.tag in ('joint', 'freejoint'):
            if fixture in element.get('name', '') and element.get('type', 'hinge') == 'hinge':
                if door_address is not None:
                    raise ValueError('Ambiguous door hinge')
                door_address = address
            kind = element.get('type', 'free' if element.tag == 'freejoint' else 'hinge')
            address += {'free': 7, 'ball': 4}.get(kind, 1)
    if door_address is None:
        raise ValueError('Door hinge not found')
    angles = demo['states'][:, 1 + door_address]
    return np.flatnonzero(np.abs(np.diff(angles)) > 0.0005) + 1


def samples_for_demo(demo):
    length = len(demo['actions_eef'])
    moving = door_motion_indices(demo)
    indices = set(np.linspace(0, length - 1, 12, dtype=int).tolist())
    if len(moving):
        indices.update(moving[np.linspace(0, len(moving) - 1, 8, dtype=int)].tolist())
    for index in np.linspace(0, length - 1, 100, dtype=int):
        if len(indices) >= 20:
            break
        indices.add(int(index))
    return [(t, 'door_motion_contact_proxy' if t in set(moving) else
             'initial' if t == 0 else 'late_task' if t >= length * .85 else 'free_or_transition')
            for t in sorted(indices)]


def alignment(policy, checkpoint, dataset, config):
    train, valid = T.load_data_for_training(config, checkpoint['shape_metadata']['all_obs_keys'])
    stats = train.get_action_normalization_stats()
    valid.set_action_normalization_stats(stats)
    child = getattr(train, 'datasets', [train])[0]
    demo = child.demos[0]
    t = 100
    index = child._demo_id_to_start_indices[demo] + t
    # Actual one-sample DataLoader batch, then native training preprocessing.
    batch = next(iter(DataLoader(train, batch_size=1, sampler=[index], num_workers=0)))
    processed = policy.policy.process_batch_for_training(batch)
    processed = policy.policy.postprocess_batch_for_training(processed, obs_normalization_stats=policy.obs_normalization_stats)
    action_indices = np.arange(t - 1, t + 15)
    gt = dataset[f'data/{demo}/actions_eef'][action_indices]
    source_pose = np.concatenate((dataset[f'data/{demo}/obs/robot0_base_to_left_eef_pos'][action_indices + 1],
        quaternion_to_rotation6d(dataset[f'data/{demo}/obs/robot0_base_to_left_eef_quat_site'][action_indices + 1])), axis=-1)
    np.testing.assert_allclose(gt, source_pose, atol=1e-7)
    denormalized = O.unnormalize_dict({'actions_eef': batch['actions'][0].numpy()}, normalization_stats=stats)['actions_eef']
    np.testing.assert_allclose(denormalized, gt, atol=2e-7)
    for key in config.observation.modalities.obs.low_dim:
        np.testing.assert_allclose(processed['obs'][key][0].cpu(), dataset[f'data/{demo}/obs/{key}'][t - 1:t + 1])
    result = {'demo': demo, 'sample_t': t, 'observation_indices_used': [t - 1, t],
              'action_indices_in_model': action_indices.tolist(),
              'pose_indices_in_model': (action_indices + 1).tolist(),
              'executed_prediction_indices': list(range(1, 9)),
              'executed_action_indices': list(range(t, t + 8)),
              'executed_pose_indices': list(range(t + 1, t + 9)),
              'batch_action_shape': list(batch['actions'].shape),
              'processed_obs_shapes': {k: list(v.shape) for k, v in processed['obs'].items()},
              'training_sampler': str(train.get_dataset_sampler()), 'validation_sampler': str(valid.get_dataset_sampler())}
    for ds in getattr(train, 'datasets', [train]) + getattr(valid, 'datasets', [valid]):
        ds.close_and_delete_hdf5_handle()
    return result, stats


def diagnose(args):
    run = Path(args.run).expanduser()
    if not (run / 'models').is_dir():
        candidates = sorted(path for path in run.glob('[0-9]' * 14) if (path / 'models').is_dir())
        if len(candidates) != 1:
            raise ValueError(f'Specify a timestamped --run; expected one unambiguous run in {run}')
        run = candidates[0]
    print('Benchmark training run:', run.resolve(), flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    summary = {'setup': {'run': str(run.resolve()), 'checkpoint_epochs': args.epochs,
                         'seed_start': args.seed, 'dataset': str(Path(args.dataset).expanduser().resolve())}}
    with h5py.File(Path(args.dataset).expanduser(), 'r', swmr=True, libver='latest') as dataset:
        demos = [x.decode() for x in dataset['mask/valid'][:]]
        samples = [(demo, t, phase) for demo in demos for t, phase in samples_for_demo(dataset[f'data/{demo}'])]
        summary['setup']['validation_states'] = samples
        for epoch in args.epochs:
            policy, checkpoint = F.policy_from_checkpoint(device=torch.device(args.device),
                ckpt_path=str(run / 'models' / f'model_epoch_{epoch}.pth'))
            config = policy.policy.global_config
            assert checkpoint['shape_metadata']['ac_dim'] == 9
            keys = checkpoint['shape_metadata']['all_obs_keys']
            if epoch == args.epochs[0]:
                summary['alignment'], fitted = alignment(policy, checkpoint, dataset, config)
                train_actions = np.concatenate([dataset[f'data/{x.decode()}/actions_eef'][:] for x in dataset['mask/train'][:]])
                bounds = (train_actions.min(0), train_actions.max(0))
                native_fit = action_stats_to_normalization_stats({'actions_eef': {'min': bounds[0][None], 'max': bounds[1][None]}}, config.train.action_config)
                for key in ('scale', 'offset'):
                    np.testing.assert_array_equal(native_fit['actions_eef'][key], policy.action_normalization_stats['actions_eef'][key])
                    np.testing.assert_array_equal(fitted['actions_eef'][key], policy.action_normalization_stats['actions_eef'][key])
                summary['normalization'] = {'checkpoint': {k: v.tolist() for k, v in policy.action_normalization_stats['actions_eef'].items()}, 'training_min': bounds[0].tolist(), 'training_max': bounds[1].tolist(), 'exact_match': True}
                summary['demonstration_jumps'] = {demo: jumps(dataset[f'data/{demo}/actions_eef'][:]) for demo in demos}
            for key in ('scale', 'offset'):
                np.testing.assert_array_equal(native_fit['actions_eef'][key], policy.action_normalization_stats['actions_eef'][key])
            counts = {'raw_encoder': 0, 'ema_encoder': 0, 'raw_diffusion': 0, 'ema_diffusion': 0}
            handles = []
            for name, module in [('raw_encoder', policy.policy.nets['policy']['obs_encoder']),
                                 ('ema_encoder', policy.policy.ema.averaged_model['policy']['obs_encoder']),
                                 ('raw_diffusion', policy.policy.nets['policy']['noise_pred_net']),
                                 ('ema_diffusion', policy.policy.ema.averaged_model['policy']['noise_pred_net'])]:
                def hook(module, inputs, result, name=name):
                    counts[name] += 1
                handles.append(module.register_forward_hook(hook))
            captured = []
            original_step = policy.policy.noise_scheduler.step
            def capture_step(*positional, **kwargs):
                result = original_step(*positional, **kwargs)
                captured.append(result.prev_sample.detach().cpu().numpy())
                return result
            policy.policy.noise_scheduler.step = capture_step
            predictions, normalized, truths, holds, displacements, gt_displacements = [], [], [], [], [], []
            boundaries_xyz, boundaries_rot, gt_boundaries_xyz, gt_boundaries_rot = [], [], [], []
            rows = []
            for sample_id, (demo, t, phase) in enumerate(samples):
                group = dataset[f'data/{demo}']; length = len(group['actions_eef'])
                history = [max(0, t - 1), t]
                ob = {key: O.process_obs(np.stack([group[f'obs/{key}'][i] for i in history]), obs_key=key) for key in keys}
                gt_indices = np.clip(np.arange(t - 1, t + 15), 0, length - 1)
                gt = group['actions_eef'][:][gt_indices]
                current = np.concatenate((group['obs/robot0_base_to_left_eef_pos'][t], quaternion_to_rotation6d(group['obs/robot0_base_to_left_eef_quat_site'][t])))
                policy.start_episode(); torch.manual_seed(args.seed + sample_id); torch.cuda.manual_seed_all(args.seed + sample_id)
                captured.clear()
                with torch.no_grad():
                    executed = np.stack([policy(ob) for _ in range(8)])
                norm = captured[-1][0]
                full = O.unnormalize_dict({'actions_eef': norm}, normalization_stats=policy.action_normalization_stats)['actions_eef']
                np.testing.assert_allclose(executed, full[1:9], atol=1e-7)
                next_t = min(t + 8, length - 1)
                next_ob = {key: O.process_obs(np.stack([group[f'obs/{key}'][i] for i in [max(0, next_t - 1), next_t]]), obs_key=key) for key in keys}
                # Empty native queue now replans from the next recorded history.
                with torch.no_grad():
                    next_first = policy(next_ob)
                boundaries_xyz.append(float(np.linalg.norm(next_first[:3] - executed[-1, :3])))
                boundaries_rot.append(float(orientation_error(executed[-1:, 3:], next_first[None, 3:])[0]))
                next_gt = group['actions_eef'][next_t]
                gt_boundaries_xyz.append(float(np.linalg.norm(next_gt[:3] - gt[8, :3])))
                gt_boundaries_rot.append(float(orientation_error(gt[8:9, 3:], next_gt[None, 3:])[0]))
                predictions.append(executed); normalized.append(norm); truths.append(gt[1:9]); holds.append(np.repeat(current[None], 8, axis=0))
                displacements.append(float(np.linalg.norm(executed[0, :3] - current[:3])))
                gt_displacements.append(float(np.linalg.norm(gt[1, :3] - current[:3])))
                for j in range(16):
                    rows.append({'demo': demo, 't': t, 'phase': phase, 'prediction_index': j, 'executed': 1 <= j <= 8,
                                 'action_source_t': int(gt_indices[j]), 'pose_source_t': min(int(gt_indices[j]) + 1, length - 1),
                                 'predicted_xyz': json.dumps(full[j, :3].tolist()), 'truth_xyz': json.dumps(gt[j, :3].tolist()),
                                 'position_error_m': float(np.linalg.norm(full[j, :3] - gt[j, :3])),
                                 'orientation_error_deg': float(orientation_error(full[j:j+1, 3:], gt[j:j+1, 3:])[0])})
                if demo == 'demo_5' and t == 0:
                    summary[f'epoch_{epoch}_initial'] = {'current_pose_9d': current.tolist(), 'first_target_displacement_m': displacements[-1], 'gt_next_displacement_m': gt_displacements[-1], 'sequence': rows[-16:]}
                if sample_id % 20 == 0:
                    print(f'Epoch {epoch}: {sample_id + 1}/{len(samples)} offline states', flush=True)
            pred, truth, hold, norm = map(np.asarray, (predictions, truths, holds, normalized))
            position = np.linalg.norm(pred[..., :3] - truth[..., :3], axis=-1)
            orientation = orientation_error(pred[..., 3:].reshape(-1, 6), truth[..., 3:].reshape(-1, 6))
            hold_position = np.linalg.norm(hold[..., :3] - truth[..., :3], axis=-1)
            hold_orientation = orientation_error(hold[..., 3:].reshape(-1, 6), truth[..., 3:].reshape(-1, 6))
            within_xyz = np.linalg.norm(np.diff(pred[..., :3], axis=1), axis=-1)
            within_rot = orientation_error(pred[:, :-1, 3:].reshape(-1, 6), pred[:, 1:, 3:].reshape(-1, 6))
            outside = (pred < bounds[0]) | (pred > bounds[1])
            summary[f'epoch_{epoch}'] = {'states': len(samples), 'executed_actions': int(pred.shape[0] * 8),
                'position_error_m': metrics(position), 'orientation_error_deg': metrics(orientation),
                'hold_position_error_m': metrics(hold_position), 'hold_orientation_error_deg': metrics(hold_orientation),
                'within_chunk_xyz_m': metrics(within_xyz), 'within_chunk_orientation_deg': metrics(within_rot),
                'replan_boundary_xyz_m': metrics(boundaries_xyz), 'replan_boundary_orientation_deg': metrics(boundaries_rot),
                'gt_within_chunk_xyz_m': metrics(np.linalg.norm(np.diff(truth[..., :3], axis=1), axis=-1)),
                'gt_within_chunk_orientation_deg': metrics(orientation_error(truth[:, :-1, 3:].reshape(-1, 6), truth[:, 1:, 3:].reshape(-1, 6))),
                'gt_replan_boundary_xyz_m': metrics(gt_boundaries_xyz), 'gt_replan_boundary_orientation_deg': metrics(gt_boundaries_rot),
                'first_target_displacement_m': metrics(displacements), 'gt_next_displacement_m': metrics(gt_displacements),
                'normalized_min': norm.min((0,1)).tolist(), 'normalized_max': norm.max((0,1)).tolist(),
                'denormalized_xyz_min': pred[..., :3].min((0,1)).tolist(), 'denormalized_xyz_max': pred[..., :3].max((0,1)).tolist(),
                'outside_training_range_component_fraction': float(outside.mean()),
                'outside_training_range_action_fraction': float(outside.any(-1).mean()),
                'outside_training_range_component_fraction_tolerance_1e_minus_6': float(((pred < bounds[0] - 1e-6) | (pred > bounds[1] + 1e-6)).mean()),
                'outside_training_range_gt_component_fraction': float(((truth < bounds[0]) | (truth > bounds[1])).mean()),
                'outside_training_range_gt_action_fraction': float(((truth < bounds[0]) | (truth > bounds[1])).any(-1).mean()),
                'runtime_weight_hooks': counts}
            with (output / f'epoch_{epoch}_predictions.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            np.savez_compressed(output / f'epoch_{epoch}_arrays.npz', predictions=pred, truths=truth, normalized=norm, holds=hold)
            for handle in handles:
                handle.remove()
            policy.policy.noise_scheduler.step = original_step
            (output / 'summary.json').write_text(json.dumps(summary, indent=2))
            del policy, checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print('Offline diagnostic complete:', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Timestamped run, or an experiment directory containing exactly one run')
    parser.add_argument("--output", default=str(new_output("CloseSingleDoor", "diagnostics", "rby1_cartesian_dp")))
    parser.add_argument('--dataset', default='~/Downloads/demo_im128_head_wrist_cartesian.hdf5')
    parser.add_argument('--device', default='cuda'); parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--epochs', type=int, nargs='+', default=[34, 50],
                        help='Saved checkpoint epochs to benchmark; defaults preserve the original comparison')
    diagnose(parser.parse_args())
