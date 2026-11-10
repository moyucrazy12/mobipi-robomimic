# RB-Y1 Cartesian Diffusion Policy

## Goal

Train a fixed-base manipulation policy for the RB-Y1 robot on RoboCasa
`CloseSingleDoor`, with the intention of later using Mobi-π to select a suitable
mobile-base pose before policy execution.

## Dataset

- 115 successful demonstrations
- 171,258 timesteps
- 100 training / 15 validation demonstrations
- Split seed: 2026
- Failed demonstrations excluded from behavior cloning
- RGB observations rendered at 128 × 128

Training dataset:

`demo_im128_head_wrist_cartesian_cmd.hdf5`

## Policy Inputs

| Input | Description |
|---|---|
| `robot0_head_camera_image` | Head RGB, 128 × 128 × 3 |
| `robot0_left_eye_in_hand_image` | Left wrist RGB, 128 × 128 × 3 |
| `robot0_base_to_left_eef_pos` | TCP position in robot-base frame |
| `robot0_base_to_left_eef_quat_site` | TCP orientation in robot-base frame |

The policy uses two observation frames.

Joint positions and velocities are available in the dataset but are intentionally
not used in the current baseline.

## Action Representation

The policy predicts a 9D **absolute Cartesian TCP target expressed in the robot
base frame**:

`[x, y, z, R[:,0], R[:,1]]`

The first three values are TCP position and the remaining six values are a 6D
rotation representation.

This is an absolute target, not a Cartesian displacement.

The prediction is converted to the IK reference frame using:

```text
T_world_ref = T_world_base @ T_base_tcp @ T_tcp_ref
```

## Reproduce training and evaluation

Run from this repository root with the existing `mobipi` Conda environment
(robosuite, RoboCasa, PyTorch and diffusion dependencies installed):

```bash
conda activate mobipi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
```

Prepare the original saved-model demonstrations and RGB/command derivative using
[the dataset preparation commands](datasets/rby1_115d.rst). Dataset and checkpoint
artifacts are supplied separately. Set `train.data[0].path` in a local copy of the
configuration to your prepared HDF5, then train with:

```bash
python robomimic/scripts/train.py \
  --config robomimic/exps/rby1_close_single_door_cartesian_cmd_115d_15pass.json
```

The checked-in configuration uses a local dataset path; update it before running
on another machine. Training writes timestamped runs beneath `train.output_dir`.
Use the resulting run directory, containing `models/model_epoch_8.pth`, below:

```bash
DATASET=/path/to/demo_im128_head_wrist_cartesian_cmd.hdf5
RUN=/path/to/absolute_training_run
python robomimic/scripts/evaluate_rby1_cartesian_checkpoints.py \
  --run-dir "$RUN" --epochs 8 --dataset "$DATASET" \
  --demo-keys demo_10 demo_19 demo_47 demo_73 demo_111 \
  --seeds 0 1 42 123 12345 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 \
  --max-steps 2000 --device cuda --no-save-videos \
  --output-dir experiment_results/absolute_review_eval
```

Choose a new evaluation output directory outside the training run and dataset
directory. It contains `rollouts.csv` and `metadata.json`. Evaluation restores
each demonstration's saved XML and state and uses the existing RB-Y1 WholeBody
IK; the adapter converts base-frame TCP commands to world-frame wrist targets.

## Evaluation

The epoch-8 absolute policy achieved **63/125 successes (50.4%)**, using five
validation demonstrations and 25 diffusion seeds per demonstration, with a
2,000-step maximum horizon. Success requires the final door hinge to meet
`>= -0.07853981634` rad; reaching the threshold and reopening is failure.
These results cover saved validation scenes and base poses; they do not establish
generalization to unseen layouts or robot-base positions. Seeds alone do not
guarantee identical results across devices or dependency versions.
