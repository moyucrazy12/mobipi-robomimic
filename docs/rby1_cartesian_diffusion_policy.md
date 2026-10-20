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
