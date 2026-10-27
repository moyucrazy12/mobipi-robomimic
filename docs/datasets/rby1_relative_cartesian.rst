Current-TCP-relative RB-Y1 Cartesian Diffusion Policy
===================================================

This opt-in branch changes only the coordinate representation of the existing
command targets. The absolute JSON, HDF5, Cartesian adapter, controller, network
architecture, observations, optimization settings and 2/16/8 horizons remain the
baseline. Preparation and validation do not launch full training.

Validation status
-----------------

The epoch-15 relative policy achieved **92/125 successes (73.6%)**, compared
with **63/125 (50.4%)** for the epoch-8 absolute policy: an improvement of
**23.2 percentage points**. Both evaluations used the same five validation
demonstrations and 25 diffusion seeds per demonstration, with a 2,000-step
maximum horizon. These results cover saved validation scenes and robot-base
poses; generalization to unseen layouts or base positions is not established.

Definition
----------

All poses are SE(3) transforms. At each replan, measure the current actual grip
TCP in robot-base coordinates, A = T_base_tcp_actual. For every future command
C[k] from the validated source ``actions_eef_cmd``:

::

    relative[k] = inv(A) @ C[k]
    reconstructed_command[k] = A @ relative[k]

Encode each relative transform as ``[dx,dy,dz,R[:,0],R[:,1]]``. Translation is in
the anchor TCP axes, not base/world axes. Rotations are composed as matrices;
there is no Euler subtraction. One anchor is used for the entire prediction.
Relative[k] is never composed with relative[k-1].

Exact training indexing
-----------------------

The existing SequenceDataset uses frame_stack=2, seq_length=15 and both padding
flags. For sample index t it returns 16 entries:

::

    observation/action indices = clip([t-1,t,...,t+14], 0, T-1)
    conditioning observations = slots 0:2 = [max(0,t-1),t]
    anchor = measured TCP from observation slot 1 = timestep t
    relative labels[k] = inv(TCP_actual[t]) @ command[clipped_index[k]]
    native executable prediction slots = 1:9 = source targets t..t+7

The previous command at slot 0 is trained in the same anchor as all future
commands. Initial and terminal padding retain native repeated samples. At t=0
both observation frames are observation 0; at the last sample future commands
repeat the last recorded command. No achieved-next-state targets are introduced.

Dynamic labels and normalization
--------------------------------

``RelativeCartesianSequenceDataset`` subclasses the native dataset and converts
raw command sequences before the existing normalization stage. Source actions
and images are read-only. It introduces no new HDF5 and no new image cache.
The key ``actions_eef_cmd`` remains the on-disk source/shape key; its labels in
the relative dataset are transformed in memory. The checkpoint config explicitly
identifies their relative semantics using ``algo.relative_actions.enabled``.

The min/max normalizer is recomputed from ALL 16 slots of EVERY padded training
window under each window's own measured anchor. It is not computed from absolute
commands or independent per-timestep deltas. Only the 100 training demos define
these statistics. Existing train.py shares those statistics with validation and
saves them in the relative experiment's checkpoint. Validation does not define
its own scale. No absolute normalization statistics are imported.

Inference and compatibility
---------------------------

``RelativeDiffusionPolicyUNet`` inherits the unchanged network, EMA selection,
DDIM and native eight-action queue. When that queue is empty, it captures the
last of the two current TCP observations as its new anchor. It holds this anchor
through the next eight executed commands, even when incoming observations change.
The initial pose enters as the native float32 lowdim observation, so its numerical
precision is the same as the rest of the policy input.

The raw model's get_action returns normalized relative commands. Native
RolloutPolicy unnormalizes once, then invokes an optional decode hook implemented
only by the relative subclass. External RolloutPolicy output is therefore an
absolute base-frame9 command, ready for the existing Cartesian adapter. This
ensures native checkpoint loading also restores the anchor-aware behavior without
a special manual action queue. start_episode/reset clears both queue and anchor.
Absolute models do not implement this hook and retain their original behavior.

The factory selects the relative subclass only when enabled. Dataset factory
selection is likewise opt-in. The default config leaves relative mode disabled.
The first implementation explicitly supports the controlled frame2/seq15,
prediction16/action8, raw TCP, min-max, non-language Robomimic baseline.

Arms and frames
---------------

The mathematical helpers and relative dataset/policy accept ``arm=left`` or
``arm=right``. Observation names are resolved by ``tcp_observation_keys``:

- robot0_base_to_{arm}_eef_pos
- robot0_base_to_{arm}_eef_quat_site (xyzw, actual grip site)

The dataset also retains world-frame robot0_{arm}_eef_pos and
robot0_{arm}_eef_quat_site. The legacy quat key is not substituted for quat_site.
The TCP/reference names are gripper0_{arm}_grip_site and gripper0_{arm}_ft_frame.

In the saved demo XML, BOTH FT-to-TCP transforms have translation
[0,0,-0.2501] m and rotation Rz(-90 degrees). Neither side introduces a different
local TCP axis convention relative to its FT frame. The replay audit checks the
compiled transforms and reports hand-to-TCP rotations for both arms. This is not
a claim that the two arms have identical world orientations or mirrored joint
kinematics. No canonical-frame conversion or image mirroring is added.

Only left-arm data and the existing left-arm Cartesian adapter/controller path
are used here. Right-arm mathematical and observation-key support is tested;
physical right-arm policy control still requires suitable right-arm commands,
observations and an IK joint/reference configuration. No right-arm behavior is
claimed without that future validation.

Validation and commands
-----------------------

Run from the robomimic repository with the existing Mobi-pi Python environment::

    python -m unittest discover -s tests -p test_rby1_relative_cartesian.py -v

    python robomimic/scripts/validate_rby1_relative_cartesian.py \
      --phase dataset \
      --config robomimic/exps/rby1_close_single_door_relative_cartesian_cmd_115d_15pass.json \
      --output-dir experiment_results/rby1_relative_preparation

    python robomimic/scripts/validate_rby1_relative_cartesian.py \
      --phase replay \
      --config robomimic/exps/rby1_close_single_door_relative_cartesian_cmd_115d_15pass.json \
      --output-dir experiment_results/rby1_relative_preparation --demo demo_10

The read-only dataset audit checks all demos/windows, normalization provenance,
real sample indexing and unchanged 9120/1583 full-pass batch counts. Saved-model
replay measures a new actual TCP anchor every eight commands, encodes all16,
decodes independently and compares reconstructed commands against the original
commands before stepping the unchanged controller. Meaningful reconstruction
error stops replay. Task failure prevents proceeding to training.

After both gates and unit tests pass, run ONLY the standard short debug::

    python robomimic/scripts/train.py \
      --config robomimic/exps/rby1_close_single_door_relative_cartesian_cmd_115d_15pass.json \
      --debug

Then verify native checkpoint reload using ``--phase reload --checkpoint PATH``
with the same config/output directory. This checks the relative normalizer,
EMA inference, one eight-action relative chunk, fixed anchor and finite decoded
absolute targets. Debug weights are not a trained-policy result.

When explicitly ready for the full experiment, use the same train.py command
without --debug. The config plans 15 full passes, batch16/workers4/LR1e-4, and
keeps the original dataset, train/valid split, images, TCP observations and model.
No long training, push, pull request or parent-submodule update is performed by
these preparation commands.

Evaluate a trained checkpoint
-----------------------------

Run from the repository root in the existing ``mobipi`` Conda environment, with
``PYTHONPATH`` including this repository and ``MUJOCO_GL=egl``. Set the dataset
path in a local copy of the relative config before training on another machine.
Use the normal training entry point::

    python robomimic/scripts/train.py \
      --config robomimic/exps/rby1_close_single_door_relative_cartesian_cmd_115d_15pass.json

Set RUN to the timestamped training directory containing
``models/model_epoch_15.pth`` and DATASET to the prepared command HDF5::

    DATASET=/path/to/demo_im128_head_wrist_cartesian_cmd.hdf5
    RUN=/path/to/relative_training_run
    python robomimic/scripts/evaluate_rby1_cartesian_checkpoints.py \
      --run-dir "$RUN" --epochs 15 --dataset "$DATASET" \
      --demo-keys demo_10 demo_19 demo_47 demo_73 demo_111 \
      --seeds 0 1 42 123 12345 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 \
      --max-steps 2000 --device cuda --no-save-videos \
      --output-dir experiment_results/relative_review_eval

Choose a new output directory outside the training run and dataset directory.
Results are saved as ``rollouts.csv`` and ``metadata.json``. Success requires
the final door hinge to meet ``>= -0.07853981634`` rad; reopening is failure.
Dataset files, checkpoints and raw experiment outputs are not included in Git.
Seeds alone do not guarantee identical results across devices or dependencies.
