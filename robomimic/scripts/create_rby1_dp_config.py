"""
Create an RB-Y1 Cartesian Diffusion Policy training config for a prepared dataset.

Copies a base config (by default the CloseDrawerLeft 50-demo one) and sets the
dataset, the experiment name and the epoch lengths: one epoch is one pass over
the training sequences at the config's batch size, as in the RB-Y1 microwave
configs. The dataset must come from prepare_rby1_dp_dataset.py (it needs
actions_eef_cmd and the train / valid masks).

Example (from external/robomimic):
    python robomimic/scripts/create_rby1_dp_config.py \\
        --dataset ../../generated_datasets/mimicgen/CloseDrawerLeft_RBY1_300mg/close_drawer_300mg_im128_cmd.hdf5 \\
        --name rby1_close_drawer_cartesian_cmd_300mg --epochs 20 --save-every 2
"""
import argparse
import json
from pathlib import Path

import h5py


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset prepared with prepare_rby1_dp_dataset.py")
    parser.add_argument("--name", required=True, help="Experiment name (also the output folder name)")
    parser.add_argument("--base-config", default="robomimic/exps/rby1_close_drawer_cartesian_cmd_50d.json")
    parser.add_argument("--epochs", type=int, default=20, help="Passes over the training data")
    parser.add_argument("--save-every", type=int, default=2, help="Save a checkpoint every N epochs")
    parser.add_argument("--output", help="Config file to write (default: robomimic/exps/<name>.json)")
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser()
    if not dataset.exists():
        parser.error(f"{dataset} does not exist")
    output = Path(args.output or f"robomimic/exps/{args.name}.json")
    if output.exists():
        parser.error(f"{output} already exists")

    cfg = json.loads(Path(args.base_config).read_text())
    with h5py.File(dataset, "r") as f:
        missing = [k for k in ("mask/train", "mask/valid") if k not in f]
        demos = sorted(f["data"])
        if "actions_eef_cmd" not in f["data"][demos[0]]:
            missing.append("data/<demo>/actions_eef_cmd")
        if missing:
            parser.error(f"{dataset} has no {', '.join(missing)}. Run prepare_rby1_dp_dataset.py on the "
                         "rendered dataset first and pass its --output here.")
        steps = {k: sum(len(f[f"data/{n.decode()}/actions"]) for n in f[f"mask/{k}"][:]) for k in ("train", "valid")}
        counts = {k: len(f[f"mask/{k}"]) for k in ("train", "valid")}

    batch = cfg["train"]["batch_size"]
    cfg["experiment"]["name"] = args.name
    cfg["experiment"]["epoch_every_n_steps"] = steps["train"] // batch
    cfg["experiment"]["validation_epoch_every_n_steps"] = steps["valid"] // batch
    cfg["experiment"]["save"]["every_n_epochs"] = args.save_every
    cfg["train"]["data"] = [{"path": str(dataset), "do_eval": False}]
    cfg["train"]["num_epochs"] = args.epochs
    output.write_text(json.dumps(cfg, indent=4))

    print(f"Wrote {output}")
    print(f"  dataset: {counts['train']} train / {counts['valid']} valid demos, "
          f"{steps['train']} / {steps['valid']} steps")
    print(f"  epoch: {cfg['experiment']['epoch_every_n_steps']} steps (batch {batch}), "
          f"{args.epochs} epochs = {args.epochs * cfg['experiment']['epoch_every_n_steps']} steps, "
          f"checkpoint every {args.save_every}")


if __name__ == "__main__":
    main()
