"""Resolve custom RB-Y1 paths, then call the unchanged native RoboMimic trainer."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from robomimic.utils.rby1_paths import resolve_training_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Explicit training parent; overrides the experiment root")
    parser.add_argument("--dataset", type=Path, help="Explicit input; preserves the dataset's other configuration fields")
    parser.add_argument("--name", help="Native experiment-name override")
    parser.add_argument("--eval_only", action="store_true", help="Pass through the native evaluation-only option")
    parser.add_argument("--print-config", action="store_true", help="Print resolved JSON and exit without importing training dependencies")
    args = parser.parse_args(argv)
    settings = resolve_training_paths(json.loads(args.config.expanduser().read_text()), args.output_dir, args.dataset)
    if args.name is not None:
        settings["experiment"]["name"] = args.name
    if args.print_config:
        print(json.dumps(settings, indent=2))
        return settings
    from robomimic.scripts import train as native_train
    # The native trainer still creates experiment_name/timestamp and saves config.json.
    # This temporary input is not a second configuration or checkpoint format.
    with tempfile.TemporaryDirectory(prefix="rby1_train_config_") as temporary:
        path = Path(temporary) / "config.json"
        path.write_text(json.dumps(settings))
        native_train.main(SimpleNamespace(config=str(path), algo=None, dataset=None,
                                         name=args.name, debug=False, eval_only=args.eval_only))


if __name__ == "__main__":
    main()
