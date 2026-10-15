"""Summarize completed saved-model RB-Y1 rollouts without Torch or MuJoCo."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from robomimic.utils.rby1_evaluation import (
    as_bool, extract_losses, read_completed_rows, row_key, write_csv,
)


def numeric(rows, key):
    result = []
    for row in rows:
        try:
            value = float(row[key])
            if math.isfinite(value):
                result.append(value)
        except (KeyError, ValueError, TypeError):
            pass
    return result


def mean(values):
    return statistics.mean(values) if values else None


def median(values):
    return statistics.median(values) if values else None


def summarize(rows):
    by_epoch = defaultdict(list)
    for row in rows:
        by_epoch[int(row["checkpoint_epoch"])].append(row)
    summary, demos, seeds, failures = [], [], [], []
    for epoch, group in sorted(by_epoch.items()):
        successful = [row for row in group if as_bool(row["success"])]
        categories = Counter(row["failure_classification"] for row in group)
        success_steps = numeric(successful, "success_timestep")
        row = dict(
            epoch=epoch, rollouts=len(group), successes=len(successful),
            success_rate=len(successful) / len(group), median_success_step=median(success_steps),
            mean_success_step=mean(success_steps),
            mean_final_hinge=mean(numeric(group, "final_hinge_angle")),
            median_final_hinge=median(numeric(group, "final_hinge_angle")),
            mean_best_hinge=mean(numeric(group, "best_hinge_angle")),
            median_best_hinge=median(numeric(group, "best_hinge_angle")),
            mean_tcp_tracking_error_mm=mean(numeric(group, "mean_tcp_tracking_error_mm")),
            rollouts_with_tcp_tracking=len(numeric(group, "mean_tcp_tracking_error_mm")),
            median_tcp_tracking_error_mm=median(numeric(group, "mean_tcp_tracking_error_mm")),
            max_tcp_tracking_error_mm=max(numeric(group, "max_tcp_tracking_error_mm"), default=None),
            mean_tcp_orientation_error_deg=mean(numeric(group, "mean_tcp_orientation_error_deg")),
            failure_counts=json.dumps({key: value for key, value in categories.items() if key != "success"}, sort_keys=True),
        )
        summary.append(row)
        for key, destination in (("demo_key", demos), ("seed", seeds)):
            values = sorted({str(item[key]) for item in group})
            for value in values:
                selected = [item for item in group if str(item[key]) == value]
                count = sum(as_bool(item["success"]) for item in selected)
                destination.append(dict(epoch=epoch, **{key: value}, rollouts=len(selected),
                                        successes=count, success_rate=count / len(selected)))
        failures.extend(dict(epoch=epoch, category=key, count=value) for key, value in sorted(categories.items()) if key != "success")
    # Direct comparisons use only identical demo/split/seed trials present in both epochs.
    paired = []
    epochs = sorted(by_epoch)
    for index, first in enumerate(epochs):
        left = {(r["split"], r["demo_key"], int(r["seed"])): r for r in by_epoch[first]}
        for second in epochs[index + 1:]:
            right = {(r["split"], r["demo_key"], int(r["seed"])): r for r in by_epoch[second]}
            common = sorted(left.keys() & right.keys())
            improvements = sum(not as_bool(left[k]["success"]) and as_bool(right[k]["success"]) for k in common)
            regressions = sum(as_bool(left[k]["success"]) and not as_bool(right[k]["success"]) for k in common)
            left_rate = mean([int(as_bool(left[k]["success"])) for k in common])
            right_rate = mean([int(as_bool(right[k]["success"])) for k in common])
            paired.append(dict(epoch_a=first, epoch_b=second, matched_rollouts=len(common),
                               success_rate_a=left_rate, success_rate_b=right_rate,
                               success_rate_delta_b_minus_a=right_rate - left_rate if common else None,
                               improvements=improvements, regressions=regressions))
    return summary, demos, seeds, failures, paired


def display(value):
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "\\|")


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(display(value) for value in row) + " |" for row in rows])


def matrix(rows, key, epochs):
    lookup = {(str(row[key]), row["epoch"]): row for row in rows}
    names = sorted({str(row[key]) for row in rows})
    values = []
    for name in names:
        cells = []
        for epoch in epochs:
            row = lookup.get((name, epoch))
            cells.append(f"{row['success_rate']:.1%} ({row['successes']}/{row['rollouts']})" if row else "—")
        values.append([name, *cells])
    return markdown_table([key, *[str(epoch) for epoch in epochs]], values)


def plots(output, summary, losses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    artifacts = []
    specs = [
        ("success_rate.png", "Success rate", [("success_rate", "Success rate")], summary),
        ("best_hinge.png", "Best hinge (rad)", [("mean_best_hinge", "Mean"), ("median_best_hinge", "Median")], summary),
        ("training_losses.png", "Diffusion loss", [("train_loss", "Train"), ("validation_loss", "Validation")], losses),
    ]
    for filename, ylabel, metrics, rows in specs:
        fig, ax = plt.subplots()
        available = False
        for key, label in metrics:
            points = [(row["epoch"], row.get(key)) for row in rows if row.get(key) is not None]
            if points:
                x, y = zip(*points)
                ax.plot(x, y, "o-", label=label)
                available = True
        if available:
            ax.set(xlabel="Epoch", ylabel=ylabel)
            if filename == "success_rate.png":
                ax.set_ylim(0, 1)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / filename, dpi=150)
            artifacts.append(filename)
        plt.close(fig)
    return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Evaluation directory or rollouts.csv")
    parser.add_argument("--output-dir", required=True, help="Separate report directory")
    parser.add_argument("--run-dir", help="Optional training run, otherwise use evaluation metadata")
    parser.add_argument("--plots", action="store_true", help="Optional CPU matplotlib plots")
    args = parser.parse_args(argv)
    source = Path(args.input).expanduser().resolve()
    csv_path = source / "rollouts.csv" if source.is_dir() else source
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows, invalid = read_completed_rows(csv_path)
    if not rows:
        raise ValueError("No valid completed rollouts to summarize")
    if len({row_key(row) for row in rows}) != len(rows):
        raise ValueError("Duplicate epoch/demo/seed trials")
    if len({(row["max_steps"], row["split"]) for row in rows}) != 1:
        raise ValueError("Mixed horizons/splits: use separate reports")
    for epoch in {row["checkpoint_epoch"] for row in rows}:
        if len({(row["checkpoint_path"], row["checkpoint_size"], row["checkpoint_mtime_ns"]) for row in rows if row["checkpoint_epoch"] == epoch}) != 1:
            raise ValueError("Mixed checkpoint identities at the same epoch")
    metadata_path = csv_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    output = Path(args.output_dir).expanduser().resolve()
    run_value = args.run_dir or metadata.get("protocol", {}).get("run_dir")
    run = Path(run_value).expanduser().resolve() if run_value else None
    if output == csv_path.parent or (run and (output == run or run in output.parents)):
        raise ValueError("Use a separate report directory outside the active training run")
    output.mkdir(parents=True, exist_ok=True)
    summary, demos, seeds, failures, paired = summarize(rows)
    exports = [
        ("checkpoint_summary.csv", summary, list(summary[0])),
        ("demo_summary.csv", demos, ["epoch", "demo_key", "rollouts", "successes", "success_rate"]),
        ("seed_summary.csv", seeds, ["epoch", "seed", "rollouts", "successes", "success_rate"]),
        ("failure_counts.csv", failures, ["epoch", "category", "count"]),
        ("paired_comparisons.csv", paired, ["epoch_a", "epoch_b", "matched_rollouts", "success_rate_a", "success_rate_b", "success_rate_delta_b_minus_a", "improvements", "regressions"]),
    ]
    for name, table, fields in exports:
        write_csv(output / name, table, fields)
    losses = extract_losses(run) if run else []
    write_csv(output / "training_losses.csv", losses, ["epoch", "train_loss", "validation_loss"])
    table = markdown_table(
        ["Epoch", "Successes", "Success Rate", "Median Success Step", "Mean Best Hinge", "Notes"],
        [[row["epoch"], f"{row['successes']}/{row['rollouts']}", f"{row['success_rate']:.1%}",
          row["median_success_step"], row["mean_best_hinge"], row["failure_counts"]] for row in summary])
    requested = metadata.get("planned_rollouts")
    missing = metadata.get("missing_epochs", [])
    lines = ["# RB-Y1 Cartesian Diffusion Policy checkpoint evaluation", "",
             f"Source: `{csv_path}`. Completed valid rows: {len(rows)}; ignored incomplete/invalid rows: {invalid}.",
             f"Requested rollout count from the latest plan: {requested if requested is not None else 'unknown'}; checkpoints pending at evaluator launch: {missing}.",
             "This report contains only completed trials; it does not assert that training or the evaluation sweep has finished.", "",
             "Success requires the final microwave hinge to be >= -0.07853981634 rad at the full horizon. First threshold reach is logged separately; reopening is failure. Runtime errors count as failed trials.",
             "Steps are 1-based after each controller step (0 means closed initially). Hinge angles are radians; larger values mean more closed.",
             "Raw success rates may cover different trial sets when evaluation is incomplete. Paired comparisons below use only shared demo/split/seed trials.", "", table, "",
             "## Success rate by demonstration", "", matrix(demos, "demo_key", [row["epoch"] for row in summary]), "",
             "## Success rate by diffusion seed", "", matrix(seeds, "seed", [row["epoch"] for row in summary]), "",
             "## Paired checkpoint comparisons", "",
             markdown_table(["Epoch A", "Epoch B", "Matched Trials", "Rate A", "Rate B", "Delta B−A", "Improvements", "Regressions"],
                            [[row[k] for k in ("epoch_a", "epoch_b", "matched_rollouts", "success_rate_a", "success_rate_b", "success_rate_delta_b_minus_a", "improvements", "regressions")] for row in paired]), "",
             "## Training losses", "",
             markdown_table(["Epoch", "Train loss", "Validation loss"], [[row["epoch"], row.get("train_loss"), row.get("validation_loss")] for row in losses]) if losses else "No completed loss blocks available. TensorBoard is not required.", "",
             "## Tracking and classification notes", "",
             "checkpoint_summary.csv contains mean/median final and best hinge, mean success step, failure counts, mean/median per-rollout mean TCP tracking error and global maximum tracking error. Tracking measures post-action TCP against the executed desired TCP; it is not demonstration path error.",
             "A tracking aggregate averages available per-rollout values, not pooled timesteps. Missing diagnostics from errors are omitted, not replaced by zero.",
             "no_meaningful_motion means TCP excursion <5 mm; partial_closure means best hinge improved >=0.1 rad without final success. No contact-based label is inferred without contact diagnostics.", ""]
    if args.plots:
        artifacts = plots(output, summary, losses)
        lines += ["## Plots", ""] + [f"![{name}]({name})" for name in artifacts]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(table)
    print(f"Reports saved to {output}; no simulation or checkpoint loading performed.")


if __name__ == "__main__":
    main()
