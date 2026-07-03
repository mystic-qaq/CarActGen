from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def env_path(name: str, default: Path | str) -> Path:
    return Path(os.environ.get(name, default))


def load_rows(paths):
    rows = []
    for path in paths:
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    return rows


def to_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def to_bool(value):
    return str(value).lower() in {"true", "1", "yes"}


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(sample_rows):
    grouped = {}
    for row in sample_rows:
        grouped.setdefault(row["entry"], []).append(row)
    out = []
    for entry, items in sorted(grouped.items()):
        out.append(
            {
                "entry": entry,
                "count": len(items),
                "sample_success_rate": np.mean([float(to_bool(r["sample_success"])) for r in items]),
                "all_watertight_rate": np.mean([float(to_bool(r["all_watertight"])) for r in items]),
                "wheel_watertight_rate": np.mean([to_float(r["wheel_watertight_count"]) / 4.0 for r in items]),
                "wheel_single_component_rate": np.mean([to_float(r["wheel_single_component_count"]) / 4.0 for r in items]),
                "max_components_mean": np.mean([to_float(r["max_components"]) for r in items]),
                "latent_std_mean": np.nanmean([to_float(r["latent_std"]) for r in items]),
                "latent_abs_gt1_mean": np.nanmean([to_float(r["latent_abs_gt1"]) for r in items]),
                "latent_abs_gt2_mean": np.nanmean([to_float(r["latent_abs_gt2"]) for r in items]),
            }
        )
    return out


def write_markdown(path: Path, summary_rows):
    lines = [
        "# Diffusion comparison summary",
        "",
        "| entry | n | success | all watertight | wheel watertight | wheel 1-comp | max comp | latent std | abs(z)>1 | abs(z)>2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['entry']} | {row['count']} | {row['sample_success_rate']:.3f} | "
            f"{row['all_watertight_rate']:.3f} | {row['wheel_watertight_rate']:.3f} | "
            f"{row['wheel_single_component_rate']:.3f} | {row['max_components_mean']:.2f} | "
            f"{row['latent_std_mean']:.3f} | {row['latent_abs_gt1_mean']:.4f} | {row['latent_abs_gt2_mean']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Combine diffusion sample-evaluation CSV shards.")
    parser.add_argument("--output_dir", type=Path, default=env_path("CARACTGEN_OUTPUT_ROOT", REPO_ROOT / "outputs") / "caractgen_diffusion_samples")
    args = parser.parse_args()

    sample_paths = sorted(args.output_dir.glob("*_sample_metrics_*.csv"))
    part_paths = sorted(args.output_dir.glob("*_part_metrics_*.csv"))
    sample_rows = load_rows(sample_paths)
    part_rows = load_rows(part_paths)
    write_csv(args.output_dir / "combined_sample_metrics.csv", sample_rows)
    write_csv(args.output_dir / "combined_part_metrics.csv", part_rows)
    summary_rows = summarize(sample_rows)
    write_csv(args.output_dir / "combined_diffusion_summary.csv", summary_rows)
    write_markdown(args.output_dir / "combined_diffusion_summary.md", summary_rows)
    print(args.output_dir.as_posix())


if __name__ == "__main__":
    main()
