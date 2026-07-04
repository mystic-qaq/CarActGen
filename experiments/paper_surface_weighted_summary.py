from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh


PART_ORDER = [
    "body_shell",
    "wheel_front_left",
    "wheel_front_right",
    "wheel_rear_left",
    "wheel_rear_right",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def source_part_areas(shape_id: str, info_root: Path, mesh_root: Path) -> dict[str, float]:
    info = json.loads((info_root / f"{shape_id}.json").read_text())
    by_name = {part["name"]: part for part in info["part"]}
    areas = {}
    for name in PART_ORDER:
        part = by_name[name]
        mesh = trimesh.load_mesh((mesh_root / part["mesh"]).as_posix(), force="mesh", process=False)
        area = float(mesh.area)
        if not np.isfinite(area) or area <= 0:
            mins = np.asarray(part["bbx"][0], dtype=float)
            maxs = np.asarray(part["bbx"][1], dtype=float)
            size = np.maximum(maxs - mins, 1e-8)
            area = float(2.0 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2]))
        areas[name] = area
    return areas


def group_key(row: dict, group_cols: list[str]) -> tuple[str, ...]:
    return tuple(row.get(col, "") for col in group_cols)


def weighted_mean(items: list[dict], weights: np.ndarray, metric: str) -> float:
    values = np.asarray([float(row[metric]) for row in items], dtype=float)
    mask = np.isfinite(values)
    if not mask.any():
        return float("nan")
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-part CarActGen metrics with source mesh surface-area weights."
    )
    parser.add_argument("--part_metrics_csv", type=Path, required=True)
    parser.add_argument("--info_root", type=Path, required=True)
    parser.add_argument("--mesh_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--group_cols",
        nargs="+",
        default=["family", "entry"],
        help="Columns that identify a method/configuration before shape aggregation.",
    )
    parser.add_argument(
        "--metric_names",
        nargs="+",
        required=True,
        help="Numeric per-part metrics to aggregate.",
    )
    parser.add_argument("--group_label", default="surface-weighted all")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.part_metrics_csv)
    area_cache = {
        shape_id: source_part_areas(shape_id, args.info_root, args.mesh_root)
        for shape_id in sorted({row["shape_id"] for row in rows})
    }

    shape_groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        shape_groups[group_key(row, args.group_cols) + (row["shape_id"],)].append(row)

    weighted_shape_rows = []
    for key, items in sorted(shape_groups.items()):
        shape_id = key[-1]
        areas = area_cache[shape_id]
        weights = np.asarray([areas[row["part_name"]] for row in items], dtype=float)
        weights = weights / weights.sum()
        out = {
            col: value for col, value in zip(args.group_cols, key[:-1])
        }
        out.update(
            {
                "shape_id": shape_id,
                "group": args.group_label,
                "part_count": len(items),
                "body_area_fraction": areas["body_shell"] / sum(areas.values()),
            }
        )
        for metric in args.metric_names:
            out[metric] = weighted_mean(items, weights, metric)
        weighted_shape_rows.append(out)

    summary_groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in weighted_shape_rows:
        summary_groups[group_key(row, args.group_cols)].append(row)

    summary_rows = []
    for key, items in sorted(summary_groups.items()):
        out = {col: value for col, value in zip(args.group_cols, key)}
        out.update(
            {
                "group": args.group_label,
                "shape_count": len(items),
                "part_count": sum(int(row["part_count"]) for row in items),
                "body_area_fraction": float(np.mean([float(row["body_area_fraction"]) for row in items])),
            }
        )
        for metric in args.metric_names:
            out[metric] = float(np.nanmean([float(row[metric]) for row in items]))
        summary_rows.append(out)

    stem = args.part_metrics_csv.stem
    write_csv(args.output_dir / f"{stem}_surface_weighted_shape_metrics.csv", weighted_shape_rows)
    write_csv(args.output_dir / f"{stem}_surface_weighted_summary.csv", summary_rows)

    print(args.output_dir / f"{stem}_surface_weighted_summary.csv")


if __name__ == "__main__":
    main()
