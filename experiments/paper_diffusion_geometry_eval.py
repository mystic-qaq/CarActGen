from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(REPO_ROOT.as_posix())

from experiments.fixed_car_template import PART_ORDER, load_source_parts


def env_path(name: str, default: Path | str | None = None) -> Path | None:
    value = os.environ.get(name)
    if value:
        return Path(value)
    return Path(default) if default is not None else None


def load_rows(paths: list[Path], family: str) -> list[dict]:
    rows = []
    for path in paths:
        with path.open() as f:
            for row in csv.DictReader(f):
                row["family"] = family
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape[0] == 0:
        return vertices
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    center = (lo + hi) * 0.5
    scale = float(np.max(hi - lo))
    if scale < 1e-8:
        scale = 1.0
    return (vertices - center) / scale


def normalized_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    out = mesh.copy()
    out.vertices = normalize_vertices(out.vertices)
    return out


def surface_points(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    if len(mesh.faces) == 0:
        return np.empty((0, 3), dtype=np.float32)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    return pts.astype(np.float32)


def geometry_metrics(source: trimesh.Trimesh, generated: trimesh.Trimesh, samples: int, seed: int) -> dict:
    src = normalized_mesh(source)
    gen = normalized_mesh(generated)
    src_pts = surface_points(src, samples, seed)
    gen_pts = surface_points(gen, samples, seed + 17)
    if len(src_pts) == 0 or len(gen_pts) == 0:
        return {
            "chamfer_l1": math.nan,
            "chamfer_l2": math.nan,
            "s2g_mean": math.nan,
            "g2s_mean": math.nan,
            "s2g_p95": math.nan,
            "g2s_p95": math.nan,
            "fscore_1pct": math.nan,
            "fscore_2pct": math.nan,
        }

    src_tree = cKDTree(src_pts)
    gen_tree = cKDTree(gen_pts)
    s2g, _ = gen_tree.query(src_pts, k=1)
    g2s, _ = src_tree.query(gen_pts, k=1)

    def fscore(threshold: float) -> float:
        precision = float((g2s < threshold).mean())
        recall = float((s2g < threshold).mean())
        denom = precision + recall
        return 0.0 if denom <= 1e-12 else 2.0 * precision * recall / denom

    return {
        "chamfer_l1": float(np.mean(s2g) + np.mean(g2s)),
        "chamfer_l2": float(np.mean(s2g**2) + np.mean(g2s**2)),
        "s2g_mean": float(np.mean(s2g)),
        "g2s_mean": float(np.mean(g2s)),
        "s2g_p95": float(np.percentile(s2g, 95)),
        "g2s_p95": float(np.percentile(g2s, 95)),
        "fscore_1pct": fscore(0.01),
        "fscore_2pct": fscore(0.02),
    }


def part_mesh_paths(sample_dir: Path) -> list[Path]:
    mesh_dir = sample_dir / "generated_part_mesh"
    paths = sorted(mesh_dir.glob("*.ply"))
    if len(paths) < len(PART_ORDER):
        return paths
    return paths[: len(PART_ORDER)]


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        part_group = "body" if row["part_name"] == "body_shell" else "wheel"
        groups.setdefault((row["family"], row["entry"], "all"), []).append(row)
        groups.setdefault((row["family"], row["entry"], part_group), []).append(row)
    out = []
    for (family, entry, group), items in sorted(groups.items()):
        out.append(
            {
                "family": family,
                "entry": entry,
                "group": group,
                "part_count": len(items),
                "sample_count": len({row["shape_id"] for row in items}),
                "chamfer_l1": np.nanmean([float(row["chamfer_l1"]) for row in items]),
                "chamfer_l2": np.nanmean([float(row["chamfer_l2"]) for row in items]),
                "fscore_1pct": np.nanmean([float(row["fscore_1pct"]) for row in items]),
                "fscore_2pct": np.nanmean([float(row["fscore_2pct"]) for row in items]),
                "s2g_p95": np.nanmean([float(row["s2g_p95"]) for row in items]),
                "g2s_p95": np.nanmean([float(row["g2s_p95"]) for row in items]),
            }
        )
    return out


def write_markdown(path: Path, rows: list[dict]):
    lines = [
        "# Diffusion generated-vs-source geometry",
        "",
        "| family | entry | group | samples | parts | Chamfer L1 | F@1% | F@2% | s2g p95 | g2s p95 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family']} | {row['entry']} | {row['group']} | {row['sample_count']} | {row['part_count']} | "
            f"{row['chamfer_l1']:.4f} | {row['fscore_1pct']:.3f} | {row['fscore_2pct']:.3f} | "
            f"{row['s2g_p95']:.4f} | {row['g2s_p95']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated part geometry against held-out source meshes.")
    parser.add_argument(
        "--original_sample_csv",
        type=Path,
        default=env_path("CARACTGEN_ORIGINAL_SAMPLE_CSV"),
    )
    parser.add_argument(
        "--adaptive_sample_csv",
        type=Path,
        default=env_path("CARACTGEN_ADAPTIVE_SAMPLE_CSV"),
    )
    parser.add_argument("--info_root", type=Path, default=env_path("CARACTGEN_INFO_ROOT", env_path("CARACTGEN_DATA_ROOT", REPO_ROOT / "data/datasets") / "1_preprocessed_info"))
    parser.add_argument("--mesh_root", type=Path, default=env_path("CARACTGEN_MESH_ROOT", env_path("CARACTGEN_DATA_ROOT", REPO_ROOT / "data/datasets") / "1_preprocessed_mesh"))
    parser.add_argument("--surface_samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=env_path("CARACTGEN_OUTPUT_ROOT", REPO_ROOT / "outputs") / "caractgen_diffusion_geometry",
    )
    args = parser.parse_args()
    if args.original_sample_csv is None or args.adaptive_sample_csv is None:
        raise ValueError(
            "Set --original_sample_csv and --adaptive_sample_csv, or "
            "CARACTGEN_ORIGINAL_SAMPLE_CSV and CARACTGEN_ADAPTIVE_SAMPLE_CSV."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    sample_rows.extend(load_rows([args.original_sample_csv], "original_artformer"))
    sample_rows.extend(load_rows([args.adaptive_sample_csv], "function_aware_diffusion"))

    rows = []
    source_cache = {}
    for sample_idx, sample in enumerate(sample_rows):
        sample_dir = Path(sample.get("sample_dir", ""))
        shape_id = sample.get("shape_id", "")
        if not sample_dir.exists() or not shape_id:
            continue
        if shape_id not in source_cache:
            _source, source_parts = load_source_parts(shape_id, args.info_root, args.mesh_root)
            source_cache[shape_id] = source_parts
        source_parts = source_cache[shape_id]
        paths = part_mesh_paths(sample_dir)
        for idx, part_name in enumerate(PART_ORDER):
            if idx >= len(paths):
                continue
            try:
                generated = trimesh.load_mesh(paths[idx].as_posix(), force="mesh", process=False)
                metrics = geometry_metrics(
                    source_parts[part_name]["mesh"],
                    generated,
                    args.surface_samples,
                    args.seed + sample_idx * 13 + idx,
                )
                row = {
                    "family": sample["family"],
                    "entry": sample["entry"],
                    "shape_id": shape_id,
                    "part_index": idx + 1,
                    "part_name": part_name,
                    "part_file": paths[idx].name,
                    "sample_dir": sample_dir.as_posix(),
                    **metrics,
                }
            except Exception as exc:
                (args.output_dir / "errors.log").open("a").write(f"{sample['entry']},{shape_id},{part_name}: {exc}\n")
                row = {
                    "family": sample["family"],
                    "entry": sample["entry"],
                    "shape_id": shape_id,
                    "part_index": idx + 1,
                    "part_name": part_name,
                    "part_file": paths[idx].name if idx < len(paths) else "",
                    "sample_dir": sample_dir.as_posix(),
                    "chamfer_l1": math.nan,
                    "chamfer_l2": math.nan,
                    "s2g_mean": math.nan,
                    "g2s_mean": math.nan,
                    "s2g_p95": math.nan,
                    "g2s_p95": math.nan,
                    "fscore_1pct": math.nan,
                    "fscore_2pct": math.nan,
                }
            rows.append(row)

    write_csv(args.output_dir / "diffusion_geometry_part_metrics.csv", rows)
    summary_rows = summarize(rows)
    write_csv(args.output_dir / "diffusion_geometry_summary.csv", summary_rows)
    write_markdown(args.output_dir / "diffusion_geometry_summary.md", summary_rows)
    print(args.output_dir.as_posix())


if __name__ == "__main__":
    main()
