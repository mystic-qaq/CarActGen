from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(REPO_ROOT.as_posix())

from experiments.fixed_car_template import PART_ORDER, build_parts, load_source_parts, render_to_image, write_structure
from experiments.sample_adaptive_object_multimodal_diffusion import (
    build_condition_batch,
    decode_latents,
    resolve_model_class_from_checkpoint,
)
from model.FunctionAware import load_function_map


def env_path(name: str, default: Path | str | None = None) -> Path | None:
    value = os.environ.get(name)
    if value:
        return Path(value)
    return Path(default) if default is not None else None


PRESETS = {
    "partlocal": [
        ("partlocal_text_cfg1.2", "text", 1.2),
        ("partlocal_image_cfg1.2", "image", 1.2),
        ("partlocal_text_image_cfg1.0", "text_image", 1.0),
        ("partlocal_text_image_cfg1.2", "text_image", 1.2),
    ],
}


def read_shape_ids(split_path: Path, split: str, max_shapes: int, start: int, end: int | None) -> list[str]:
    payload = json.loads(split_path.read_text())
    shape_ids = payload[split][:max_shapes]
    return shape_ids[start:end]


def stable_seed(base_seed: int, *parts: str) -> int:
    payload = "|".join(parts).encode("utf8")
    digest = hashlib.sha1(payload).hexdigest()
    return int((base_seed + int(digest[:8], 16)) % (2**31 - 1))


def component_stats(mesh: trimesh.Trimesh) -> tuple[int, float]:
    try:
        comps = mesh.split(only_watertight=False)
    except Exception:
        comps = []
    if not comps:
        return 0, 0.0
    faces = sorted((len(c.faces) for c in comps), reverse=True)
    return len(faces), float(faces[0] / max(len(mesh.faces), 1))


def mesh_part_row(mesh_path: Path, sample_meta: dict) -> dict:
    mesh = trimesh.load(mesh_path.as_posix(), force="mesh", process=False)
    components, largest_ratio = component_stats(mesh)
    name = mesh_path.stem
    is_wheel = "wheel" in name
    return {
        **sample_meta,
        "part_file": mesh_path.name,
        "part_group": "wheel" if is_wheel else "body",
        "file_size_mb": mesh_path.stat().st_size / 1024 / 1024,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "components": components,
        "largest_component_face_ratio": largest_ratio,
        "single_component": components == 1,
        "extent_x": float(mesh.extents[0]) if len(mesh.vertices) else math.nan,
        "extent_y": float(mesh.extents[1]) if len(mesh.vertices) else math.nan,
        "extent_z": float(mesh.extents[2]) if len(mesh.vertices) else math.nan,
    }


def latent_stats(latents: torch.Tensor) -> dict:
    z = latents.detach().cpu().float()
    return {
        "latent_std": float(z.std()),
        "latent_mean": float(z.mean()),
        "latent_min": float(z.min()),
        "latent_max": float(z.max()),
        "latent_abs_gt1": float((z.abs() > 1.0).float().mean()),
        "latent_abs_gt2": float((z.abs() > 2.0).float().mean()),
    }


def write_summary(sample_rows: list[dict], output_dir: Path):
    groups: dict[str, list[dict]] = {}
    for row in sample_rows:
        groups.setdefault(row["entry"], []).append(row)
    summary_rows = []
    for entry, items in sorted(groups.items()):
        summary_rows.append(
            {
                "entry": entry,
                "count": len(items),
                "sample_success_rate": np.mean([float(r["sample_success"]) for r in items]),
                "all_watertight_rate": np.mean([float(r["all_watertight"]) for r in items]),
                "wheel_watertight_rate": np.mean([float(r["wheel_watertight_count"]) / 4.0 for r in items]),
                "wheel_single_component_rate": np.mean([float(r["wheel_single_component_count"]) / 4.0 for r in items]),
                "max_components_mean": np.mean([float(r["max_components"]) for r in items]),
                "latent_std_mean": np.mean([float(r["latent_std"]) for r in items]),
                "latent_abs_gt2_mean": np.mean([float(r["latent_abs_gt2"]) for r in items]),
            }
        )

    if not summary_rows:
        return
    summary_csv = output_dir / "diffusion_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# Diffusion comparison summary",
        "",
        "| entry | n | success | all watertight | wheel watertight | wheel 1-comp | max comp | latent std | abs(z)>2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['entry']} | {row['count']} | {row['sample_success_rate']:.3f} | "
            f"{row['all_watertight_rate']:.3f} | {row['wheel_watertight_rate']:.3f} | "
            f"{row['wheel_single_component_rate']:.3f} | {row['max_components_mean']:.2f} | "
            f"{row['latent_std_mean']:.3f} | {row['latent_abs_gt2_mean']:.4f} |"
        )
    (output_dir / "diffusion_summary.md").write_text("\n".join(lines) + "\n")


def existing_or_generate_sample(
    model,
    checkpoint: Path,
    args,
    entry: str,
    condition_mode: str,
    guidance_scale: float,
    shape_id: str,
    function_map,
) -> tuple[Path, torch.Tensor]:
    output_dir = args.output_dir / "samples" / entry / shape_id
    latent_path = output_dir / "generated_latents.pt"
    if latent_path.exists() and (output_dir / "generated_part_mesh").exists() and not args.overwrite:
        return output_dir, torch.load(latent_path, map_location="cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    condition_args = SimpleNamespace(
        dataset_path=args.dataset_path,
        image_embedding_dir=args.image_embedding_dir,
        shape_id=shape_id,
    )
    batch = build_condition_batch(condition_args, model, function_map)
    use_text = condition_mode in {"text", "text_image"}
    use_image = condition_mode in {"image", "text_image"}

    seed = stable_seed(args.seed, checkpoint.name, entry, shape_id)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        latents = model.sample_latents(
            batch,
            guidance_scale=guidance_scale,
            use_text=use_text,
            use_image=use_image,
            clip_denoised=args.clip_denoised,
        )
    latents = latents[0, : len(PART_ORDER)]
    function_ids = batch["function_id"][0, : len(PART_ORDER)]
    torch.save(latents.detach().cpu(), latent_path)

    _source, source_parts = load_source_parts(shape_id, args.info_root, args.mesh_root)
    meshes = decode_latents(model, latents, function_ids, output_dir, args)
    generated_source_parts = {
        name: {
            "source": source_parts[name]["source"],
            "mesh": meshes[name],
            "mesh_path": output_dir / "generated_part_mesh" / f"{idx + 1:02d}_{name}.ply",
        }
        for idx, name in enumerate(PART_ORDER)
    }
    parts = build_parts(generated_source_parts, "source-bbox")
    write_structure(parts, output_dir / "structure.json", {"category": "car", "shape_id": shape_id}, "source-bbox")
    with open(output_dir / "processed_nodes.pkl", "wb") as f:
        pickle.dump(parts, f)
    render_to_image(parts, 0.0).save(output_dir / "pose_000.png")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint.as_posix(),
                "entry": entry,
                "condition_mode": condition_mode,
                "guidance_scale": guidance_scale,
                "clip_denoised": args.clip_denoised,
                "shape_id": shape_id,
                "seed": seed,
            },
            indent=2,
        )
    )
    return output_dir, latents.detach().cpu()


def main():
    parser = argparse.ArgumentParser(description="Evaluate object-level multimodal car diffusion samples on a clean held-out split.")
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_path", type=Path, default=env_path("CARACTGEN_LATENT_DATASET", env_path("CARACTGEN_OUTPUT_ROOT", REPO_ROOT / "outputs") / "datasets/2.1_adaptive_multimodal_latentcode_sketch_dinov2"))
    parser.add_argument("--split_path", type=Path, default=env_path("CARACTGEN_SPLIT_PATH", REPO_ROOT / "data/caractgen_metadata/splits/object_sketch_dinov2_partlocal_seed123456798.json"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_shapes", type=int, default=24)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, default=env_path("CARACTGEN_OUTPUT_ROOT", REPO_ROOT / "outputs") / "caractgen_diffusion_samples")
    parser.add_argument("--image_embedding_dir", type=Path, default=None)
    parser.add_argument("--info_root", type=Path, default=env_path("CARACTGEN_INFO_ROOT", env_path("CARACTGEN_DATA_ROOT", REPO_ROOT / "data/datasets") / "1_preprocessed_info"))
    parser.add_argument("--mesh_root", type=Path, default=env_path("CARACTGEN_MESH_ROOT", env_path("CARACTGEN_DATA_ROOT", REPO_ROOT / "data/datasets") / "1_preprocessed_mesh"))
    parser.add_argument("--sdf_resolution", type=int, default=128)
    parser.add_argument("--max_batch", type=int, default=32768)
    parser.add_argument("--clip_denoised", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shape_ids = read_shape_ids(args.split_path, args.split, args.max_shapes, args.start, args.end)
    shape_file = args.output_dir / f"{args.preset}_shapes_{args.start}_{'end' if args.end is None else args.end}.txt"
    shape_file.write_text("\n".join(shape_ids) + "\n")

    model_class = resolve_model_class_from_checkpoint(args.checkpoint)
    model = model_class.load_from_checkpoint(args.checkpoint, map_location="cuda").cuda().eval()
    model.requires_grad_(False)
    if model.sdf is not None:
        model.sdf = model.sdf.cuda().eval()

    function_map = load_function_map(args.info_root)
    sample_rows = []
    part_rows = []

    sample_csv = args.output_dir / f"{args.preset}_sample_metrics_{args.start}_{'end' if args.end is None else args.end}.csv"
    part_csv = args.output_dir / f"{args.preset}_part_metrics_{args.start}_{'end' if args.end is None else args.end}.csv"
    sample_fields = [
        "preset",
        "entry",
        "shape_id",
        "condition_mode",
        "guidance_scale",
        "sample_dir",
        "sample_success",
        "all_watertight",
        "wheel_watertight_count",
        "wheel_single_component_count",
        "max_components",
        "latent_std",
        "latent_mean",
        "latent_min",
        "latent_max",
        "latent_abs_gt1",
        "latent_abs_gt2",
    ]
    part_fields = [
        "preset",
        "entry",
        "shape_id",
        "condition_mode",
        "guidance_scale",
        "sample_dir",
        "part_file",
        "part_group",
        "file_size_mb",
        "vertices",
        "faces",
        "watertight",
        "components",
        "largest_component_face_ratio",
        "single_component",
        "extent_x",
        "extent_y",
        "extent_z",
    ]

    with sample_csv.open("w", newline="") as sf, part_csv.open("w", newline="") as pf:
        sample_writer = csv.DictWriter(sf, fieldnames=sample_fields)
        part_writer = csv.DictWriter(pf, fieldnames=part_fields)
        sample_writer.writeheader()
        part_writer.writeheader()

        for entry, condition_mode, guidance_scale in PRESETS[args.preset]:
            for shape_id in shape_ids:
                try:
                    sample_dir, latents = existing_or_generate_sample(
                        model,
                        args.checkpoint,
                        args,
                        entry,
                        condition_mode,
                        guidance_scale,
                        shape_id,
                        function_map,
                    )
                    base_meta = {
                        "preset": args.preset,
                        "entry": entry,
                        "shape_id": shape_id,
                        "condition_mode": condition_mode,
                        "guidance_scale": guidance_scale,
                        "sample_dir": sample_dir.as_posix(),
                    }
                    rows = [
                        mesh_part_row(mesh_path, base_meta)
                        for mesh_path in sorted((sample_dir / "generated_part_mesh").glob("*.ply"))
                    ]
                    wheel_rows = [row for row in rows if row["part_group"] == "wheel"]
                    summary = {
                        **base_meta,
                        "sample_success": bool(
                            rows
                            and all(bool(r["watertight"]) for r in rows)
                            and len(wheel_rows) == 4
                            and all(int(r["components"]) == 1 for r in wheel_rows)
                        ),
                        "all_watertight": bool(rows and all(bool(r["watertight"]) for r in rows)),
                        "wheel_watertight_count": sum(int(bool(r["watertight"])) for r in wheel_rows),
                        "wheel_single_component_count": sum(int(r["components"]) == 1 for r in wheel_rows),
                        "max_components": max([int(r["components"]) for r in rows] or [0]),
                        **latent_stats(latents),
                    }
                except Exception as exc:
                    (args.output_dir / f"{args.preset}_errors.log").open("a").write(f"{entry},{shape_id}: {exc}\n")
                    summary = {
                        "preset": args.preset,
                        "entry": entry,
                        "shape_id": shape_id,
                        "condition_mode": condition_mode,
                        "guidance_scale": guidance_scale,
                        "sample_dir": "",
                        "sample_success": False,
                        "all_watertight": False,
                        "wheel_watertight_count": 0,
                        "wheel_single_component_count": 0,
                        "max_components": 0,
                        "latent_std": math.nan,
                        "latent_mean": math.nan,
                        "latent_min": math.nan,
                        "latent_max": math.nan,
                        "latent_abs_gt1": math.nan,
                        "latent_abs_gt2": math.nan,
                    }
                    rows = []

                sample_writer.writerow(summary)
                sf.flush()
                sample_rows.append(summary)
                for row in rows:
                    part_writer.writerow(row)
                    part_rows.append(row)
                pf.flush()

    write_summary(sample_rows, args.output_dir)
    print(args.output_dir.as_posix())


if __name__ == "__main__":
    main()
