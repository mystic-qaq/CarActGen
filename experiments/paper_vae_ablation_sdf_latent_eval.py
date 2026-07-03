from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(REPO_ROOT.as_posix())

from experiments.fixed_car_template import PART_ORDER
from experiments.paper_vae_sdf_latent_eval import (
    encode_function_aware,
    encode_original,
    latent_stats,
    predict_sdf,
    read_shape_ids,
    regression_metrics,
    summarize,
    take_rows,
    write_csv,
    write_markdown,
)
from model.FunctionAware import FunctionAwareSDFAutoEncoder, load_function_map
from model.SDFAutoEncoder import SDFAutoEncoder


ABLATION_ORDER = [
    "no_adaptive_sampling",
    "no_decoder_film",
    "no_eikonal",
    "no_film_conditioning",
    "no_function_loss_weight",
    "no_plane_recon",
]


def best_val_checkpoint(root: Path) -> Path | None:
    candidates = []
    for path in root.rglob("*.ckpt"):
        if path.name == "last.ckpt":
            continue
        match = re.search(r"val_loss=([0-9.]+)", path.name)
        if not match:
            continue
        value = float(match.group(1).rstrip("."))
        candidates.append((value, path.stat().st_mtime, path))
    if candidates:
        return min(candidates)[2]
    last = sorted(root.rglob("last.ckpt"), key=lambda item: item.stat().st_mtime, reverse=True)
    return last[0] if last else None


def systems_from_clean_ablation_root(args) -> list[dict]:
    manifest = json.loads((args.ablation_root / "manifest.json").read_text())
    systems = []
    if args.original_ckpt:
        systems.append({"name": "original_vae", "kind": "original", "checkpoint": args.original_ckpt.as_posix()})
    if args.full_ckpt:
        systems.append({"name": "full_function_aware", "kind": "function_aware", "checkpoint": args.full_ckpt.as_posix()})

    ablations = manifest.get("ablations", {})
    names = [name for name in ABLATION_ORDER if name in ablations]
    names.extend(sorted(name for name in ablations if name not in names))
    for name in names:
        ckpt = best_val_checkpoint(args.ablation_root / name / "checkpoint")
        if ckpt is None:
            continue
        systems.append({"name": name, "kind": "function_aware", "checkpoint": ckpt.as_posix()})
    return systems


def load_system(spec: dict, device: torch.device):
    if spec["kind"] == "original":
        model = SDFAutoEncoder.load_from_checkpoint(spec["checkpoint"], map_location=device).to(device).eval()
    else:
        model = FunctionAwareSDFAutoEncoder.load_from_checkpoint(spec["checkpoint"], map_location=device).to(device).eval()
    model.requires_grad_(False)
    return model


def evaluate_system(model, spec: dict, args, shape_ids: list[str], function_map: dict) -> list[dict]:
    rows = []
    device = args.device
    for shape_idx, shape_id in enumerate(shape_ids):
        for part_index, part_name in enumerate(PART_ORDER, start=1):
            stem = f"{shape_id}_{part_index}"
            sdf_path = args.eval_sdf_dataset / f"{stem}.sdf.npz"
            if not sdf_path.exists():
                continue
            data = np.load(sdf_path, allow_pickle=True)
            function_id_value, function_label = function_map.get(stem, (1, "static_part"))
            rng = np.random.default_rng(args.seed + shape_idx * 97 + part_index)

            pc_np = take_rows(data["point_on"], args.pc_size, rng)
            surface_xyz = take_rows(data["point_surface"], args.query_samples, rng)
            surface_sdf = take_rows(data["sdf_surface"], args.query_samples, rng)
            uniform_xyz = take_rows(data["point_uniform"], args.query_samples, rng)
            uniform_sdf = take_rows(data["sdf_uniform"], args.query_samples, rng)

            pc = torch.from_numpy(pc_np).unsqueeze(0).to(device)
            surface_xyz_t = torch.from_numpy(surface_xyz).unsqueeze(0).to(device)
            surface_sdf_t = torch.from_numpy(surface_sdf).view(1, -1)
            uniform_xyz_t = torch.from_numpy(uniform_xyz).unsqueeze(0).to(device)
            uniform_sdf_t = torch.from_numpy(uniform_sdf).view(1, -1)
            fn = torch.tensor([function_id_value], dtype=torch.long, device=device)

            try:
                if spec["kind"] == "original":
                    plane, z, plane_l1 = encode_original(model, pc)
                    model_fn = None
                else:
                    plane, z, plane_l1 = encode_function_aware(model, pc, fn)
                    model_fn = fn
                pred_surface = predict_sdf(model, plane, surface_xyz_t, model_fn, args.max_batch)
                pred_uniform = predict_sdf(model, plane, uniform_xyz_t, model_fn, args.max_batch)
                surface = regression_metrics(pred_surface, surface_sdf_t)
                uniform = regression_metrics(pred_uniform, uniform_sdf_t)
                row = {
                    "system": spec["name"],
                    "checkpoint": spec["checkpoint"],
                    "shape_id": shape_id,
                    "part_index": part_index,
                    "part_name": part_name,
                    "function_label": function_label,
                    "sdf_path": sdf_path.as_posix(),
                    "plane_l1": plane_l1,
                    "surface_mae": surface["mae"],
                    "surface_rmse": surface["rmse"],
                    "surface_sign_acc": surface["sign_acc"],
                    "uniform_mae": uniform["mae"],
                    "uniform_rmse": uniform["rmse"],
                    "uniform_sign_acc": uniform["sign_acc"],
                    **latent_stats(z),
                }
            except Exception as exc:
                row = {
                    "system": spec["name"],
                    "checkpoint": spec["checkpoint"],
                    "shape_id": shape_id,
                    "part_index": part_index,
                    "part_name": part_name,
                    "function_label": function_label,
                    "sdf_path": sdf_path.as_posix(),
                    "plane_l1": math.nan,
                    "surface_mae": math.nan,
                    "surface_rmse": math.nan,
                    "surface_sign_acc": math.nan,
                    "uniform_mae": math.nan,
                    "uniform_rmse": math.nan,
                    "uniform_sign_acc": math.nan,
                    "latent_mean": math.nan,
                    "latent_std": math.nan,
                    "latent_abs_mean": math.nan,
                    "latent_abs_p95": math.nan,
                    "latent_abs_gt1": math.nan,
                    "latent_abs_gt2": math.nan,
                    "latent_l2": math.nan,
                }
                (args.output_dir / "errors.log").open("a").write(f"{spec['name']},{stem}: {exc}\n")
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate clean VAE ablations with held-out SDF queries.")
    parser.add_argument("--ablation_root", type=Path, required=True)
    parser.add_argument("--original_ckpt", type=Path)
    parser.add_argument("--full_ckpt", type=Path)
    parser.add_argument("--split_path", type=Path, default=REPO_ROOT / "data/caractgen_metadata/splits/object_sketch_dinov2_partlocal_seed123456798.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_shapes", type=int, default=48)
    parser.add_argument("--eval_sdf_dataset", type=Path, required=True)
    parser.add_argument("--info_root", type=Path, required=True)
    parser.add_argument("--pc_size", type=int, default=4096)
    parser.add_argument("--query_samples", type=int, default=4096)
    parser.add_argument("--max_batch", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluator.")
    args.device = torch.device("cuda")

    systems = systems_from_clean_ablation_root(args)
    if not systems:
        raise RuntimeError(f"No systems found under {args.ablation_root}")
    (args.output_dir / "systems.json").write_text(json.dumps(systems, indent=2) + "\n")

    function_map = load_function_map(args.info_root)
    shape_ids = read_shape_ids(args.split_path, args.split, args.max_shapes)
    (args.output_dir / "shape_ids.txt").write_text("\n".join(shape_ids) + "\n")

    all_rows = []
    for spec in systems:
        print(f"Evaluating {spec['name']} from {spec['checkpoint']}", flush=True)
        model = load_system(spec, args.device)
        all_rows.extend(evaluate_system(model, spec, args, shape_ids, function_map))
        del model
        torch.cuda.empty_cache()

    write_csv(args.output_dir / "vae_ablation_sdf_latent_part_metrics.csv", all_rows)
    summary_rows = summarize(all_rows)
    write_csv(args.output_dir / "vae_ablation_sdf_latent_summary.csv", summary_rows)
    write_markdown(args.output_dir / "vae_ablation_sdf_latent_summary.md", summary_rows)
    print(args.output_dir.as_posix())


if __name__ == "__main__":
    main()

