from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(REPO_ROOT.as_posix())

from experiments.fixed_car_template import PART_ORDER
from model.FunctionAware import FunctionAwareSDFAutoEncoder, load_function_map
from model.SDFAutoEncoder import SDFAutoEncoder


def env_path(name: str, default: Path | str | None = None) -> Path | None:
    value = os.environ.get(name)
    if value:
        return Path(value)
    return Path(default) if default is not None else None


def read_shape_ids(path: Path, split: str, max_shapes: int) -> list[str]:
    payload = json.loads(path.read_text())
    shape_ids = payload[split]
    return shape_ids if max_shapes <= 0 else shape_ids[:max_shapes]


def take_rows(array: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    array = np.asarray(array)
    if count <= 0 or array.shape[0] <= count:
        return array.astype(np.float32)
    idx = rng.choice(array.shape[0], size=count, replace=False)
    return array[idx].astype(np.float32)


def predict_sdf(model, plane: torch.Tensor, xyz: torch.Tensor, function_id: torch.Tensor | None, max_batch: int) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for head in range(0, xyz.shape[1], max_batch):
            chunk = xyz[:, head : head + max_batch]
            if hasattr(model, "decode_sdf_from_plane_features"):
                pred = model.decode_sdf_from_plane_features(plane, chunk, function_id=function_id)
            else:
                features = model.encoder.forward_with_plane_features(plane, chunk)
                pred = model.decoder(torch.cat((chunk, features), dim=-1))
            outputs.append(pred.squeeze(-1).detach().cpu())
    return torch.cat(outputs, dim=1)


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    pred = pred.float().flatten()
    target = target.float().flatten()
    err = pred - target
    mae = float(err.abs().mean())
    rmse = float(torch.sqrt((err * err).mean()))
    sign_acc = float(((pred < 0) == (target < 0)).float().mean())
    return {"mae": mae, "rmse": rmse, "sign_acc": sign_acc}


def latent_stats(z: torch.Tensor) -> dict:
    z = z.detach().cpu().float().flatten()
    return {
        "latent_mean": float(z.mean()),
        "latent_std": float(z.std()),
        "latent_abs_mean": float(z.abs().mean()),
        "latent_abs_p95": float(torch.quantile(z.abs(), 0.95)),
        "latent_abs_gt1": float((z.abs() > 1.0).float().mean()),
        "latent_abs_gt2": float((z.abs() > 2.0).float().mean()),
        "latent_l2": float(torch.linalg.vector_norm(z)),
    }


def encode_original(model, pc: torch.Tensor):
    with torch.no_grad():
        plane_features = model.encoder.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)
        mu, log_var = model.vae_model.encode(original_features)
        plane = model.vae_model.decode(mu)
        plane_l1 = F.l1_loss(plane, original_features).item()
    return plane, mu, plane_l1


def encode_function_aware(model, pc: torch.Tensor, function_id: torch.Tensor):
    with torch.no_grad():
        plane_features = model.encoder.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)
        conditioned = model._film(original_features, function_id, model.input_film)
        mu, log_var = model.vae_model.encode(conditioned)
        plane = model.decode_latent(mu, function_id)
        plane_l1 = F.l1_loss(plane, conditioned).item()
    return plane, mu, plane_l1


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        part_group = "body" if row["part_name"] == "body_shell" else "wheel"
        groups.setdefault((row["system"], "all"), []).append(row)
        groups.setdefault((row["system"], part_group), []).append(row)
        groups.setdefault((row["system"], row["function_label"]), []).append(row)

    out = []
    for (system, group), items in sorted(groups.items()):
        out.append(
            {
                "system": system,
                "group": group,
                "count": len(items),
                "surface_mae": np.nanmean([float(row["surface_mae"]) for row in items]),
                "uniform_mae": np.nanmean([float(row["uniform_mae"]) for row in items]),
                "surface_sign_acc": np.nanmean([float(row["surface_sign_acc"]) for row in items]),
                "uniform_sign_acc": np.nanmean([float(row["uniform_sign_acc"]) for row in items]),
                "plane_l1": np.nanmean([float(row["plane_l1"]) for row in items]),
                "latent_std": np.nanmean([float(row["latent_std"]) for row in items]),
                "latent_abs_gt1": np.nanmean([float(row["latent_abs_gt1"]) for row in items]),
                "latent_abs_gt2": np.nanmean([float(row["latent_abs_gt2"]) for row in items]),
                "latent_l2": np.nanmean([float(row["latent_l2"]) for row in items]),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict]):
    lines = [
        "# VAE SDF and latent summary",
        "",
        "| system | group | n | surface MAE | uniform MAE | surface sign | uniform sign | plane L1 | latent std | abs(z)>1 | abs(z)>2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['system']} | {row['group']} | {row['count']} | "
            f"{row['surface_mae']:.5f} | {row['uniform_mae']:.5f} | "
            f"{row['surface_sign_acc']:.3f} | {row['uniform_sign_acc']:.3f} | "
            f"{row['plane_l1']:.5f} | {row['latent_std']:.3f} | "
            f"{row['latent_abs_gt1']:.4f} | {row['latent_abs_gt2']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate original and function-aware SDF VAEs with held-out SDF queries.")
    parser.add_argument("--original_ckpt", type=Path, default=env_path("CARACTGEN_ORIGINAL_VAE_CKPT"))
    parser.add_argument("--function_ckpt", type=Path, default=env_path("CARACTGEN_FUNCTION_VAE_CKPT"))
    parser.add_argument("--split_path", type=Path, default=env_path("CARACTGEN_SPLIT_PATH", REPO_ROOT / "data/caractgen_metadata/splits/object_sketch_dinov2_partlocal_seed123456798.json"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_shapes", type=int, default=48)
    parser.add_argument("--eval_sdf_dataset", type=Path, default=env_path("CARACTGEN_SDF_ROOT", env_path("CARACTGEN_DATA_ROOT", REPO_ROOT / "data/datasets") / "2_gensdf_dataset_adaptive"))
    parser.add_argument("--info_root", type=Path, default=env_path("CARACTGEN_INFO_ROOT", env_path("CARACTGEN_DATA_ROOT", REPO_ROOT / "data/datasets") / "1_preprocessed_info"))
    parser.add_argument("--pc_size", type=int, default=4096)
    parser.add_argument("--query_samples", type=int, default=4096)
    parser.add_argument("--max_batch", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--output_dir", type=Path, default=env_path("CARACTGEN_OUTPUT_ROOT", REPO_ROOT / "outputs") / "caractgen_vae_sdf_latent")
    args = parser.parse_args()

    if args.original_ckpt is None or args.function_ckpt is None:
        raise ValueError("Set --original_ckpt and --function_ckpt, or CARACTGEN_ORIGINAL_VAE_CKPT and CARACTGEN_FUNCTION_VAE_CKPT.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluator.")

    device = torch.device("cuda")
    original = SDFAutoEncoder.load_from_checkpoint(args.original_ckpt, map_location=device).to(device).eval()
    function_aware = FunctionAwareSDFAutoEncoder.load_from_checkpoint(args.function_ckpt, map_location=device).to(device).eval()
    original.requires_grad_(False)
    function_aware.requires_grad_(False)

    function_map = load_function_map(args.info_root)
    shape_ids = read_shape_ids(args.split_path, args.split, args.max_shapes)
    (args.output_dir / "shape_ids.txt").write_text("\n".join(shape_ids) + "\n")

    rows = []
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

            specs = []
            try:
                plane, z, plane_l1 = encode_original(original, pc)
                specs.append(("original_vae", original, plane, None, z, plane_l1))
            except Exception as exc:
                (args.output_dir / "errors.log").open("a").write(f"original_vae,{stem}: {exc}\n")
            try:
                plane, z, plane_l1 = encode_function_aware(function_aware, pc, fn)
                specs.append(("function_aware_vae", function_aware, plane, fn, z, plane_l1))
            except Exception as exc:
                (args.output_dir / "errors.log").open("a").write(f"function_aware_vae,{stem}: {exc}\n")

            for system, model, plane, model_fn, z, plane_l1 in specs:
                try:
                    pred_surface = predict_sdf(model, plane, surface_xyz_t, model_fn, args.max_batch)
                    pred_uniform = predict_sdf(model, plane, uniform_xyz_t, model_fn, args.max_batch)
                    surface = regression_metrics(pred_surface, surface_sdf_t)
                    uniform = regression_metrics(pred_uniform, uniform_sdf_t)
                    row = {
                        "system": system,
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
                        "system": system,
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
                    (args.output_dir / "errors.log").open("a").write(f"{system},{stem}: {exc}\n")
                rows.append(row)

    write_csv(args.output_dir / "vae_sdf_latent_part_metrics.csv", rows)
    summary_rows = summarize(rows)
    write_csv(args.output_dir / "vae_sdf_latent_summary.csv", summary_rows)
    write_markdown(args.output_dir / "vae_sdf_latent_summary.md", summary_rows)
    print(args.output_dir.as_posix())


if __name__ == "__main__":
    main()
