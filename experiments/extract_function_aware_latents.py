import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())

from model.FunctionAware import FunctionAwareGenSDFDataset, FunctionAwareSDFAutoEncoder
from utils import to_cuda
from utils.mylogging import Log


def env_path(name: str, default: Path | str) -> Path:
    return Path(os.environ.get(name, default))


def load_existing_text(existing_dir: Path):
    result = {}
    if existing_dir is None or not existing_dir.exists():
        return result
    for npz_path in existing_dir.glob("*.npz"):
        data = np.load(npz_path, allow_pickle=True)
        if "text" in data.files:
            result[npz_path.stem] = {
                "text": data["text"].astype(np.float32),
                "text_label": data["text_label"] if "text_label" in data.files else np.array("", dtype=str),
            }
    return result


def shape_stem_from_part_stem(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 2:
        return stem
    return "_".join(parts[:-1])


def load_stats_shape_ids(split_path: Path | None, split_key: str) -> set[str] | None:
    if split_path is None:
        return None
    payload = json.loads(split_path.read_text())
    if split_key not in payload:
        raise KeyError(f"Split key {split_key!r} not found in {split_path}")
    return set(payload[split_key])


def load_image_embedding(image_dir: Path | None, stem: str):
    if image_dir is None or not image_dir.exists():
        return None
    candidates = [
        image_dir / f"{stem}.npy",
        image_dir / f"{stem}.npz",
        image_dir / f"{shape_stem_from_part_stem(stem)}.npy",
        image_dir / f"{shape_stem_from_part_stem(stem)}.npz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".npy":
            return np.load(path, allow_pickle=True).astype(np.float32)
        data = np.load(path, allow_pickle=True)
        for key in ["image", "image_embedding", "image_embed", "image_feature", "images"]:
            if key in data.files:
                return data[key].astype(np.float32)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf_ckpt_path", required=True)
    parser.add_argument("--dataset_dir", type=Path, default=env_path("CARACTGEN_DATA_ROOT", ROOT / "data/datasets"))
    parser.add_argument("--mesh_info_dir", type=Path, default=env_path("CARACTGEN_INFO_ROOT", env_path("CARACTGEN_DATA_ROOT", ROOT / "data/datasets") / "1_preprocessed_info"))
    parser.add_argument("--existing_text_latent_dir", type=Path, default=env_path("CARACTGEN_TEXT_LATENT_ROOT", env_path("CARACTGEN_DATA_ROOT", ROOT / "data/datasets") / "2.1_text_n_latentcode"))
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--samples_per_mesh", type=int, default=16000)
    parser.add_argument("--pc_size", type=int, default=4096)
    parser.add_argument("--uniform_sample_ratio", type=float, default=0.25)
    parser.add_argument("--sdf_subdir", default="2_gensdf_dataset")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--image_embedding_dir", type=Path, default=None)
    parser.add_argument("--stats_split_path", type=Path, default=None)
    parser.add_argument("--stats_split", default="train")
    parser.add_argument("--num_functions", type=int, default=8)
    parser.add_argument("--reset_output", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = args.output_path
    if args.reset_output:
        shutil.rmtree(output_path.as_posix(), ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)

    Log.info("Loading function-aware SDF checkpoint %s", args.sdf_ckpt_path)
    model = FunctionAwareSDFAutoEncoder.load_from_checkpoint(args.sdf_ckpt_path, map_location=device)
    model = model.to(device)
    model.eval()

    text_map = load_existing_text(args.existing_text_latent_dir)
    rng = np.random.RandomState(42)

    dataset = FunctionAwareGenSDFDataset(
        dataset_dir=args.dataset_dir,
        train=None,
        samples_per_mesh=args.samples_per_mesh,
        pc_size=args.pc_size,
        uniform_sample_ratio=args.uniform_sample_ratio,
        mesh_info_dir=args.mesh_info_dir,
        sdf_subdir=args.sdf_subdir,
        sample_repeat_by_function=None,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    success = []
    for batch in tqdm(loader, desc="Extracting function-aware latents"):
        x = to_cuda(batch)
        with torch.no_grad():
            latents = model.encode_latent_from_point_cloud(
                x["point_cloud"],
                x["function_id"],
                deterministic=True,
            )
        latents = latents.detach().cpu().numpy()
        function_ids = batch["function_id"].numpy()
        function_labels = batch["function_label"]
        filenames = batch["filename"]

        for idx, filename in enumerate(filenames):
            stem = Path(filename).stem.replace(".sdf", "")
            text_entry = text_map.get(stem)
            if text_entry is None:
                text = rng.randn(16, 1024).astype(np.float32)
                text_label = np.array(function_labels[idx], dtype=str)
            else:
                text = text_entry["text"]
                text_label = text_entry["text_label"]
            image = load_image_embedding(args.image_embedding_dir, stem)
            payload = {
                "latent_code": latents[idx],
                "text": text,
                "text_label": text_label,
                "function_id": np.array(function_ids[idx], dtype=np.int64),
                "function_label": np.array(function_labels[idx], dtype=str),
            }
            if image is not None:
                payload["image"] = image
                payload["image_label"] = np.array(shape_stem_from_part_stem(stem), dtype=str)
            np.savez(output_path / f"{stem}.npz", **payload)
            success.append(stem)

    stats_shape_ids = load_stats_shape_ids(args.stats_split_path, args.stats_split)
    if success:
        stat_latents = []
        stat_function_ids = []
        for stem in success:
            if stats_shape_ids is not None and shape_stem_from_part_stem(stem) not in stats_shape_ids:
                continue
            data = np.load(output_path / f"{stem}.npz", allow_pickle=True)
            stat_latents.append(data["latent_code"].astype(np.float32))
            stat_function_ids.append(int(data["function_id"]) if "function_id" in data.files else 0)
        if not stat_latents:
            raise RuntimeError("No latents matched the requested stats split.")
        latent_stack = np.stack(stat_latents, axis=0)
        np.savez(
            output_path / "latent_stats.npz",
            mean=latent_stack.mean(axis=0).astype(np.float32),
            std=np.maximum(latent_stack.std(axis=0).astype(np.float32), 1e-6),
        )
        global_mean = latent_stack.mean(axis=0).astype(np.float32)
        global_std = np.maximum(latent_stack.std(axis=0).astype(np.float32), 1e-6)
        function_mean = np.repeat(global_mean[None], args.num_functions, axis=0)
        function_std = np.repeat(global_std[None], args.num_functions, axis=0)
        counts = np.zeros(args.num_functions, dtype=np.int64)
        for function_id in range(args.num_functions):
            values = [
                latent
                for latent, fid in zip(stat_latents, stat_function_ids)
                if max(0, min(int(fid), args.num_functions - 1)) == function_id
            ]
            counts[function_id] = len(values)
            if values:
                stack = np.stack(values, axis=0)
                function_mean[function_id] = stack.mean(axis=0).astype(np.float32)
                function_std[function_id] = np.maximum(stack.std(axis=0).astype(np.float32), 1e-6)
        np.savez(
            output_path / "function_latent_stats.npz",
            mean=function_mean.astype(np.float32),
            std=function_std.astype(np.float32),
            counts=counts,
            stats_split=np.array(args.stats_split, dtype=str),
            stats_split_path=np.array(args.stats_split_path.as_posix() if args.stats_split_path else "", dtype=str),
        )

    with open(output_path / "meta.json", "w") as f:
        json.dump({
            "ckpt": str(args.sdf_ckpt_path),
            "dataset_dir": args.dataset_dir.as_posix(),
            "sdf_subdir": args.sdf_subdir,
            "mesh_info_dir": args.mesh_info_dir.as_posix(),
            "image_embedding_dir": args.image_embedding_dir.as_posix() if args.image_embedding_dir else None,
            "latent_stats_path": (output_path / "latent_stats.npz").as_posix(),
            "function_latent_stats_path": (output_path / "function_latent_stats.npz").as_posix(),
            "stats_split_path": args.stats_split_path.as_posix() if args.stats_split_path else None,
            "stats_split": args.stats_split,
            "success": success,
            "success_count": len(success),
        }, f, indent=2)

    Log.info("Done. success count = %s", len(success))
