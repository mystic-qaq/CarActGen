from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(REPO_ROOT.as_posix())

import utils.mesh as MeshUtils
from experiments.fixed_car_template import PART_ORDER, build_parts, load_source_parts, render_to_image, write_structure
from model.FunctionAware import (
    AdaptiveObjectGatedMultimodalFunctionAwareDiffusion,
    AdaptiveObjectMultimodalFunctionAwareDiffusion,
    AdaptiveObjectPartLocalMultimodalFunctionAwareDiffusion,
    FUNCTION_TO_ID,
    FunctionAwareSDFAutoEncoder,
    load_function_map,
)
from model.SDFAutoEncoder import SDFAutoEncoder


def env_path(name: str, default: Path | str | None = None) -> Path | None:
    value = os.environ.get(name)
    if value:
        return Path(value)
    return Path(default) if default is not None else None


def load_npz_condition(dataset_path: Path, stem: str, key_names):
    npz_path = dataset_path / f"{stem}.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    for key in key_names:
        if key in data.files:
            return data[key].astype(np.float32)
    return None


def load_image_condition(dataset_path: Path, stem: str, image_embedding_dir: Path | None):
    image = load_npz_condition(dataset_path, stem, ["image", "image_embedding", "image_embed", "image_feature", "images"])
    if image is not None:
        return image
    if image_embedding_dir is None:
        return None
    shape_stem = "_".join(stem.split("_")[:-1])
    for path in [image_embedding_dir / f"{stem}.npy", image_embedding_dir / f"{shape_stem}.npy"]:
        if path.exists():
            return np.load(path, allow_pickle=True).astype(np.float32)
    return None


def default_text_shape(model):
    m = model.config["model"]
    return tuple(m.get("text_shape", [16, m.get("text_dim", 1024)]))


def default_image_shape(model):
    m = model.config["model"]
    return tuple(m.get("image_shape", [32, m.get("image_dim", 1408)]))


def resolve_model_class_from_checkpoint(checkpoint_path: Path):
    payload = torch.load(checkpoint_path, map_location="cpu")
    hyper = payload.get("hyper_parameters", {})
    config = hyper.get("config") or hyper.get("configs") or {}
    architecture = config.get("model", {}).get("architecture", "baseline_object")
    if architecture == "gated_object":
        return AdaptiveObjectGatedMultimodalFunctionAwareDiffusion
    if architecture == "partlocal_object":
        return AdaptiveObjectPartLocalMultimodalFunctionAwareDiffusion
    return AdaptiveObjectMultimodalFunctionAwareDiffusion


def build_condition_batch(args, model, function_map):
    max_parts = int(model.config["model"].get("max_parts", 16))
    text_shape = default_text_shape(model)
    image_shape = default_image_shape(model)

    text = np.zeros((1, max_parts) + text_shape, dtype=np.float32)
    image = np.zeros((1, max_parts) + image_shape, dtype=np.float32)
    has_text = np.zeros((1, max_parts), dtype=np.float32)
    has_image = np.zeros((1, max_parts), dtype=np.float32)
    function_ids = np.zeros((1, max_parts), dtype=np.int64)
    mask = np.zeros((1, max_parts), dtype=np.float32)

    for idx, _name in enumerate(PART_ORDER, start=1):
        token = idx - 1
        stem = f"{args.shape_id}_{idx}"
        function_id, _ = function_map.get(stem, (FUNCTION_TO_ID["static_part"], "static_part"))
        function_ids[0, token] = function_id
        mask[0, token] = 1.0

        text_value = load_npz_condition(args.dataset_path, stem, ["text", "text_embedding", "text_embed"])
        if text_value is not None:
            text[0, token] = text_value.astype(np.float32)
            has_text[0, token] = 1.0

        image_value = load_image_condition(args.dataset_path, stem, args.image_embedding_dir)
        if image_value is not None:
            if image_value.ndim == 1:
                image_value = image_value[None, :]
            image[0, token] = image_value.astype(np.float32)
            has_image[0, token] = 1.0

    return {
        "function_id": torch.from_numpy(function_ids),
        "text": torch.from_numpy(text),
        "image": torch.from_numpy(image),
        "has_text": torch.from_numpy(has_text),
        "has_image": torch.from_numpy(has_image),
        "mask": torch.from_numpy(mask),
    }


def decode_latents(model, latents, function_ids, output_dir: Path, args):
    if model.sdf is None:
        raise RuntimeError("SDF model was not loaded in the diffusion checkpoint config.")
    model.sdf = model.sdf.to(latents.device).eval()
    mesh_dir = output_dir / "generated_part_mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    meshes = {}
    with torch.no_grad():
        if hasattr(model.sdf, "decode_latent"):
            planes = model.sdf.decode_latent(latents, function_ids.to(latents.device))
        else:
            planes = model.sdf.vae_model.decode(latents)
    for idx, name in enumerate(PART_ORDER):
        mesh_path = mesh_dir / f"{idx + 1:02d}_{name}.ply"
        create_kwargs = {
            "N": args.sdf_resolution,
            "max_batch": args.max_batch,
            "from_plane_features": True,
        }
        if hasattr(model.sdf, "decode_latent"):
            create_kwargs["function_id"] = function_ids[[idx]].to(latents.device)
        MeshUtils.create_mesh(model.sdf, planes[[idx]], mesh_path.as_posix(), **create_kwargs)
        meshes[name] = trimesh.load_mesh(mesh_path.as_posix())
    return meshes


def ensure_sdf_loaded(model, device):
    if model.sdf is not None:
        model.sdf = model.sdf.to(device).eval()
        model.sdf.requires_grad_(False)
        return
    sdf_path = model.config.get("evaluation", {}).get("sdf_model_path")
    if not sdf_path:
        return
    try:
        model.sdf = FunctionAwareSDFAutoEncoder.load_from_checkpoint(str(sdf_path), map_location=device)
    except Exception:
        model.sdf = SDFAutoEncoder.load_from_checkpoint(str(sdf_path), map_location=device)
    model.sdf = model.sdf.to(device).eval()
    model.sdf.requires_grad_(False)


def maybe_copy_viewer(output_dir: Path, viewer_template: Path | None):
    link = output_dir / "reconstructed_part_mesh"
    if not link.exists():
        try:
            os.symlink("generated_part_mesh", link, target_is_directory=True)
        except FileExistsError:
            pass
    if viewer_template and viewer_template.exists():
        shutil.copyfile(viewer_template, output_dir / "viewer.html")


def main():
    parser = argparse.ArgumentParser(description="Sample fixed car parts from object-level adaptive multimodal diffusion.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--shape_id", default="car_drivaer_117")
    parser.add_argument("--condition_mode", choices=["unconditional", "text", "image", "text_image"], default="unconditional")
    parser.add_argument("--image_embedding_dir", type=Path, default=None)
    parser.add_argument("--info_root", type=Path, default=env_path("CARACTGEN_INFO_ROOT", env_path("CARACTGEN_DATA_ROOT", REPO_ROOT / "data/datasets") / "1_preprocessed_info"))
    parser.add_argument("--mesh_root", type=Path, default=env_path("CARACTGEN_MESH_ROOT", env_path("CARACTGEN_DATA_ROOT", REPO_ROOT / "data/datasets") / "1_preprocessed_mesh"))
    parser.add_argument("--output_root", type=Path, default=env_path("CARACTGEN_OUTPUT_ROOT", REPO_ROOT / "outputs") / "samples")
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--clip_denoised", type=float, default=2.5)
    parser.add_argument("--sdf_resolution", type=int, default=128)
    parser.add_argument("--max_batch", type=int, default=32768)
    parser.add_argument("--viewer_template", type=Path, default=env_path("CARACTGEN_VIEWER_TEMPLATE"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because mesh extraction uses .cuda().")
    device = torch.device("cuda")

    model_class = resolve_model_class_from_checkpoint(Path(args.checkpoint))
    model = model_class.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device).eval()
    model.requires_grad_(False)
    ensure_sdf_loaded(model, device)

    function_map = load_function_map(args.info_root)
    batch = build_condition_batch(args, model, function_map)
    use_text = args.condition_mode in {"text", "text_image"}
    use_image = args.condition_mode in {"image", "text_image"}
    if use_text and not bool(batch["has_text"].any()):
        raise ValueError("Text condition requested, but no text embeddings were found.")
    if use_image and not bool(batch["has_image"].any()):
        raise ValueError("Image condition requested, but no image embeddings were found.")

    with torch.no_grad():
        latents = model.sample_latents(
            batch,
            guidance_scale=args.guidance_scale,
            use_text=use_text,
            use_image=use_image,
            clip_denoised=args.clip_denoised,
        )
    latents = latents[0, : len(PART_ORDER)]
    function_ids = batch["function_id"][0, : len(PART_ORDER)]

    output_dir = args.output_root / f"{time.strftime('%m-%d-%H%M%S')}_{args.shape_id}_{args.condition_mode}_object_ddpm_cfg{args.guidance_scale}"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(latents.detach().cpu(), output_dir / "generated_latents.pt")

    _, source_parts = load_source_parts(args.shape_id, args.info_root, args.mesh_root)
    meshes = decode_latents(model, latents, function_ids, output_dir, args)
    generated_source_parts = {
        name: {"source": source_parts[name]["source"], "mesh": meshes[name], "mesh_path": output_dir / "generated_part_mesh" / f"{idx + 1:02d}_{name}.ply"}
        for idx, name in enumerate(PART_ORDER)
    }
    parts = build_parts(generated_source_parts, "source-bbox")
    write_structure(parts, output_dir / "structure.json", {"category": "car", "shape_id": args.shape_id}, "source-bbox")
    with open(output_dir / "processed_nodes.pkl", "wb") as f:
        pickle.dump(parts, f)
    render_to_image(parts, 0.0).save(output_dir / "pose_000.png")
    maybe_copy_viewer(output_dir, args.viewer_template)
    (output_dir / "manifest.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print(output_dir.as_posix())


if __name__ == "__main__":
    main()
