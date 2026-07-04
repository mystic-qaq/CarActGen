from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(REPO_ROOT.as_posix())

from experiments.layout_net import (
    PART_ORDER,
    batch_from_sample_condition,
    decode_layout_target,
    load_layout_checkpoint,
    load_shape_condition,
    predict_layout_vector,
)


def env_path(name: str, default: Path | str) -> Path:
    return Path(os.environ.get(name, default))


def load_generated_latents(path: Path) -> np.ndarray:
    latents = torch.load(path, map_location="cpu")
    if torch.is_tensor(latents):
        latents = latents.detach().cpu().numpy()
    latents = np.asarray(latents, dtype=np.float32)
    if latents.ndim == 3 and latents.shape[0] == 1:
        latents = latents[0]
    if latents.shape != (len(PART_ORDER), 768):
        raise ValueError(f"expected generated latents with shape {(len(PART_ORDER), 768)}, got {latents.shape} from {path}")
    return latents


def patch_structure(structure_path: Path, layout: dict, output_path: Path):
    structure = json.loads(structure_path.read_text())
    for part in structure.get("parts", []):
        spec = layout.get(part.get("name"))
        if not spec:
            continue
        part["bbx"] = spec["bbx"]
        part["joint_data_origin"] = spec["joint_data_origin"]
        part["joint_data_direction"] = spec["joint_data_direction"]
    structure["template_mode"] = "layout-net"
    output_path.write_text(json.dumps(structure, indent=2) + "\n")


def iter_sample_dirs(samples_root: Path, method_prefixes: tuple[str, ...]):
    for method_dir in sorted(path for path in samples_root.iterdir() if path.is_dir()):
        if method_prefixes and not any(method_dir.name.startswith(prefix) for prefix in method_prefixes):
            continue
        for sample_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
            if (sample_dir / "generated_latents.pt").exists():
                yield method_dir.name, sample_dir


def main():
    parser = argparse.ArgumentParser(description="Apply a trained LayoutNet checkpoint to saved PartLocal sample latents.")
    parser.add_argument(
        "--samples_root",
        type=Path,
        required=True,
        help="Directory containing method subdirectories, e.g. eval_clean_partlocal/samples.",
    )
    parser.add_argument(
        "--condition_root",
        type=Path,
        default=env_path(
            "CARACTGEN_CONDITION_ROOT",
            env_path("CARACTGEN_OUTPUT_ROOT", REPO_ROOT / "outputs")
            / "caractgen_clean_partlocal/datasets/2.1_clean_trainonly_vae_latent_sketch_dinov2",
        ),
        help="Clean latent/text/image condition directory used by LayoutNet.",
    )
    parser.add_argument(
        "--layout_checkpoint",
        type=Path,
        default=env_path("CARACTGEN_LAYOUT_CKPT", REPO_ROOT / "checkpoints/layout_net/condition_latent/best.pt"),
    )
    parser.add_argument("--method_prefix", action="append", default=["partlocal"], help="Method directory prefix to process.")
    parser.add_argument("--disable_layout_symmetry", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats, payload = load_layout_checkpoint(args.layout_checkpoint, device=device)
    processed = 0
    skipped = 0
    failures = []
    for method_name, sample_dir in iter_sample_dirs(args.samples_root, tuple(args.method_prefix)):
        output_path = sample_dir / "layout_prediction.json"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        shape_id = sample_dir.name
        try:
            generated_latents = load_generated_latents(sample_dir / "generated_latents.pt")
            _source_latents, text, image, function_ids = load_shape_condition(args.condition_root, shape_id)
            vector = predict_layout_vector(
                model,
                stats,
                generated_latents,
                text,
                image,
                function_ids,
                device=device,
            )
            layout = decode_layout_target(vector, symmetrize=not args.disable_layout_symmetry)
            output = {
                "method": method_name,
                "shape_id": shape_id,
                "layout_checkpoint": args.layout_checkpoint.as_posix(),
                "layout_epoch": int(payload.get("epoch", -1)),
                "symmetrized": not args.disable_layout_symmetry,
                "layout": layout,
            }
            output_path.write_text(json.dumps(output, indent=2) + "\n")
            structure_path = sample_dir / "structure.json"
            if structure_path.exists():
                patch_structure(structure_path, layout, sample_dir / "structure_layout_net.json")
            processed += 1
        except Exception as exc:
            failures.append({"sample_dir": sample_dir.as_posix(), "error": str(exc)})

    summary = {"processed": processed, "skipped": skipped, "failed": len(failures), "failures": failures}
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
