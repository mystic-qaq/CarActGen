from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs/1_SDF/train_function_aware_car_adaptive.yaml"
SPLIT_PATH = Path(
    os.environ.get(
        "CARACTGEN_SPLIT_PATH",
        REPO_ROOT / "data/caractgen_metadata/splits/object_sketch_dinov2_partlocal_seed123456798.json",
    )
)
BASE_DATASET = Path(os.environ.get("CARACTGEN_DATA_ROOT", REPO_ROOT / "data/datasets"))
OUTPUT_ROOT = Path(os.environ.get("CARACTGEN_OUTPUT_ROOT", REPO_ROOT / "outputs")) / "vae_ablations_clean"


ABLATIONS = [
    "no_adaptive_sampling",
    "no_decoder_film",
    "no_eikonal",
    "no_film_conditioning",
    "no_function_loss_weight",
    "no_plane_recon",
]


def apply_ablation(cfg: dict, name: str):
    fa = cfg["function_aware"]
    data = cfg["dataset_n_dataloader"]
    if name == "no_adaptive_sampling":
        data["sdf_subdir"] = "2_gensdf_dataset"
        data.pop("uniform_sample_ratio_by_function", None)
        data.pop("sample_repeat_by_function", None)
        data.pop("val_sample_repeat_by_function", None)
        data.pop("point_cloud_surface_ratio_by_function", None)
    elif name == "no_decoder_film":
        fa["decoder_film"] = False
    elif name == "no_eikonal":
        fa["eikonal_weight"] = 0.0
    elif name == "no_film_conditioning":
        fa["decoder_film"] = False
        fa["latent_film"] = False
        fa["latent_shift"] = False
    elif name == "no_function_loss_weight":
        fa["loss_weight_by_function"] = {}
    elif name == "no_plane_recon":
        fa["plane_recon_weight"] = 0.0
    else:
        raise ValueError(f"unknown ablation: {name}")


def build_config(args, name: str) -> dict:
    cfg = yaml.safe_load(args.base_config.read_text())
    cfg["wandb"]["use"] = False
    cfg["seed"] = int(args.seed)
    cfg["default_root_dir"] = (args.output_root / name).as_posix()

    cfg.pop("initialize_from_function_aware_sdf", None)
    cfg["initialize_from_sdf"] = args.original_trainonly_vae.as_posix()

    data = cfg["dataset_n_dataloader"]
    data["dataset_dir"] = args.base_dataset.as_posix()
    data["mesh_info_dir"] = (args.base_dataset / "1_preprocessed_info").as_posix()
    data["sdf_subdir"] = "2_gensdf_dataset_adaptive"
    data["split_path"] = args.split_path.as_posix()
    data["train_split"] = "train"
    data["val_split"] = "val"
    data["n_workers"] = int(args.num_workers)
    data["batch_size"] = int(args.batch_size)

    cfg["evaluation"]["eval_mesh_output_path"] = (args.output_root / name / "tempmesh").as_posix()
    cfg["evaluation"]["resolution"] = int(args.eval_resolution)
    cfg["evaluation"]["count"] = int(args.eval_count)
    cfg["evaluation"]["freq_epoch"] = int(args.val_freq)
    cfg["evaluation"]["vis_epoch_freq"] = 1000000

    cfg["checkpoint"]["path"] = (args.output_root / name / "checkpoint").as_posix()
    cfg["checkpoint"]["freq"] = int(args.checkpoint_freq_steps)
    cfg["checkpoint"]["freq_epoch"] = int(args.val_freq)
    cfg["checkpoint"]["save_top_k"] = int(args.save_top_k)
    cfg["checkpoint"]["save_last"] = True
    cfg["checkpoint"]["monitor"] = "val_loss"
    cfg["checkpoint"]["mode"] = "min"
    cfg["checkpoint"]["filename"] = "function_sdf_{epoch:04d}-{val_loss:.5f}"

    cfg["num_epochs"] = int(args.epochs)
    cfg["sdf_lr"] = float(args.lr)
    apply_ablation(cfg, name)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Prepare clean train/val VAE ablation configs.")
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--base_config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--base_dataset", type=Path, default=BASE_DATASET)
    parser.add_argument("--split_path", type=Path, default=SPLIT_PATH)
    parser.add_argument("--original_trainonly_vae", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--checkpoint_freq_steps", type=int, default=500)
    parser.add_argument("--save_top_k", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_resolution", type=int, default=64)
    parser.add_argument("--eval_count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123456820)
    args = parser.parse_args()

    split = json.loads(args.split_path.read_text())
    train_ids = set(split["train"])
    val_ids = set(split["val"])
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise ValueError(f"train/val split overlap: {overlap[:10]}")
    if not args.original_trainonly_vae.exists():
        raise FileNotFoundError(args.original_trainonly_vae)

    config_dir = args.output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "output_root": args.output_root.as_posix(),
        "split_path": args.split_path.as_posix(),
        "train_shapes": len(split["train"]),
        "val_shapes": len(split["val"]),
        "test_shapes": len(split["test"]),
        "base_dataset": args.base_dataset.as_posix(),
        "original_trainonly_vae": args.original_trainonly_vae.as_posix(),
        "epochs": args.epochs,
        "validation": "best checkpoint selected by val_loss on validation split",
        "ablations": {},
    }
    for name in ABLATIONS:
        cfg = build_config(args, name)
        path = config_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        manifest["ablations"][name] = path.as_posix()
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print((args.output_root / "manifest.json").as_posix())


if __name__ == "__main__":
    main()

