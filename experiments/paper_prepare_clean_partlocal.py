from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("CARACTGEN_OUTPUT_ROOT", REPO_ROOT / "outputs")) / "caractgen_clean_partlocal"
SPLIT_PATH = Path(
    os.environ.get(
        "CARACTGEN_SPLIT_PATH",
        REPO_ROOT / "data/caractgen_metadata/splits/object_sketch_dinov2_partlocal_seed123456798.json",
    )
)
BASE_DATASET = Path(os.environ.get("CARACTGEN_DATA_ROOT", REPO_ROOT / "data/datasets"))
BASE_VAE_CONFIG = REPO_ROOT / "configs/1_SDF/train_function_aware_car_adaptive.yaml"
BASE_DIFF_CONFIG = REPO_ROOT / "configs/2_Diff/train_adaptive_object_multimodal_car_sketch_dinov2_partlocal.yaml"


def optional_env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


DEFAULT_TRAINONLY_VAE_INIT = optional_env_path("CARACTGEN_TRAINONLY_FUNCTION_VAE_CKPT")
DEFAULT_ORIGINAL_TRAINONLY_VAE = optional_env_path("CARACTGEN_ORIGINAL_TRAINONLY_VAE_CKPT")


def prepare_vae_config(args) -> Path:
    cfg = yaml.safe_load(args.base_vae_config.read_text())
    cfg["wandb"]["use"] = False
    cfg["seed"] = int(args.seed)
    cfg["default_root_dir"] = (args.output_root / "vae").as_posix()
    cfg["dataset_n_dataloader"]["dataset_dir"] = args.base_dataset.as_posix()
    cfg["dataset_n_dataloader"]["mesh_info_dir"] = (args.base_dataset / "1_preprocessed_info").as_posix()
    cfg["dataset_n_dataloader"]["sdf_subdir"] = args.sdf_subdir
    cfg["dataset_n_dataloader"]["split_path"] = args.split_path.as_posix()
    cfg["dataset_n_dataloader"]["train_split"] = "train"
    cfg["dataset_n_dataloader"]["val_split"] = "val"
    cfg["evaluation"]["eval_mesh_output_path"] = (args.output_root / "vae_tempmesh").as_posix()
    cfg["evaluation"]["freq_epoch"] = int(args.vae_val_freq)
    cfg["evaluation"]["vis_epoch_freq"] = 1000000
    cfg["checkpoint"]["path"] = (args.output_root / "vae_checkpoints").as_posix()
    cfg["checkpoint"]["freq"] = int(args.vae_checkpoint_freq_steps)
    cfg["checkpoint"]["freq_epoch"] = int(args.vae_val_freq)
    cfg["checkpoint"]["save_top_k"] = int(args.vae_save_top_k)
    cfg["checkpoint"]["save_last"] = True
    cfg["checkpoint"]["monitor"] = "val_loss"
    cfg["checkpoint"]["mode"] = "min"
    cfg["checkpoint"]["filename"] = "function_sdf_{epoch:04d}-{val_loss:.5f}"
    cfg["num_epochs"] = int(args.vae_epochs)
    cfg["sdf_lr"] = float(args.vae_lr)

    init_ckpt = args.vae_init_ckpt
    if init_ckpt and init_ckpt.exists():
        cfg.pop("initialize_from_sdf", None)
        cfg["initialize_from_function_aware_sdf"] = init_ckpt.as_posix()
    elif args.original_trainonly_vae and args.original_trainonly_vae.exists():
        cfg.pop("initialize_from_function_aware_sdf", None)
        cfg["initialize_from_sdf"] = args.original_trainonly_vae.as_posix()
    else:
        raise FileNotFoundError(
            "No clean VAE initializer found. Set --vae_init_ckpt for a clean "
            "function-aware checkpoint, or --original_trainonly_vae for a clean "
            "original VAE checkpoint."
        )

    path = args.output_root / "configs" / "train_function_aware_vae_clean.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def prepare_diff_config(args, latent_dataset: Path) -> Path:
    cfg = yaml.safe_load(args.base_diff_config.read_text())
    cfg["wandb"]["use"] = False
    cfg["seed"] = int(args.seed) + 1
    cfg["default_root_dir"] = (args.output_root / "partlocal_diffusion").as_posix()
    cfg["num_epochs"] = int(args.diff_epochs)
    cfg["dataset_n_dataloader"]["dataset_path"] = latent_dataset.as_posix()
    cfg["dataset_n_dataloader"]["split_path"] = args.split_path.as_posix()
    cfg["checkpoint"]["path"] = (args.output_root / "partlocal_diffusion" / "checkpoint").as_posix()
    cfg["checkpoint"]["freq"] = int(args.diff_val_freq)
    cfg["checkpoint"]["save_top_k"] = int(args.diff_save_top_k)
    cfg["checkpoint"]["save_last"] = True
    cfg["checkpoint"]["save_on_train_epoch_end"] = True
    cfg["evaluation"]["freq_epoch"] = int(args.diff_val_freq)
    cfg["evaluation"]["eval_mesh_output_path"] = (args.output_root / "diff_tempmesh").as_posix()

    path = args.output_root / "configs" / "train_partlocal_clean.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def write_manifest(args, vae_config: Path, diff_config: Path, latent_dataset: Path):
    split = json.loads(args.split_path.read_text())
    manifest = {
        "output_root": args.output_root.as_posix(),
        "split_path": args.split_path.as_posix(),
        "train_shapes": len(split["train"]),
        "val_shapes": len(split["val"]),
        "test_shapes": len(split["test"]),
        "base_dataset": args.base_dataset.as_posix(),
        "vae_config": vae_config.as_posix(),
        "diff_config": diff_config.as_posix(),
        "latent_dataset": latent_dataset.as_posix(),
        "protocol": {
            "vae_training_shapes": "train split only",
            "vae_validation_shapes": "validation split only",
            "diffusion_training_shapes": "train split only via split_path",
            "diffusion_validation_shapes": "validation split only via split_path",
            "latent_normalization_stats": "computed from train split only",
            "test_shapes": "held out from VAE training, VAE checkpoint selection, diffusion training, diffusion checkpoint selection, and latent normalization",
        },
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare clean train-only-VAE PartLocal rerun configs.")
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--split_path", type=Path, default=SPLIT_PATH)
    parser.add_argument("--base_dataset", type=Path, default=BASE_DATASET)
    parser.add_argument("--base_vae_config", type=Path, default=BASE_VAE_CONFIG)
    parser.add_argument("--base_diff_config", type=Path, default=BASE_DIFF_CONFIG)
    parser.add_argument("--vae_init_ckpt", type=Path, default=DEFAULT_TRAINONLY_VAE_INIT)
    parser.add_argument("--original_trainonly_vae", type=Path, default=DEFAULT_ORIGINAL_TRAINONLY_VAE)
    parser.add_argument("--sdf_subdir", default="2_gensdf_dataset_adaptive")
    parser.add_argument("--seed", type=int, default=123456810)
    parser.add_argument("--vae_epochs", type=int, default=160)
    parser.add_argument("--vae_lr", type=float, default=1e-5)
    parser.add_argument("--vae_val_freq", type=int, default=10)
    parser.add_argument("--vae_checkpoint_freq_steps", type=int, default=500)
    parser.add_argument("--vae_save_top_k", type=int, default=3)
    parser.add_argument("--diff_epochs", type=int, default=5000)
    parser.add_argument("--diff_val_freq", type=int, default=20)
    parser.add_argument("--diff_save_top_k", type=int, default=5)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    latent_dataset = args.output_root / "datasets" / "2.1_clean_trainonly_vae_latent_sketch_dinov2"
    vae_config = prepare_vae_config(args)
    diff_config = prepare_diff_config(args, latent_dataset)
    write_manifest(args, vae_config, diff_config, latent_dataset)
    print((args.output_root / "manifest.json").as_posix())


if __name__ == "__main__":
    main()
