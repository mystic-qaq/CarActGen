from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())

from model.FunctionAware import (
    AdaptiveObjectGatedMultimodalFunctionAwareDiffusion,
    AdaptiveObjectMultimodalDiffusionDataset,
    AdaptiveObjectMultimodalFunctionAwareDiffusion,
    AdaptiveObjectPartLocalMultimodalFunctionAwareDiffusion,
)
from utils import parse_config_from_args
from utils.mylogging import Log


os.environ["WANDB_CACHE_DIR"] = (Path() / "wandb/cache").resolve().as_posix()
os.environ["WANDB_DATA_DIR"] = (Path() / "wandb/data").resolve().as_posix()


MODEL_REGISTRY = {
    "baseline_object": AdaptiveObjectMultimodalFunctionAwareDiffusion,
    "gated_object": AdaptiveObjectGatedMultimodalFunctionAwareDiffusion,
    "partlocal_object": AdaptiveObjectPartLocalMultimodalFunctionAwareDiffusion,
}


def resolve_model_class(config):
    architecture = config.get("model", {}).get("architecture", "baseline_object")
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Unknown object diffusion architecture: {architecture}")
    return MODEL_REGISTRY[architecture]


def _normalize_fixed_ids(dataset, shape_ids):
    if not shape_ids:
        return []
    valid = []
    available = {shape_id for shape_id, _parts in dataset.objects}
    for shape_id in shape_ids:
        if shape_id in available:
            valid.append(shape_id)
        else:
            Log.warning("Fixed split id %s was requested but is not present in the dataset", shape_id)
    return valid


def build_or_load_split(
    dataset,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    split_path: Path | None,
    fixed_val_ids=None,
    fixed_test_ids=None,
):
    shape_to_index = {shape_id: idx for idx, (shape_id, _parts) in enumerate(dataset.objects)}
    fixed_val_ids = _normalize_fixed_ids(dataset, fixed_val_ids)
    fixed_test_ids = _normalize_fixed_ids(dataset, fixed_test_ids)
    overlap = sorted(set(fixed_val_ids) & set(fixed_test_ids))
    if overlap:
        raise ValueError(f"Shape ids cannot be fixed to both val and test: {overlap}")

    if split_path and split_path.exists():
        split = json.loads(split_path.read_text())
        train_ids = split["train"]
        val_ids = split["val"]
        test_ids = split.get("test", [])
        if fixed_val_ids and not set(fixed_val_ids).issubset(set(val_ids)):
            raise ValueError(f"Existing split at {split_path} does not contain the required fixed val ids: {fixed_val_ids}")
        if fixed_test_ids and not set(fixed_test_ids).issubset(set(test_ids)):
            raise ValueError(f"Existing split at {split_path} does not contain the required fixed test ids: {fixed_test_ids}")
    else:
        perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
        reserved = set(fixed_val_ids) | set(fixed_test_ids)
        free_idx = [idx for idx in perm if dataset.objects[idx][0] not in reserved]

        target_test_len = max(len(fixed_test_ids), int(round(len(dataset) * test_ratio))) if test_ratio > 0 else len(fixed_test_ids)
        target_val_len = max(len(fixed_val_ids), int(round(max(1, len(dataset) - target_test_len) * val_ratio))) if val_ratio > 0 else len(fixed_val_ids)

        extra_test_len = max(0, target_test_len - len(fixed_test_ids))
        extra_val_len = max(0, target_val_len - len(fixed_val_ids))

        extra_test_idx = free_idx[:extra_test_len]
        extra_val_idx = free_idx[extra_test_len : extra_test_len + extra_val_len]
        extra_train_idx = free_idx[extra_test_len + extra_val_len :]

        test_ids = fixed_test_ids + [dataset.objects[idx][0] for idx in extra_test_idx]
        val_ids = fixed_val_ids + [dataset.objects[idx][0] for idx in extra_val_idx]
        train_ids = [dataset.objects[idx][0] for idx in extra_train_idx]

        if not train_ids:
            raise ValueError("Split construction produced an empty train set.")

        test_len = len(test_ids)
        remaining = max(1, len(dataset) - test_len)
        val_len = max(1, int(round(remaining * val_ratio))) if val_ratio > 0 else 0

        split = {
            "seed": int(seed),
            "size": len(dataset),
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
            "fixed_val_ids": fixed_val_ids,
            "fixed_test_ids": fixed_test_ids,
        }
        if split_path:
            split_path.parent.mkdir(parents=True, exist_ok=True)
            split_path.write_text(json.dumps(split, indent=2))

    train_idx = [shape_to_index[shape_id] for shape_id in train_ids]
    val_idx = [shape_to_index[shape_id] for shape_id in val_ids]
    test_idx = [shape_to_index[shape_id] for shape_id in test_ids]
    return train_idx, val_idx, test_idx


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    run_name = os.environ.get("RUN_NAME", time.strftime("%m-%d-%I%p-%M-%S"))
    config = parse_config_from_args()
    seed_everything(config["seed"])

    d_configs = config["dataset_n_dataloader"]
    model_config = config["model"]
    dataset = AdaptiveObjectMultimodalDiffusionDataset(
        dataset_path=Path(d_configs["dataset_path"]),
        text_shape=tuple(model_config.get("text_shape", [16, model_config.get("text_dim", 1024)])),
        image_shape=tuple(model_config.get("image_shape", [32, model_config.get("image_dim", 1408)])),
        max_parts=int(model_config.get("max_parts", 16)),
        num_functions=len(config.get("function_aware", {}).get("vocab", [])) or 8,
        cache=d_configs.get("cache", True),
        normalize_latents=True,
    )

    val_ratio = float(d_configs.get("val_ratio", 0.10))
    test_ratio = float(d_configs.get("test_ratio", 0.0))
    split_path = Path(d_configs["split_path"]) if d_configs.get("split_path") else None
    train_idx, val_idx, test_idx = build_or_load_split(
        dataset,
        seed=int(config["seed"]),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_path=split_path,
        fixed_val_ids=d_configs.get("fixed_val_ids"),
        fixed_test_ids=d_configs.get("fixed_test_ids"),
    )
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx) if test_idx else None
    loader = DataLoader(
        train_set,
        num_workers=d_configs["n_workers"],
        batch_size=d_configs["batch_size"],
        drop_last=len(train_set) >= d_configs["batch_size"],
        shuffle=True,
        pin_memory=True,
        persistent_workers=d_configs["n_workers"] > 0,
    )
    val_loader = DataLoader(
        val_set,
        num_workers=max(0, min(2, d_configs["n_workers"])),
        batch_size=d_configs["batch_size"],
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        persistent_workers=d_configs["n_workers"] > 0,
    )
    config.setdefault("latent_normalization", {})["stats_path"] = dataset.stats_path.as_posix()
    config["evaluation"]["sdf_model_path"] = dataset.get_gensdf_ckpt_path()
    print("Len =", len(dataset), "train =", len(train_set), "val =", len(val_set), "test =", 0 if test_set is None else len(test_set))
    print("function_latent_stats =", dataset.stats_path)
    if split_path:
        print("split_path =", split_path)
    if test_set is not None:
        print("example_test_shape =", dataset.objects[test_idx[0]][0] if test_idx else "none")

    optional_kw_args = {}
    if config["wandb"].get("use", False):
        optional_kw_args["logger"] = WandbLogger(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            name=run_name,
            log_model=False,
        )

    if isinstance(config["devices"], list) or config["devices"] > 1:
        find_unused = bool(config.get("ddp_find_unused_parameters", False))
        optional_kw_args["strategy"] = DDPStrategy(find_unused_parameters=find_unused)

    model = resolve_model_class(config)(config)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=config["checkpoint"].get("save_top_k", 3),
        save_last=config["checkpoint"].get("save_last", True),
        save_on_train_epoch_end=config["checkpoint"].get("save_on_train_epoch_end", True),
        every_n_epochs=config["checkpoint"]["freq"],
        dirpath=config["checkpoint"]["path"] + "/" + run_name,
        filename="adaptive_object_multimodal_diffusion-{epoch:04d}-{val_loss:.5f}",
        monitor="val_loss",
        mode="min",
    )

    trainer = Trainer(
        devices=config["devices"],
        accelerator=config["accelerator"],
        benchmark=True,
        callbacks=[ModelSummary(max_depth=1), checkpoint_callback, TQDMProgressBar()],
        check_val_every_n_epoch=config["evaluation"]["freq_epoch"],
        num_sanity_val_steps=0,
        default_root_dir=config["default_root_dir"],
        max_epochs=config["num_epochs"],
        profiler="simple",
        log_every_n_steps=10,
        **optional_kw_args,
    )

    Log.info("Start adaptive object-level multimodal diffusion training...")
    trainer.fit(model=model, train_dataloaders=loader, val_dataloaders=val_loader)
