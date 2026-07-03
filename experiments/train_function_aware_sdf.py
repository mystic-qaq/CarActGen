import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())

from model.FunctionAware import FunctionAwareGenSDFDataset, FunctionAwareSDFAutoEncoder
from utils import parse_config_from_args
from utils.mylogging import Log


os.environ["WANDB_CACHE_DIR"] = (Path() / "wandb/cache").resolve().as_posix()
os.environ["WANDB_DATA_DIR"] = (Path() / "wandb/data").resolve().as_posix()


def load_shape_split(d_configs: dict):
    split_path = d_configs.get("split_path")
    if not split_path:
        return None, None
    split = json.loads(Path(split_path).read_text())
    train_key = d_configs.get("train_split", "train")
    val_key = d_configs.get("val_split", "val")
    train_ids = split.get(train_key, [])
    val_ids = split.get(val_key, [])
    if not train_ids:
        raise ValueError(f"No train shape ids found in {split_path} under key {train_key!r}")
    if not val_ids:
        raise ValueError(f"No validation shape ids found in {split_path} under key {val_key!r}")
    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise ValueError(f"Train/validation shape splits overlap: {overlap[:10]}")
    return train_ids, val_ids


def build_checkpoint_callback(config: dict, run_name: str):
    checkpoint_config = config["checkpoint"]
    kwargs = {
        "save_top_k": checkpoint_config.get("save_top_k", -1),
        "save_last": checkpoint_config.get("save_last", True),
        "dirpath": checkpoint_config["path"] + "/" + run_name,
        "filename": checkpoint_config.get("filename", "function_sdf_{epoch:04d}-{loss:.5f}"),
    }
    if checkpoint_config.get("monitor"):
        kwargs["monitor"] = checkpoint_config["monitor"]
        kwargs["mode"] = checkpoint_config.get("mode", "min")
        kwargs["filename"] = checkpoint_config.get("filename", "function_sdf_{epoch:04d}-{val_loss:.5f}")
    if checkpoint_config.get("freq_epoch"):
        kwargs["every_n_epochs"] = checkpoint_config["freq_epoch"]
    else:
        kwargs["every_n_train_steps"] = checkpoint_config["freq"]
    return ModelCheckpoint(**kwargs)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    run_name = time.strftime("%m-%d-%I%p-%M-%S")
    config = parse_config_from_args()
    seed_everything(config["seed"])

    optional_kw_args = {}
    if config["wandb"].get("use", True):
        optional_kw_args["logger"] = WandbLogger(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            name=run_name,
            log_model=False,
        )

    if config.get("initialize_from_function_aware_sdf"):
        model = FunctionAwareSDFAutoEncoder.load_from_checkpoint(
            config["initialize_from_function_aware_sdf"],
            configs=config,
            map_location="cpu",
            strict=not bool(config.get("allow_partial_checkpoint", False)),
        )
        Log.info("Initialized function-aware SDF from %s", config["initialize_from_function_aware_sdf"])
    elif config.get("initialize_from_sdf"):
        model = FunctionAwareSDFAutoEncoder(config)
        model.initialize_from_sdf_checkpoint(config["initialize_from_sdf"])
    else:
        model = FunctionAwareSDFAutoEncoder(config)

    d_configs = config["dataset_n_dataloader"]
    train_shape_ids, val_shape_ids = load_shape_split(d_configs)

    def make_dataset(is_train: bool):
        include_shape_ids = None
        if train_shape_ids is not None:
            include_shape_ids = train_shape_ids if is_train else val_shape_ids
        return FunctionAwareGenSDFDataset(
            dataset_dir=Path(d_configs["dataset_dir"]),
            train=None,
            samples_per_mesh=d_configs["samples_per_mesh"],
            pc_size=d_configs["pc_size"],
            uniform_sample_ratio=d_configs["uniform_sample_ratio"],
            mesh_info_dir=Path(d_configs["mesh_info_dir"]),
            sdf_subdir=d_configs.get("sdf_subdir", "2_gensdf_dataset"),
            uniform_sample_ratio_by_function=d_configs.get("uniform_sample_ratio_by_function"),
            sample_repeat_by_function=d_configs.get("sample_repeat_by_function") if is_train else d_configs.get("val_sample_repeat_by_function"),
            point_cloud_surface_ratio=d_configs.get("point_cloud_surface_ratio", 0.0),
            point_cloud_surface_ratio_by_function=d_configs.get("point_cloud_surface_ratio_by_function"),
            point_cloud_surface_abs_percentile=d_configs.get("point_cloud_surface_abs_percentile", 70.0),
            limit=d_configs.get("limit", -1) if is_train else d_configs.get("val_limit", -1),
            include_shape_ids=include_shape_ids,
        )

    dataloaders = [
        DataLoader(
            make_dataset(is_train),
            num_workers=d_configs["n_workers"],
            batch_size=d_configs["batch_size"],
            drop_last=is_train,
            shuffle=is_train,
            pin_memory=True,
            persistent_workers=d_configs["n_workers"] > 0,
        )
        for is_train in [True, False]
    ]

    checkpoint_callback = build_checkpoint_callback(config, run_name)

    if isinstance(config["devices"], list) or config["devices"] > 1:
        optional_kw_args["strategy"] = DDPStrategy(find_unused_parameters=True)

    trainer = Trainer(
        devices=config["devices"],
        accelerator=config["accelerator"],
        benchmark=True,
        callbacks=[ModelSummary(max_depth=1), checkpoint_callback, TQDMProgressBar()],
        check_val_every_n_epoch=config["evaluation"]["freq_epoch"],
        default_root_dir=config["default_root_dir"],
        max_epochs=config["num_epochs"],
        profiler="simple",
        log_every_n_steps=5,
        **optional_kw_args,
    )

    Log.info("Start function-aware SDF training...")
    trainer.fit(model=model, train_dataloaders=dataloaders[0], val_dataloaders=dataloaders[1])
