import time
import json
import torch

from torch.utils.data import DataLoader

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, ModelSummary
from lightning.pytorch.strategies import DDPStrategy

from utils.mylogging import Log
from utils import parse_config_from_args

from pathlib import Path
from model.SDFAutoEncoder import SDFAutoEncoder
from model.SDFAutoEncoder.dataloader import GenSDFDataset

import os
# Set WANDB_CACHE_DIR to a local directory to avoid no space left error
# cmd clean up the cache: wandb artifact cache cleanup 1GB
os.environ['WANDB_CACHE_DIR'] = (Path() / 'wandb/cache').resolve().as_posix()
os.environ['WANDB_DATA_DIR'] = (Path() / 'wandb/data').resolve().as_posix()


def load_shape_split(d_configs: dict):
    split_path = d_configs.get('split_path')
    if not split_path:
        return None, None
    split = json.loads(Path(split_path).read_text())
    train_key = d_configs.get('train_split', 'train')
    val_key = d_configs.get('val_split', 'val')
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

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    run_name = time.strftime("%m-%d-%I%p-%M-%S")
    config = parse_config_from_args()
    seed_everything(config['seed'])
    optional_kw_args = dict()
    if config.get('wandb', {}).get('use', True):
        optional_kw_args['logger'] = WandbLogger(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=run_name,
            log_model=False,
        )

    if config['pretrained_model']:
        model = SDFAutoEncoder.load_from_checkpoint(config['pretrained_model'], configs=config, map_location='cpu')
    else:
        model = SDFAutoEncoder(config)

    # Configure data module
    d_configs = config['dataset_n_dataloader']
    train_shape_ids, val_shape_ids = load_shape_split(d_configs)

    def make_dataset(is_train: bool):
        include_shape_ids = None
        if train_shape_ids is not None:
            include_shape_ids = train_shape_ids if is_train else val_shape_ids
        return GenSDFDataset(
            dataset_dir=Path(d_configs['dataset_dir']),
            train=None,
            samples_per_mesh=d_configs['samples_per_mesh'],
            pc_size=d_configs['pc_size'],
            uniform_sample_ratio=d_configs['uniform_sample_ratio'],
            sdf_subdir=d_configs.get('sdf_subdir', '2_gensdf_dataset'),
            include_shape_ids=include_shape_ids,
        )

    dataloader = [
               DataLoader(make_dataset(is_train),
                num_workers=d_configs['n_workers'], batch_size=d_configs['batch_size'],
                drop_last=is_train, shuffle=is_train, pin_memory=True, persistent_workers=True)
            for is_train in [True, False]
    ]


    # Configure save checkpoint callback
    checkpoint_kwargs = {
        'save_top_k': config['checkpoint'].get('save_top_k', -1),
        'save_last': True,
        'dirpath': config['checkpoint']['path'] + '/' + run_name,
        'filename': config['checkpoint'].get('filename', "sdf_{epoch:04d}-{loss:.5f}"),
    }
    if config['checkpoint'].get('monitor'):
        checkpoint_kwargs['monitor'] = config['checkpoint']['monitor']
        checkpoint_kwargs['mode'] = config['checkpoint'].get('mode', 'min')
        checkpoint_kwargs['filename'] = config['checkpoint'].get('filename', "sdf_{epoch:04d}-{val_loss:.5f}")
    if config['checkpoint'].get('freq_epoch'):
        checkpoint_kwargs['every_n_epochs'] = config['checkpoint']['freq_epoch']
    else:
        checkpoint_kwargs['every_n_train_steps'] = config['checkpoint']['freq']
    checkpoint_callback = ModelCheckpoint(**checkpoint_kwargs)

    # Configure trainer
    if isinstance(config['devices'], list) or config['devices'] > 1:
        optional_kw_args['strategy'] = DDPStrategy(find_unused_parameters=True)

    trainer = Trainer(devices=config['devices'], accelerator=config["accelerator"],
                      benchmark=True,
                      callbacks=[ModelSummary(max_depth=1), checkpoint_callback, TQDMProgressBar()],
                      check_val_every_n_epoch=config['evaluation']['freq_epoch'],
                      default_root_dir=config['default_root_dir'],
                      max_epochs=config['num_epochs'], profiler="simple",
                      log_every_n_steps=5,
                      **optional_kw_args)

    Log.info("Start training...")

    trainer.fit(model=model, train_dataloaders=dataloader[0],
                               val_dataloaders=dataloader[1])
