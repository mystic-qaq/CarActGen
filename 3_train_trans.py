import time
import torch

from torch.utils.data import DataLoader

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, ModelSummary
from lightning.pytorch.strategies import DDPStrategy

from utils.mylogging import Log
from utils import parse_config_from_args

from pathlib import Path
from model.Transformer import TransDiffusionCombineModel
from model.Transformer.dataloader import TransDiffusionDataset

import os
# Set WANDB_CACHE_DIR to a local directory to avoid no space left error
# cmd to clean up the cache: `wandb artifact cache cleanup 1GB`
os.environ['WANDB_CACHE_DIR'] = (Path() / 'wandb/cache').resolve().as_posix()
os.environ['WANDB_DATA_DIR'] = (Path() / 'wandb/data').resolve().as_posix()

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    run_name = time.strftime("%m-%d-%I%p-%M-%S")
    config = parse_config_from_args()
    print(config['device'])
    use_wandb = config['wandb']['use']

    seed_everything(config['seed'])

    if use_wandb:
        wandb_logger = WandbLogger(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=run_name,
            log_model=False,
        )


    # Configure data module
    d_configs = config['dataset_n_dataloader']
    train_dataloader = DataLoader(TransDiffusionDataset(
                dataset_path=d_configs['dataset_path'],
                cut_off=d_configs['cut_off'],
                enc_data_fieldname=d_configs['enc_data_fieldname'],
                cache_data=d_configs.get('cache_data', True),
            ),
            num_workers=d_configs['n_workers'], batch_size=d_configs['batch_size'],
            drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True)

    # config['evaluation']['sdf_model_path'] = train_dataloader.dataset.get_best_sdf_ckpt_path()
    config['diffusion_model']['pretrained_model_path'] = train_dataloader.dataset.get_best_diffusion_ckpt_path()

    # Configure model
    if config['base_on_model'] is not None:
        Log.info('Using pretrained model: %s', config['base_on_model'])
        model = TransDiffusionCombineModel.load_from_checkpoint(config['base_on_model'], map_location='cpu')
    else:
        Log.info('not found `base_on_model`, train new model.')
        model = TransDiffusionCombineModel(config)

    # Configure save checkpoint callback
    checkpoint_callback = ModelCheckpoint(
            save_top_k=config['checkpoint'].get('save_top_k', 1),
            save_last=config['checkpoint'].get('save_last', True),
            every_n_train_steps=config['checkpoint']['freq'],
            dirpath=config['checkpoint']['path'] + '/' + run_name,
            filename="transformer-{epoch:04d}-{step:08d}",
        )

    # Configure trainer
    optional_kw_args = dict()
    if use_wandb:
        optional_kw_args['logger'] = wandb_logger
    if isinstance(config['devices'], list) or config['devices'] > 1:
        optional_kw_args['strategy'] = DDPStrategy(
            find_unused_parameters=config.get('find_unused_parameters', False)
        )
    trainer = Trainer(devices=config['devices'], accelerator=config["accelerator"],
                      benchmark=True,
                      callbacks=[ModelSummary(max_depth=1), checkpoint_callback, TQDMProgressBar()],
                      check_val_every_n_epoch=config['evaluation']['freq'],
                      log_every_n_steps=config.get('log_every_n_steps', 50),
                      num_sanity_val_steps=0,
                    #   val_check_interval=0.1,
                      default_root_dir=config['default_root_dir'],
                      max_epochs=config['num_epochs'], profiler="simple",
                      **optional_kw_args )
    Log.info("Start training...")

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=train_dataloader)
