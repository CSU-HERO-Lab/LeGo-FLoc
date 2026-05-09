#!/usr/bin/env python3
import argparse
import os
import sys
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lego_floc.config import load_config
from lego_floc.datamodule import DepthDataModule
from lego_floc.lightning_module import DepthLightningModule


def build_logger(config, project_folder):
    if config.get('use_wandb', False):
        return WandbLogger(project=config.get('project_name', 'lego_floc'), name=config['run_name'], config=config)
    return TensorBoardLogger(save_dir=project_folder, name='lightning_logs')


def build_checkpoint_callback(config):
    eval_cfg = config.get('localization_eval', {})
    every_n_epochs = None
    if eval_cfg.get('enabled', False):
        monitor = eval_cfg.get('monitor_metric') or ('val_session360_1m_recall' if eval_cfg.get('mode') in {'both', 'session360'} else 'val_frame_1m_recall')
        mode = eval_cfg.get('monitor_mode', 'max')
        every_n_epochs = max(int(eval_cfg.get('eval_freq', 1)), 1)
    else:
        monitor = 'val_depth_loss'
        mode = 'min'
    return ModelCheckpoint(dirpath=os.path.join(config['project_folder'], 'checkpoints'), filename='{epoch:02d}-{' + monitor + ':.4f}', auto_insert_metric_name=False, save_top_k=3, save_last=True, monitor=monitor, mode=mode, every_n_epochs=every_n_epochs)


def main():
    parser = argparse.ArgumentParser(description='Train LeGo-FLoc depth model')
    parser.add_argument('--config', '-c', required=True)
    parser.add_argument('--fast-dev-run', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.fast_dev_run:
        config['localization_eval']['enabled'] = False
    run_name = config.get('run_name', 'lego_floc') + '_' + time.strftime('%Y%m%d_%H%M%S')
    config['run_name'] = run_name
    config['project_folder'] = os.path.join('logs', 'runs', run_name)
    os.makedirs(config['project_folder'], exist_ok=True)
    data_module = DepthDataModule(config['datasets'], config['batch_size'], config['num_workers'], config.get('eval_batch_size'))
    model = DepthLightningModule(config=config)
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=config.get('gpu_ids', [0]) if torch.cuda.is_available() else 1, max_epochs=int(config['epochs']), callbacks=[build_checkpoint_callback(config)], logger=build_logger(config, config['project_folder']), default_root_dir=config['project_folder'], log_every_n_steps=int(config.get('wandb_log_freq', 10)), fast_dev_run=args.fast_dev_run)
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()
