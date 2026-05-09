#!/usr/bin/env python3
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lego_floc.config import load_config
from lego_floc.evaluator import evaluate_dataset, print_summary
from lego_floc.lightning_module import DepthLightningModule

DATASET_TYPE = "palms"
DEFAULT_MODE = "both"


def main():
    parser = argparse.ArgumentParser(description='Evaluate LeGo-FLoc on ' + DATASET_TYPE)
    parser.add_argument('--config', '-c', required=True)
    parser.add_argument('--checkpoint', '-k', required=True)
    parser.add_argument('--split-key', default=None)
    parser.add_argument('--mode', default=None, choices=['frame', 'session360', 'both'])
    args = parser.parse_args()
    config = load_config(args.config)
    if config['dataset']['type'] != DATASET_TYPE:
        raise ValueError('Config dataset.type={} does not match {}'.format(config['dataset']['type'], DATASET_TYPE))
    eval_cfg = dict(config['localization_eval'])
    eval_cfg['mode'] = args.mode or eval_cfg.get('mode', DEFAULT_MODE)
    if args.split_key is not None:
        eval_cfg['split_key'] = args.split_key
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    module = DepthLightningModule.load_from_checkpoint(args.checkpoint, map_location=device)
    model = module.model.to(device).eval()
    results = evaluate_dataset(dataset_type=DATASET_TYPE, dataset_path=eval_cfg['dataset_path'], desdf_path=eval_cfg['desdf_path'], model=model, device=device, split_file=eval_cfg.get('split_file'), split_key=eval_cfg.get('split_key', 'test'), mode=eval_cfg.get('mode', DEFAULT_MODE), ray_v=eval_cfg.get('ray_v'), default_map_resolution=eval_cfg.get('default_map_resolution'), default_fwidth=eval_cfg.get('default_fwidth'), rgb_image_size=config['datasets'].get('rgb_img_size'))
    for key, metrics in results.items():
        print_summary(key, metrics)


if __name__ == '__main__':
    main()
