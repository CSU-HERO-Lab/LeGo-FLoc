from copy import deepcopy
from pathlib import Path

import yaml

_METHOD_TO_ENCODER = {'3dp': 'res50_3D', 'rsk': 'res50_RSK', 'fusion': 'fusion_3dp_rsk'}
_METHOD_PUBLIC = {'3dp': '3DP', 'rsk': 'RSK', 'fusion': 'fusion'}


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return normalize_config(yaml.safe_load(f) or {})


def normalize_config(config):
    cfg = deepcopy(config)
    method = str(cfg.get('method', '')).lower()
    if method not in _METHOD_TO_ENCODER:
        raise ValueError('method must be one of: 3DP, RSK, fusion')
    cfg['method'] = _METHOD_PUBLIC[method]
    cfg['encoder_type'] = _METHOD_TO_ENCODER[method]
    cfg.setdefault('decoder_type', 'depth')

    dataset = deepcopy(cfg.get('dataset') or {})
    if not dataset:
        raise ValueError('config requires dataset section')
    dataset_type = str(dataset.get('type', '')).lower()
    if dataset_type not in {'gibson', 's3d', 'palms'}:
        raise ValueError('dataset.type must be one of: gibson, s3d, palms')
    cfg['dataset']['type'] = dataset_type
    splits = dataset.get('splits') or dataset.get('data_splits')
    if not splits:
        raise ValueError('dataset.splits is required')
    cfg['datasets'] = {
        'dataset_type': dataset_type,
        'data_folder': dataset['data_folder'],
        'data_splits': splits,
        'val_split': dataset.get('val_split', 'val'),
        'rgb_img_size': dataset.get('rgb_img_size', [640, 480]),
        'floorplan_img_size': dataset.get('floorplan_img_size', [256, 256]),
        'default_map_resolution': dataset.get('default_map_resolution'),
        'default_fwidth': dataset.get('default_fwidth'),
    }

    if method == '3dp':
        if not cfg.get('pretrained_3dp_path'):
            raise ValueError('method=3DP requires pretrained_3dp_path')
        cfg['resnet50_3d_ckpt_path'] = cfg['pretrained_3dp_path']
    elif method == 'rsk':
        if not cfg.get('pretrained_rsk_path'):
            raise ValueError('method=RSK requires pretrained_rsk_path')
        cfg['resnet50_rsk_ckpt_path'] = cfg['pretrained_rsk_path']
    else:
        if not cfg.get('expert_3dp_checkpoint'):
            raise ValueError('method=fusion requires expert_3dp_checkpoint')
        if not cfg.get('expert_rsk_checkpoint'):
            raise ValueError('method=fusion requires expert_rsk_checkpoint')
        cfg['expert_3dp_ckpt_path'] = cfg['expert_3dp_checkpoint']
        cfg['expert_rsk_ckpt_path'] = cfg['expert_rsk_checkpoint']

    eval_cfg = cfg.setdefault('localization_eval', {})
    eval_cfg.setdefault('enabled', True)
    eval_cfg.setdefault('dataset_type', dataset_type)
    eval_cfg.setdefault('dataset_path', dataset['data_folder'])
    eval_cfg.setdefault('split_file', splits)
    eval_cfg.setdefault('split_key', cfg['datasets']['val_split'])
    if dataset_type == 'gibson':
        eval_cfg.setdefault('mode', 'frame')
        eval_cfg.setdefault('ray_v', 11)
        eval_cfg.setdefault('default_fwidth', 3.0 / 8.0)
        eval_cfg.setdefault('default_map_resolution', 0.01)
    elif dataset_type == 's3d':
        eval_cfg.setdefault('mode', 'frame')
        eval_cfg.setdefault('ray_v', 9)
        eval_cfg.setdefault('default_map_resolution', 0.02)
    else:
        eval_cfg.setdefault('mode', 'both')
        eval_cfg.setdefault('ray_v', 9)
        eval_cfg.setdefault('default_map_resolution', 0.02)
    if not eval_cfg.get('desdf_path'):
        raise ValueError('localization_eval.desdf_path is required')

    cfg.setdefault('project_name', f'lego_floc_{dataset_type}_{method}')
    cfg.setdefault('run_name', cfg['project_name'])
    cfg.setdefault('batch_size', 16)
    cfg.setdefault('eval_batch_size', cfg['batch_size'])
    cfg.setdefault('num_workers', 4)
    cfg.setdefault('epochs', 20)
    cfg.setdefault('gpu_ids', [0])
    cfg.setdefault('lr', 1e-4)
    cfg.setdefault('image_log_freq', 500)
    cfg.setdefault('num_images_log', 4)
    cfg.setdefault('wandb_log_freq', 10)
    cfg.setdefault('use_wandb', False)
    cfg.setdefault('encoding_size', 128)
    cfg.setdefault('encoder_target_width', 40)
    cfg.setdefault('d_min', 0.1)
    cfg.setdefault('d_max', 20.0)
    cfg.setdefault('d_hyp', -0.2)
    cfg.setdefault('D', 128)
    cfg.setdefault('freeze_pretrained_encoder', False)
    cfg.setdefault('resnet50_pretrained', False)
    return cfg
