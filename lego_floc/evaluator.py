import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms

from lego_floc.localization import get_ray_from_depth, localize
from lego_floc.palms360 import build_descriptor_360, depth_profiles_for_map_matching, localize_360, parse_partial_uniform_policy, select_session360_frame_data, wrap_to_pi

_NORMALIZER = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
S3D_FWIDTH = 1.0 / math.tan(0.698132) / 2.0


def load_image_tensor(image_path, rgb_image_size=None):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'Failed to load image: {image_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if rgb_image_size is not None:
        img = cv2.resize(img, (int(rgb_image_size[0]), int(rgb_image_size[1])), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return _NORMALIZER(tensor)


def predict_depth40(model, image_tensor, device):
    with torch.no_grad():
        obs = image_tensor.unsqueeze(0).to(device)
        features = model('encode', obs_img=obs)
        pred = model('decoder_inference', depth_cond=features)
    return pred.squeeze(0).detach().cpu().numpy().astype(np.float32)


def iter_scene_images(scene_dir, dataset_type):
    image_dir = Path(scene_dir) / ('rgb' if dataset_type == 'gibson' else 'imgs')
    files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
    if dataset_type == 'gibson':
        def sort_key(path):
            if '-' in path.stem:
                major, minor = path.stem.split('-', 1)
                return (int(major), int(minor))
            return (int(path.stem), 0)
    else:
        def sort_key(path):
            return (int(path.stem), 0)
    return sorted(files, key=sort_key)


def _load_meta(scene_dir, dataset_type, default_map_resolution, default_fwidth):
    if dataset_type == 'palms':
        with open(os.path.join(scene_dir, 'meta.json'), 'r', encoding='utf-8') as f:
            meta = json.load(f)
        meta.setdefault('map_resolution', default_map_resolution if default_map_resolution is not None else 0.02)
        return meta
    if dataset_type == 's3d':
        return {'map_resolution': float(default_map_resolution if default_map_resolution is not None else 0.02), 'desdf_resolution': 0.02}
    if dataset_type == 'gibson':
        return {'map_resolution': float(default_map_resolution if default_map_resolution is not None else 0.01), 'desdf_resolution': 0.1, 'default_fwidth': float(default_fwidth if default_fwidth is not None else 3.0 / 8.0)}
    raise ValueError(f'Unsupported dataset_type: {dataset_type}')


def _load_poses(scene_dir, dataset_type, map_resolution):
    if dataset_type == 'gibson':
        occ = cv2.imread(os.path.join(scene_dir, 'map.png'))[:, :, 0]
        h, w = occ.shape
        lines = [line.strip() for line in Path(os.path.join(scene_dir, 'poses.txt')).read_text(encoding='utf-8').splitlines() if line.strip()]
        poses = np.zeros((len(lines), 3), dtype=np.float32)
        for idx, line in enumerate(lines):
            x_world, y_world, th = map(float, line.split()[:3])
            poses[idx, 0] = x_world / map_resolution + w / 2.0
            poses[idx, 1] = y_world / map_resolution + h / 2.0
            poses[idx, 2] = th
        return poses
    return np.atleast_2d(np.loadtxt(os.path.join(scene_dir, 'poses_map.txt'), dtype=np.float32))[:, :3]


def _frame_eval_params(dataset_type, meta):
    if dataset_type == 'gibson':
        return 10.0, 0.1
    if dataset_type == 's3d':
        return 5.0, 0.1
    stride = float(meta['desdf_resolution']) / float(meta['map_resolution'])
    return stride, float(meta['desdf_resolution'])


def evaluate_scene_frame(scene_dir, desdf_item, meta, model, device, dataset_type, ray_v, default_fwidth=None, rgb_image_size=None):
    poses = _load_poses(scene_dir, dataset_type, float(meta['map_resolution']))
    if dataset_type == 'gibson':
        fwidths = np.full((poses.shape[0],), float(default_fwidth if default_fwidth is not None else meta.get('default_fwidth', 3.0 / 8.0)), dtype=np.float32)
    else:
        fwidth_path = os.path.join(scene_dir, 'fwidths.txt')
        if os.path.exists(fwidth_path):
            fwidths = np.atleast_1d(np.loadtxt(fwidth_path, dtype=np.float32))
        else:
            fwidths = np.full((poses.shape[0],), float(default_fwidth if default_fwidth is not None else S3D_FWIDTH), dtype=np.float32)
    image_files = iter_scene_images(scene_dir, dataset_type)
    stride, translation_resolution = _frame_eval_params(dataset_type, meta)
    desdf_t = torch.as_tensor(desdf_item['desdf'], dtype=torch.float32, device=device)
    loc_errors, orn_errors, pred_depths = [], [], []
    for idx, image_path in enumerate(image_files):
        pred_depth = predict_depth40(model, load_image_tensor(image_path, rgb_image_size), device)
        pred_depths.append(pred_depth)
        gt = poses[idx, :3].copy()
        gt[0] = (gt[0] - desdf_item['l']) / stride
        gt[1] = (gt[1] - desdf_item['t']) / stride
        profile = depth_profiles_for_map_matching(pred_depth, meta)
        rays = get_ray_from_depth(profile, V=int(ray_v), F_W=float(fwidths[idx]))
        _, _, _, pred = localize(desdf_t, torch.as_tensor(rays, device=device), return_np=False)
        pred_np = pred.detach().cpu().numpy()
        loc_errors.append(float(np.linalg.norm(pred_np[:2] - gt[:2]) * translation_resolution))
        orn_errors.append(float(abs(wrap_to_pi(pred_np[2] - gt[2])) / math.pi * 180.0))
    return np.asarray(loc_errors, dtype=np.float32), np.asarray(orn_errors, dtype=np.float32), pred_depths


def evaluate_scene_session360(scene_dir, desdf_item, meta, pred_depths, frame_policy='all', subset_fallback='all'):
    fwidths = np.atleast_1d(np.loadtxt(os.path.join(scene_dir, 'fwidths.txt'), dtype=np.float32))
    local_yaws = np.atleast_1d(np.loadtxt(os.path.join(scene_dir, 'local_yaws.txt'), dtype=np.float32))
    center_offset = float(meta.get('session360_center_offset', math.pi))
    centers = np.array([wrap_to_pi(float(v) + center_offset) for v in local_yaws], dtype=np.float32)
    pred_depths = depth_profiles_for_map_matching(np.asarray(pred_depths, dtype=np.float32), meta)
    pred_depths, centers, fwidths, _, subset_info = select_session360_frame_data(pred_depths, centers, fwidths, frame_policy=frame_policy, subset_fallback=subset_fallback)
    fill_missing = parse_partial_uniform_policy(frame_policy) is None
    descriptor = build_descriptor_360(pred_depths, centers, fwidths, fill_missing=fill_missing)
    pred = localize_360(desdf_item['desdf'], descriptor, flip_query=False)
    label_pose = np.asarray(meta['label_pose_map'], dtype=np.float32).copy()
    stride = float(meta['desdf_resolution']) / float(meta['map_resolution'])
    gt = label_pose.copy()
    gt[0] = (gt[0] - desdf_item['l']) / stride
    gt[1] = (gt[1] - desdf_item['t']) / stride
    loc_error = float(np.linalg.norm(pred[:2] - gt[:2]) * meta['desdf_resolution'])
    orn_error = float(abs(wrap_to_pi(pred[2] - gt[2])) / math.pi * 180.0)
    return pred, gt, loc_error, orn_error, subset_info


def summarize_errors(loc_errors, orn_errors, count_key):
    if loc_errors.size == 0:
        return {count_key: 0, 'recall_0_1m': float('nan'), 'recall_1m': float('nan'), 'recall_0_5m': float('nan'), 'loc_error_mean': float('nan'), 'loc_error_median': float('nan'), 'recall_1m_30deg': float('nan')}
    return {count_key: int(loc_errors.size), 'recall_0_1m': float(np.mean(loc_errors < 0.1)), 'recall_1m': float(np.mean(loc_errors < 1.0)), 'recall_0_5m': float(np.mean(loc_errors < 0.5)), 'loc_error_mean': float(loc_errors.mean()), 'loc_error_median': float(np.median(loc_errors)), 'recall_1m_30deg': float(np.mean(np.logical_and(loc_errors < 1.0, orn_errors < 30.0)))}


def evaluate_dataset(dataset_type, dataset_path, desdf_path, model, device, split_file=None, split_key='test', mode='frame', ray_v=None, default_map_resolution=None, default_fwidth=None, rgb_image_size=None, session360_frame_policy='all', session360_subset_fallback='all'):
    dataset_type = str(dataset_type).lower()
    split_file = split_file or os.path.join(dataset_path, 'split.yaml')
    with open(split_file, 'r', encoding='utf-8') as f:
        scenes = (yaml.safe_load(f) or {})[split_key]
    if dataset_type in {'s3d', 'gibson'}:
        mode = 'frame'
    if ray_v is None:
        ray_v = 11 if dataset_type == 'gibson' else 9
    frame_loc_all, frame_orn_all = [], []
    sess_loc, sess_orn = [], []
    for scene in scenes:
        scene_dir = os.path.join(dataset_path, scene)
        desdf_item = np.load(os.path.join(desdf_path, scene, 'desdf.npy'), allow_pickle=True).item()
        desdf_item['desdf'][desdf_item['desdf'] > 20] = 20
        meta = _load_meta(scene_dir, dataset_type, default_map_resolution, default_fwidth)
        frame_loc, frame_orn, pred_depths = evaluate_scene_frame(scene_dir, desdf_item, meta, model, device, dataset_type, ray_v, default_fwidth, rgb_image_size)
        frame_loc_all.append(frame_loc)
        frame_orn_all.append(frame_orn)
        if dataset_type == 'palms' and mode in {'session360', 'both'}:
            _, _, loc, orn, _ = evaluate_scene_session360(scene_dir, desdf_item, meta, pred_depths, session360_frame_policy, session360_subset_fallback)
            sess_loc.append(loc)
            sess_orn.append(orn)
    results = {}
    frame_loc_all = np.concatenate(frame_loc_all) if frame_loc_all else np.zeros((0,), dtype=np.float32)
    frame_orn_all = np.concatenate(frame_orn_all) if frame_orn_all else np.zeros((0,), dtype=np.float32)
    results['frame'] = summarize_errors(frame_loc_all, frame_orn_all, 'samples')
    if dataset_type == 'palms' and mode in {'session360', 'both'}:
        results['session360'] = summarize_errors(np.asarray(sess_loc, dtype=np.float32), np.asarray(sess_orn, dtype=np.float32), 'scenes')
    return results


def print_summary(mode, metrics):
    print('==============================')
    print(f'Mode: {mode}')
    count_key = 'samples' if mode == 'frame' else 'scenes'
    print(f'{count_key}: {metrics[count_key]}')
    print(f"0.1m recall = {metrics['recall_0_1m']:.4f}")
    print(f"0.5m recall = {metrics['recall_0_5m']:.4f}")
    print(f"1m recall = {metrics['recall_1m']:.4f}")
    print(f"1m 30deg recall = {metrics['recall_1m_30deg']:.4f}")
    print(f"mean loc error = {metrics['loc_error_mean']:.4f}")
    print(f"median loc error = {metrics['loc_error_median']:.4f}")
