import json
import math
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DepthDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_splits_path: str,
        split: str,
        rgb_image_size: Tuple[int, int],
        floorplan_img_size: Tuple[int, int],
        dataset_type: str = 's3d',
        default_map_resolution: float = None,
        default_fwidth: float = None,
    ):
        self.data_folder = data_folder
        self.data_splits_path = data_splits_path
        self.split = split
        self.rgb_image_size = tuple(rgb_image_size) if rgb_image_size is not None else None
        self.floorplan_img_size = tuple(floorplan_img_size) if floorplan_img_size is not None else None
        self.dataset_type = str(dataset_type).lower()
        self.default_map_resolution = float(default_map_resolution) if default_map_resolution is not None else None
        self.default_fwidth = float(default_fwidth) if default_fwidth is not None else None
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        with open(self.data_splits_path, 'r', encoding='utf-8') as f:
            data_splits = yaml.safe_load(f) or {}
        self.data_split = [''.join(x.split()) for x in data_splits[self.split]]
        self.data = self._load_data(self.data_folder, self.data_split)

    def _scene_map_size(self, map_path):
        with Image.open(map_path) as map_img:
            return map_img.size

    def _load_meta(self, scene_dir):
        meta_path = os.path.join(scene_dir, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _sort_key(self, path: Path):
        stem = path.stem
        if self.dataset_type == 'gibson' and '-' in stem:
            major, minor = stem.split('-', 1)
            return (int(major), int(minor))
        return (int(stem), 0)

    def _default_fwidth_value(self):
        if self.default_fwidth is not None:
            return self.default_fwidth
        return 1.0 / math.tan(0.698132) / 2.0

    def _load_scene_records(self, scene_dir):
        meta = self._load_meta(scene_dir)
        floorplan_path = os.path.join(scene_dir, 'map.png')
        if self.dataset_type == 'gibson':
            map_resolution = self.default_map_resolution if self.default_map_resolution is not None else 0.01
            pose_path = os.path.join(scene_dir, 'poses.txt')
            rgb_dir = os.path.join(scene_dir, 'rgb')
        elif self.dataset_type in {'palms', 's3d'}:
            map_resolution = float(meta.get('map_resolution', self.default_map_resolution if self.default_map_resolution is not None else 0.02))
            pose_path = os.path.join(scene_dir, 'poses_map.txt')
            rgb_dir = os.path.join(scene_dir, 'imgs')
        else:
            raise ValueError(f'Unsupported dataset_type: {self.dataset_type}')

        pose_lines = Path(pose_path).read_text(encoding='utf-8').splitlines()
        ray_lines = Path(os.path.join(scene_dir, 'depth40.txt')).read_text(encoding='utf-8').splitlines()
        pose_data = [list(map(float, line.split()))[:3] for line in pose_lines if line.strip()]
        ray_data = [list(map(float, line.split())) for line in ray_lines if line.strip()]

        if self.dataset_type == 'gibson':
            map_w, map_h = self._scene_map_size(floorplan_path)
            for pose in pose_data:
                pose[0] = pose[0] / map_resolution + map_w / 2.0
                pose[1] = pose[1] / map_resolution + map_h / 2.0
            fwidth_data = [self._default_fwidth_value()] * len(ray_data)
        else:
            fwidth_path = os.path.join(scene_dir, 'fwidths.txt')
            if os.path.exists(fwidth_path):
                fwidth_data = [float(line.strip()) for line in Path(fwidth_path).read_text(encoding='utf-8').splitlines() if line.strip()]
            else:
                fwidth_data = [self._default_fwidth_value()] * len(ray_data)

        image_files = sorted(
            [p for p in Path(rgb_dir).iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}],
            key=self._sort_key,
        )
        if not (len(image_files) == len(ray_data) == len(pose_data)):
            raise RuntimeError(
                f'Scene {scene_dir} has inconsistent counts: images={len(image_files)} rays={len(ray_data)} poses={len(pose_data)}'
            )

        scene_name = os.path.basename(scene_dir)
        records = []
        for idx, image_path in enumerate(image_files):
            records.append({
                'rgb_image': str(image_path),
                'floorplan_image': floorplan_path,
                'floorplan_path': floorplan_path,
                'pose': pose_data[idx],
                'ray': ray_data[idx],
                'fwidth': fwidth_data[idx],
                'map_resolution': map_resolution,
                'scene_name': scene_name,
                'frame_name': image_path.name,
            })
        return records

    def _load_data(self, data_folder, data_split):
        data = []
        missing = []
        for scene_name in data_split:
            scene_dir = os.path.join(data_folder, scene_name)
            if not os.path.exists(scene_dir):
                missing.append(scene_name)
                continue
            data.extend(self._load_scene_records(scene_dir))
        if missing:
            preview = ', '.join(missing[:5])
            suffix = '' if len(missing) <= 5 else ', ...'
            print(f'[WARN] Missing {len(missing)} scenes under {data_folder}: {preview}{suffix}')
        return data

    def _load_floorplan(self, floorplan_path):
        floorplan_img = cv2.imread(floorplan_path, cv2.IMREAD_GRAYSCALE)
        if floorplan_img is None:
            raise RuntimeError(f'Failed to load floorplan: {floorplan_path}')
        if self.floorplan_img_size is not None:
            floorplan_img = cv2.resize(floorplan_img, self.floorplan_img_size, interpolation=cv2.INTER_NEAREST)
        floorplan_img = floorplan_img.astype(np.float32) / 255.0
        floorplan_img = np.expand_dims(floorplan_img, axis=0)
        return floorplan_img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        data = self.data[i]
        with Image.open(data['rgb_image']) as rgb_pil:
            rgb_pil = rgb_pil.convert('RGB')
            if self.rgb_image_size is not None:
                rgb_pil = rgb_pil.resize(self.rgb_image_size, Image.BILINEAR)
            rgb_image = self.rgb_transform(rgb_pil)

        pose = torch.tensor(data['pose'], dtype=torch.float32)
        ray = torch.tensor(data['ray'], dtype=torch.float32)
        fwidth = torch.tensor(float(data['fwidth']), dtype=torch.float32)
        map_resolution = torch.tensor(float(data['map_resolution']), dtype=torch.float32)

        with Image.open(data['floorplan_image']) as floorplan_img:
            w, h = floorplan_img.size
        wh_tensor = torch.tensor([w, h], dtype=torch.float32)
        floorplan_tensor = self._load_floorplan(data['floorplan_image'])

        return (
            rgb_image,
            pose,
            ray,
            torch.as_tensor(floorplan_tensor, dtype=torch.float32),
            wh_tensor,
            fwidth,
            map_resolution,
            data['floorplan_path'],
        )
