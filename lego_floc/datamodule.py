import pytorch_lightning as pl
from torch.utils.data import DataLoader
import yaml

from lego_floc.dataset import DepthDataset


class DepthDataModule(pl.LightningDataModule):
    def __init__(self, data_config: dict, batch_size: int, num_workers: int, eval_batch_size: int = None):
        super().__init__()
        self.data_config = data_config
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None

    def _resolve_val_split(self):
        requested_split = self.data_config.get('val_split', 'val')
        with open(self.data_config['data_splits'], 'r', encoding='utf-8') as f:
            available_splits = yaml.safe_load(f) or {}
        if requested_split in available_splits:
            return requested_split
        if 'test' in available_splits:
            print(f"[WARN] Requested val split '{requested_split}' not found; fallback to 'test'.")
            return 'test'
        raise ValueError(
            f"Validation split '{requested_split}' not found in {self.data_config['data_splits']}. "
            f"Available splits: {list(available_splits.keys())}"
        )

    def _build_dataset(self, split):
        return DepthDataset(
            data_folder=self.data_config['data_folder'],
            data_splits_path=self.data_config['data_splits'],
            split=split,
            rgb_image_size=self.data_config.get('rgb_img_size'),
            floorplan_img_size=self.data_config.get('floorplan_img_size'),
            dataset_type=self.data_config.get('dataset_type', 's3d'),
            default_map_resolution=self.data_config.get('default_map_resolution'),
            default_fwidth=self.data_config.get('default_fwidth'),
        )

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            val_split = self._resolve_val_split()
            self.train_dataset = self._build_dataset('train')
            self.val_dataset = self._build_dataset(val_split)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )
