import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW

from lego_floc.config import normalize_config
from lego_floc.depth_model import DepthPredModel
from lego_floc.evaluator import evaluate_dataset


class DepthLightningModule(pl.LightningModule):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = kwargs
        elif kwargs:
            merged = dict(config)
            merged.update(kwargs)
            config = merged
        self.config = normalize_config(config)
        self.save_hyperparameters(self.config)
        self.localization_eval_cfg = self.config.get('localization_eval', {})
        self.model = DepthPredModel(self.config)

    def forward(self, func_name, **kwargs):
        return self.model(func_name, **kwargs)

    def training_step(self, batch, batch_idx):
        obs, pose, ray, floorplan, wh, fwidth, map_resolution, floorplan_path = batch
        output = self.model('decoder_train', depth_cond=self.model('encode', obs_img=obs), gt_ray=ray)
        loss = output['loss']
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        obs, pose, ray, floorplan, wh, fwidth, map_resolution, floorplan_path = batch
        pred = self.model('decoder_inference', depth_cond=self.model('encode', obs_img=obs))
        val_loss = F.mse_loss(pred, ray)
        val_l1 = F.l1_loss(pred, ray)
        self.log('val_depth_loss', val_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_depth_l1', val_l1, on_epoch=True, logger=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=float(self.config['lr']))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_depth_loss'}}

    def on_validation_epoch_end(self):
        cfg = self.localization_eval_cfg
        if not cfg.get('enabled', False) or self.trainer is None or self.trainer.sanity_checking:
            return
        if (self.current_epoch + 1) % max(int(cfg.get('eval_freq', 1)), 1) != 0:
            return
        if not self.trainer.is_global_zero:
            return
        was_training = self.model.training
        self.model.eval()
        try:
            results = evaluate_dataset(
                dataset_type=cfg.get('dataset_type', self.config['dataset']['type']),
                dataset_path=cfg['dataset_path'],
                desdf_path=cfg['desdf_path'],
                model=self.model,
                device=self.device,
                split_file=cfg.get('split_file'),
                split_key=cfg.get('split_key', 'val'),
                mode=cfg.get('mode', 'frame'),
                ray_v=cfg.get('ray_v'),
                default_map_resolution=cfg.get('default_map_resolution'),
                default_fwidth=cfg.get('default_fwidth'),
                rgb_image_size=self.config['datasets'].get('rgb_img_size'),
            )
            metrics = {}
            if 'frame' in results:
                frame = results['frame']
                metrics.update({'val_frame_1m_recall': frame['recall_1m'], 'val_frame_0_5m_recall': frame['recall_0_5m'], 'val_frame_0_1m_recall': frame['recall_0_1m'], 'val_frame_1m_30deg_recall': frame['recall_1m_30deg']})
            if 'session360' in results:
                sess = results['session360']
                metrics.update({'val_session360_1m_recall': sess['recall_1m'], 'val_session360_0_5m_recall': sess['recall_0_5m'], 'val_session360_0_1m_recall': sess['recall_0_1m'], 'val_session360_1m_30deg_recall': sess['recall_1m_30deg']})
            for key, value in metrics.items():
                self.log(key, float(value), on_epoch=True, logger=True, prog_bar=key.endswith('1m_recall'))
            print('[localization_eval]', {k: round(float(v), 4) for k, v in metrics.items()})
        finally:
            self.model.train(was_training)
