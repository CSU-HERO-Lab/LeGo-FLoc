import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW

from lego_floc.config import normalize_config
from lego_floc.evaluator import evaluate_dataset
from lego_floc.legacy_model import LegacyDepthNet, LegacyDualExpertFusion


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
        self.method = str(self.config['method']).lower()
        self._build_model()

    def _build_model(self):
        params = dict(
            d_min=float(self.config.get('d_min', 0.1)),
            d_max=float(self.config.get('d_max', 20.0)),
            d_hyp=float(self.config.get('d_hyp', -0.2)),
            D=int(self.config.get('D', 128)),
        )
        if self.method in {'3dp', 'rsk'}:
            self.encoder = LegacyDepthNet(
                method=self.method,
                pretrained_3dp_path=self.config.get('pretrained_3dp_path'),
                pretrained_rsk_path=self.config.get('pretrained_rsk_path'),
                **params,
            )
        elif self.method == 'fusion':
            expert_3dp = self._load_expert(self.config['expert_3dp_ckpt_path'], method='3dp', params=params)
            expert_rsk = self._load_expert(self.config['expert_rsk_ckpt_path'], method='rsk', params=params)
            self.comp_d_net = LegacyDualExpertFusion(
                mv_net=expert_rsk,
                mono_net=expert_3dp,
                selector_hidden_dim=int(self.config.get('fusion_selector_hidden_dim', 64)),
                **params,
            )
        else:
            raise ValueError(f'Unsupported method: {self.config["method"]}')

    def _load_expert(self, ckpt_path, method, params):
        expert = LegacyDepthNet(
            method=method,
            pretrained_3dp_path=self.config.get('pretrained_3dp_path'),
            pretrained_rsk_path=self.config.get('pretrained_rsk_path'),
            load_pretrained=False,
            **params,
        )
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        encoder_state = {}
        for key, value in state.items():
            if key.startswith('encoder.'):
                encoder_state[key[len('encoder.'):]] = value
        if not encoder_state:
            raise ValueError(f'Expert checkpoint has no encoder.* keys: {ckpt_path}')
        missing, unexpected = expert.load_state_dict(encoder_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f'Failed to load expert {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}')
        expert.requires_grad_(False)
        expert.eval()
        return expert

    @property
    def model(self):
        return self

    def forward(self, func_name, **kwargs):
        if func_name == 'encode':
            return kwargs['obs_img']
        if func_name == 'decoder_train':
            pred = self._predict(kwargs['depth_cond'])
            loss = F.l1_loss(pred, kwargs['gt_ray'])
            shape_weight = self.config.get('fusion_shape_loss_weight') if self.method == 'fusion' else None
            out = {'pred': pred, 'loss': loss}
            if shape_weight is not None:
                shape_loss = float(shape_weight) * (1 - F.cosine_similarity(pred, kwargs['gt_ray'], dim=-1).mean())
                out['shape_loss'] = shape_loss
                out['loss'] = loss + shape_loss
            return out
        if func_name == 'decoder_inference':
            return self._predict(kwargs['depth_cond'])
        raise NotImplementedError(func_name)

    def _predict(self, obs_img):
        if self.method in {'3dp', 'rsk'}:
            pred, _, _ = self.encoder(obs_img)
            return pred
        return self.comp_d_net(obs_img)['d_comp']

    def training_step(self, batch, batch_idx):
        obs, pose, ray, floorplan, wh, fwidth, map_resolution, floorplan_path = batch
        output = self('decoder_train', depth_cond=obs, gt_ray=ray)
        loss = output['loss']
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        obs, pose, ray, floorplan, wh, fwidth, map_resolution, floorplan_path = batch
        pred = self('decoder_inference', depth_cond=obs)
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
        was_training = self.training
        self.eval()
        try:
            results = evaluate_dataset(
                dataset_type=cfg.get('dataset_type', self.config['dataset']['type']),
                dataset_path=cfg['dataset_path'],
                desdf_path=cfg['desdf_path'],
                model=self,
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
            self.train(was_training)
