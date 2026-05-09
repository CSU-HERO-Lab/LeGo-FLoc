import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from lego_floc.backbones import ResNet50FeatureExtractor, ResNet50RSKFeatureExtractor
from lego_floc.config import normalize_config
from lego_floc.decoder import DepthMlpDecoder
from lego_floc.fusion import FrozenDualExpertFusion


class DepthPredModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = normalize_config(config)
        self.encoder_type = self.config['encoder_type']
        self._init_encoders()
        self._init_decoders()

    def forward(self, func_name, **kwargs):
        if func_name == 'encode':
            return self._encode(**kwargs)
        if func_name == 'decoder_train':
            return self._decoder_train(kwargs['depth_cond'], kwargs['gt_ray'])
        if func_name == 'decoder_inference':
            return self._decoder_inference(kwargs['depth_cond'])
        raise NotImplementedError(func_name)

    def _encode(self, obs_img):
        if self.encoder_type in {'res50_3D', 'res50_RSK'}:
            features, _, _ = self.res50_encoder(obs_img=obs_img)
            return features
        if self.encoder_type == 'fusion_3dp_rsk':
            return obs_img
        raise ValueError(f'Unsupported encoder_type: {self.encoder_type}')

    def _decoder_train(self, cond, gt_ray):
        if self.encoder_type == 'fusion_3dp_rsk':
            out = self.fusion_model(cond)
            pred = out['pred']
            loss = F.l1_loss(pred, gt_ray)
            shape_weight = self.config.get('fusion_shape_loss_weight')
            if shape_weight is not None:
                shape_loss = float(shape_weight) * (1 - F.cosine_similarity(pred, gt_ray, dim=-1).mean())
                out['shape_loss'] = shape_loss
                loss = loss + shape_loss
            out['loss'] = loss
            return out
        pred = self.decoder(cond)
        return {'pred': pred, 'loss': F.l1_loss(pred, gt_ray)}

    def _decoder_inference(self, cond):
        if self.encoder_type == 'fusion_3dp_rsk':
            return self.fusion_model(cond)['pred']
        return self.decoder(cond)

    @staticmethod
    def _strip_model_prefix(state_dict):
        return {key[len('model.'):]: value for key, value in state_dict.items() if key.startswith('model.')}

    def _load_frozen_expert(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Expert checkpoint not found: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        expert_cfg = ckpt.get('hyper_parameters')
        if not isinstance(expert_cfg, dict):
            raise ValueError(f'Checkpoint {ckpt_path} has no hyper_parameters config')
        expert = DepthPredModel(expert_cfg)
        state_dict = self._strip_model_prefix(ckpt.get('state_dict', ckpt))
        missing, unexpected = expert.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(f'Failed to load expert {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}')
        expert.requires_grad_(False)
        expert.eval()
        return expert

    def _init_encoders(self):
        target_width = int(self.config.get('encoder_target_width', 40))
        embed_dim = int(self.config.get('encoding_size', 128))
        if self.encoder_type == 'res50_3D':
            path = self.config['resnet50_3d_ckpt_path']
            if not os.path.exists(path):
                raise FileNotFoundError(f'pretrained_3dp_path not found: {path}')
            self.res50_encoder = ResNet50FeatureExtractor(
                embed_dim=embed_dim,
                target_width=target_width,
                pretrained=bool(self.config.get('resnet50_pretrained', True)),
                freeze_pretrained=bool(self.config.get('freeze_pretrained_encoder', False)),
                checkpoint_path=path,
                checkpoint_variant='3dp',
            )
        elif self.encoder_type == 'res50_RSK':
            path = self.config['resnet50_rsk_ckpt_path']
            if not os.path.exists(path):
                raise FileNotFoundError(f'pretrained_rsk_path not found: {path}')
            self.res50_encoder = ResNet50RSKFeatureExtractor(
                embed_dim=embed_dim,
                target_width=target_width,
                checkpoint_path=path,
                freeze_pretrained=bool(self.config.get('freeze_pretrained_encoder', False)),
            )
        elif self.encoder_type == 'fusion_3dp_rsk':
            expert_3dp = self._load_frozen_expert(self.config['expert_3dp_ckpt_path'])
            expert_rsk = self._load_frozen_expert(self.config['expert_rsk_ckpt_path'])
            self.fusion_model = FrozenDualExpertFusion(expert_3dp, expert_rsk, selector_hidden_dim=int(self.config.get('fusion_selector_hidden_dim', 64)))
        else:
            raise ValueError(f'Unsupported encoder_type: {self.encoder_type}')

    def _init_decoders(self):
        if self.encoder_type == 'fusion_3dp_rsk':
            return
        self.decoder = DepthMlpDecoder(
            d_min=float(self.config.get('d_min', 0.1)),
            d_max=float(self.config.get('d_max', 20.0)),
            d_hyp=float(self.config.get('d_hyp', -0.2)),
            D=int(self.config.get('D', 128)),
            input_dim=int(self.config.get('encoding_size', 128)),
        )
