import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter



class Attention(nn.Module):
    def forward(self, query, key, value, attn_mask=None):
        scores = torch.einsum('nld,nsd->nls', query, key)
        if attn_mask is not None:
            scores[attn_mask] = -torch.inf
        dim = query.shape[2]
        weights = torch.softmax(scores / (dim ** 2), dim=2)
        out = torch.einsum('nsd,nls->nld', value, weights)
        return out, weights


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.convbn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.convbn(x)


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.convbn = ConvBn(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return F.relu(self.convbn(x))


class LegacyDepthNet(nn.Module):
    def __init__(self, method, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128, pretrained_3dp_path=None, pretrained_rsk_path=None, load_pretrained=True):
        super().__init__()
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.depth_feature = LegacyDepthFeature(method=method, pretrained_3dp_path=pretrained_3dp_path, pretrained_rsk_path=pretrained_rsk_path, load_pretrained=load_pretrained)

    def forward(self, x, mask=None):
        logits, attn = self.depth_feature(x, mask)
        d_vals = torch.linspace(self.d_min ** self.d_hyp, self.d_max ** self.d_hyp, self.D, device=logits.device) ** (1 / self.d_hyp)
        prob = F.softmax(logits, dim=-1)
        depth = torch.sum(prob * d_vals, dim=-1)
        return depth, attn, prob


class LegacyDepthFeature(nn.Module):
    def __init__(self, method, pretrained_3dp_path=None, pretrained_rsk_path=None, load_pretrained=True):
        super().__init__()
        method = str(method).lower()
        if method == '3dp':
            backbone = resnet50(pretrained=False, replace_stride_with_dilation=[False, False, True])
            if load_pretrained:
                if not pretrained_3dp_path or not os.path.exists(pretrained_3dp_path):
                    raise FileNotFoundError(f'pretrained_3dp_path not found: {pretrained_3dp_path}')
                params = torch.load(pretrained_3dp_path, map_location='cpu')
                cleaned = OrderedDict()
                for key, value in params.items():
                    key = key.replace('module.', '')
                    key = key.replace('.shortcut.weight', '.downsample.0.weight')
                    key = key.replace('.shortcut.norm.', '.downsample.1.')
                    cleaned[key] = value
                backbone.load_state_dict(cleaned, strict=False)
            self.resnet = nn.Sequential(IntermediateLayerGetter(backbone, return_layers={'layer4': 'feat'}))
        elif method == 'rsk':
            backbone = resnet50(pretrained=False, replace_stride_with_dilation=[False, False, True])
            if load_pretrained:
                if not pretrained_rsk_path or not os.path.exists(pretrained_rsk_path):
                    raise FileNotFoundError(f'pretrained_rsk_path not found: {pretrained_rsk_path}')
                params = torch.load(pretrained_rsk_path, map_location='cpu')
                if isinstance(params, dict) and 'state_dict' in params:
                    params = params['state_dict']
                cleaned = OrderedDict()
                prefix_map = {'0.': 'conv1.', '1.': 'bn1.', '4.': 'layer1.', '5.': 'layer2.', '6.': 'layer3.', '7.': 'layer4.'}
                target_shapes = backbone.state_dict()
                for key, value in params.items():
                    key = key.replace('module.base.', '').replace('base.', '')
                    mapped = None
                    for src_prefix, dst_prefix in prefix_map.items():
                        if key.startswith(src_prefix):
                            mapped = dst_prefix + key[len(src_prefix):]
                            break
                    if mapped is not None and mapped in target_shapes and target_shapes[mapped].shape == value.shape:
                        cleaned[mapped] = value
                missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
                print(f'[LegacyDepthFeature] loaded RSK backbone {pretrained_rsk_path} (matched={len(cleaned)}, missing={len(missing)}, unexpected={len(unexpected)})')
            self.resnet = nn.Sequential(IntermediateLayerGetter(backbone, return_layers={'layer4': 'feat'}))
        else:
            raise ValueError('LegacyDepthFeature only supports method=3DP or method=RSK')

        self.conv = ConvBnReLU(in_channels=2048, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.pos_mlp_2d = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.pos_mlp_1d = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.q_proj = nn.Linear(160, 128, bias=False)
        self.k_proj = nn.Linear(160, 128, bias=False)
        self.v_proj = nn.Linear(160, 128, bias=False)
        self.attn = Attention()

    def forward(self, x, mask=None):
        x = self.resnet(x)['feat']
        x = self.conv(x)
        feat_h, feat_w = list(x.shape[2:])
        batch_n = x.shape[0]
        query = x.mean(dim=2).permute(0, 2, 1)
        x = x.view(list(x.shape[:2]) + [-1]).permute(0, 2, 1)

        pos_x = torch.linspace(0, 1, feat_w, device=x.device) - 0.5
        pos_y = torch.linspace(0, 1, feat_h, device=x.device) - 0.5
        try:
            grid_x, grid_y = torch.meshgrid(pos_x, pos_y, indexing='ij')
        except TypeError:
            grid_x, grid_y = torch.meshgrid(pos_x, pos_y)
        pos_grid_2d = torch.stack((grid_x, grid_y), dim=-1)
        pos_enc_2d = self.pos_mlp_2d(pos_grid_2d).reshape((1, -1, 32)).repeat((batch_n, 1, 1))
        x = torch.cat((x, pos_enc_2d), dim=-1)

        pos_v = torch.linspace(0, 1, feat_w, device=x.device) - 0.5
        pos_enc_1d = self.pos_mlp_1d(pos_v.reshape((-1, 1))).reshape((1, -1, 32)).repeat((batch_n, 1, 1))
        query = torch.cat((query, pos_enc_1d), dim=-1)

        query = self.q_proj(query)
        key = self.k_proj(x)
        value = self.v_proj(x)
        if mask is not None:
            mask = tvf.resize(mask, (feat_h, feat_w), tvf.InterpolationMode.NEAREST).type(torch.bool)
            mask = torch.logical_not(mask).reshape((mask.shape[0], 1, -1)).repeat(1, feat_w, 1)
        out, attn = self.attn(query, key, value, attn_mask=mask)
        return out, attn


class FusionSelector(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.mlp(x)


class LegacyDualExpertFusion(nn.Module):
    def __init__(self, mv_net, mono_net, d_min=0.1, d_max=20.0, d_hyp=-0.2, D=128, selector_hidden_dim=64):
        super().__init__()
        self.mv_net = mv_net
        self.mono_net = mono_net
        self.mv_net.requires_grad_(False)
        self.mono_net.requires_grad_(False)
        self.mv_net.eval()
        self.mono_net.eval()
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.selector = FusionSelector(hidden_dim=selector_hidden_dim)

    def _expert_predict(self, expert, obs_img):
        with torch.no_grad():
            depth, _, prob = expert(obs_img)
        return depth, prob

    def forward(self, batch_or_img):
        obs_img = batch_or_img['ref_img'] if isinstance(batch_or_img, dict) else batch_or_img
        depth_mv, prob_mv = self._expert_predict(self.mv_net, obs_img)
        depth_mono, prob_mono = self._expert_predict(self.mono_net, obs_img)
        weights = self.selector(torch.stack((depth_mv.mean(dim=-1), depth_mono.mean(dim=-1)), dim=-1))
        probs = torch.stack((prob_mv, prob_mono), dim=1)
        prob_comp = (probs * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        d_vals = torch.linspace(self.d_min ** self.d_hyp, self.d_max ** self.d_hyp, self.D, device=prob_comp.device) ** (1 / self.d_hyp)
        d_comp = torch.sum(prob_comp * d_vals.view(1, 1, -1), dim=-1)
        return {
            'd_comp': d_comp,
            'prob_comp': prob_comp,
            'prob_mono': prob_mono.unsqueeze(1),
            'comp_w': weights,
            'mono_dict': {'d_mono': depth_mono.unsqueeze(1), 'prob_mono': prob_mono.unsqueeze(1)},
        }
