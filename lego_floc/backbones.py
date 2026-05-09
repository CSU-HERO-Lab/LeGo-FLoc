import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResNet50FeatureExtractor(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        pos_embed_dim=32,
        num_heads=8,
        target_width=40,
        pretrained=True,
        freeze_pretrained=False,
        checkpoint_path=None,
        checkpoint_variant=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.target_width = int(target_width)

        try:
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = resnet50(weights=weights)
        except TypeError:
            backbone = resnet50(pretrained=pretrained)

        if checkpoint_path is not None:
            params = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(params, dict) and 'state_dict' in params:
                params = params['state_dict']
            cleaned = OrderedDict()
            for key, value in params.items():
                new_key = key.replace('module.', '')
                if checkpoint_variant == '3dp':
                    new_key = new_key.replace('.shortcut.weight', '.downsample.0.weight')
                    new_key = new_key.replace('.shortcut.norm.', '.downsample.1.')
                cleaned[new_key] = value
            missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
            print(
                f"[ResNet50FeatureExtractor] loaded checkpoint {checkpoint_path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )

        self.pretrained = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        for param in self.pretrained.parameters():
            param.requires_grad = not freeze_pretrained

        total = sum(p.numel() for p in self.pretrained.parameters())
        trainable = sum(p.numel() for p in self.pretrained.parameters() if p.requires_grad)
        print(
            f"[ResNet50FeatureExtractor] pretrained trainable params: {trainable}/{total} "
            f"(pretrained={pretrained}, freeze_pretrained={freeze_pretrained})"
        )

        self.conv = ConvBnReLU(
            in_channels=2048,
            out_channels=self.embed_dim,
            kernel_size=1,
            padding=0,
            stride=1,
        )
        self.vertical_pool = VerticalAttentionPooling(in_channels=self.embed_dim)

        self.pos_mlp_2d = nn.Sequential(
            nn.Linear(2, self.pos_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_embed_dim, self.pos_embed_dim),
        )
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, self.pos_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_embed_dim, self.pos_embed_dim),
        )

        total_embed_dim = self.embed_dim + self.pos_embed_dim
        self.q_proj = nn.Linear(total_embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(total_embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(total_embed_dim, self.embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, obs_img, mask=None):
        bsz = obs_img.shape[0]
        features_tensor_2d = self.pretrained(obs_img)
        feat_h, feat_w = features_tensor_2d.shape[-2:]
        target_w = self.target_width
        target_h = max(1, int(round(float(feat_h) * float(target_w) / float(feat_w))))
        target_size = (target_h, target_w)

        interpolated_features = F.interpolate(
            features_tensor_2d,
            size=target_size,
            mode='bilinear',
            align_corners=False,
        )

        x_2d = self.conv(interpolated_features)
        x_weighted, pooling_weights = self.vertical_pool(x_2d)
        query = x_weighted.permute(0, 2, 1)

        x_2d = x_2d.view(bsz, self.embed_dim, -1).permute(0, 2, 1)

        pos_x = torch.linspace(0, 1, target_w, device=x_2d.device) - 0.5
        pos_y = torch.linspace(0, 1, target_h, device=x_2d.device) - 0.5
        pos_grid_2d_y, pos_grid_2d_x = torch.meshgrid(pos_y, pos_x, indexing='ij')
        pos_grid_2d = torch.stack((pos_grid_2d_x, pos_grid_2d_y), dim=-1)

        pos_enc_2d = self.pos_mlp_2d(pos_grid_2d)
        pos_enc_2d = pos_enc_2d.reshape((1, -1, self.pos_embed_dim)).repeat((bsz, 1, 1))
        x_2d = torch.cat((x_2d, pos_enc_2d), dim=-1)

        pos_v = torch.linspace(0, 1, target_w, device=query.device) - 0.5
        pos_enc_1d = self.pos_mlp_1d(pos_v.reshape((-1, 1)))
        pos_enc_1d = pos_enc_1d.reshape((1, -1, self.pos_embed_dim)).repeat((bsz, 1, 1))
        query = torch.cat((query, pos_enc_1d), dim=-1)

        query = self.q_proj(query)
        key = self.k_proj(x_2d)
        value = self.v_proj(x_2d)

        x_out, attn_w = self.attn(query, key, value)
        return x_out, attn_w, None


class IBN(nn.Module):
    def __init__(self, planes, ratio=0.5):
        super().__init__()
        half1 = int(planes * ratio)
        half2 = planes - half1
        self.half1 = half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, [self.half1, x.size(1) - self.half1], dim=1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        return torch.cat((out1, out2), dim=1)


class BottleneckIBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNetIBNBackbone(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], stride=1, ibn=True)
        self.layer2 = self._make_layer(128, layers[1], stride=2, ibn=True)
        self.layer3 = self._make_layer(256, layers[2], stride=2, ibn=True)
        self.layer4 = self._make_layer(512, layers[3], stride=2, ibn=False)
        self.base = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )

    def _make_layer(self, planes, blocks, stride=1, ibn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * BottleneckIBN.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BottleneckIBN.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BottleneckIBN.expansion),
            )
        layers = [BottleneckIBN(self.inplanes, planes, ibn=ibn, stride=stride, downsample=downsample)]
        self.inplanes = planes * BottleneckIBN.expansion
        for _ in range(1, blocks):
            layers.append(BottleneckIBN(self.inplanes, planes, ibn=ibn, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.base(x)


class ResNet50RSKFeatureExtractor(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        pos_embed_dim=32,
        num_heads=8,
        target_width=40,
        checkpoint_path=None,
        freeze_pretrained=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.target_width = int(target_width)

        backbone = ResNetIBNBackbone([3, 4, 6, 3])
        self.pretrained = backbone.base
        if checkpoint_path is not None:
            params = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(params, dict) and 'state_dict' in params:
                params = params['state_dict']
            cleaned = OrderedDict()
            for key, value in params.items():
                if key.startswith('module.base.'):
                    cleaned[key[len('module.base.'):]] = value
                elif key.startswith('base.'):
                    cleaned[key[len('base.'):]] = value
            missing, unexpected = self.pretrained.load_state_dict(cleaned, strict=False)
            print(
                f"[ResNet50RSKFeatureExtractor] loaded checkpoint {checkpoint_path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        for param in self.pretrained.parameters():
            param.requires_grad = not freeze_pretrained

        total = sum(p.numel() for p in self.pretrained.parameters())
        trainable = sum(p.numel() for p in self.pretrained.parameters() if p.requires_grad)
        print(
            f"[ResNet50RSKFeatureExtractor] pretrained trainable params: {trainable}/{total} "
            f"(freeze_pretrained={freeze_pretrained})"
        )

        self.conv = ConvBnReLU(
            in_channels=2048,
            out_channels=self.embed_dim,
            kernel_size=1,
            padding=0,
            stride=1,
        )
        self.vertical_pool = VerticalAttentionPooling(in_channels=self.embed_dim)

        self.pos_mlp_2d = nn.Sequential(
            nn.Linear(2, self.pos_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_embed_dim, self.pos_embed_dim),
        )
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, self.pos_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_embed_dim, self.pos_embed_dim),
        )

        total_embed_dim = self.embed_dim + self.pos_embed_dim
        self.q_proj = nn.Linear(total_embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(total_embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(total_embed_dim, self.embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, obs_img, mask=None):
        bsz = obs_img.shape[0]
        features_tensor_2d = self.pretrained(obs_img)
        feat_h, feat_w = features_tensor_2d.shape[-2:]
        target_w = self.target_width
        target_h = max(1, int(round(float(feat_h) * float(target_w) / float(feat_w))))
        target_size = (target_h, target_w)

        interpolated_features = F.interpolate(
            features_tensor_2d,
            size=target_size,
            mode='bilinear',
            align_corners=False,
        )

        x_2d = self.conv(interpolated_features)
        x_weighted, _ = self.vertical_pool(x_2d)
        query = x_weighted.permute(0, 2, 1)

        x_2d = x_2d.view(bsz, self.embed_dim, -1).permute(0, 2, 1)

        pos_x = torch.linspace(0, 1, target_w, device=x_2d.device) - 0.5
        pos_y = torch.linspace(0, 1, target_h, device=x_2d.device) - 0.5
        pos_grid_2d_y, pos_grid_2d_x = torch.meshgrid(pos_y, pos_x, indexing='ij')
        pos_grid_2d = torch.stack((pos_grid_2d_x, pos_grid_2d_y), dim=-1)
        pos_enc_2d = self.pos_mlp_2d(pos_grid_2d)
        pos_enc_2d = pos_enc_2d.reshape((1, -1, self.pos_embed_dim)).repeat((bsz, 1, 1))
        x_2d = torch.cat((x_2d, pos_enc_2d), dim=-1)

        pos_v = torch.linspace(0, 1, target_w, device=query.device) - 0.5
        pos_enc_1d = self.pos_mlp_1d(pos_v.reshape((-1, 1)))
        pos_enc_1d = pos_enc_1d.reshape((1, -1, self.pos_embed_dim)).repeat((bsz, 1, 1))
        query = torch.cat((query, pos_enc_1d), dim=-1)

        query = self.q_proj(query)
        key = self.k_proj(x_2d)
        value = self.v_proj(x_2d)
        x_out, attn_w = self.attn(query, key, value)
        return x_out, attn_w, None


class VerticalAttentionPooling(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x, mask=None):
        scores = self.net(x)
        attn_weights = F.softmax(scores, dim=2)
        x_weighted = (x * attn_weights).sum(dim=2)
        return x_weighted, attn_weights
