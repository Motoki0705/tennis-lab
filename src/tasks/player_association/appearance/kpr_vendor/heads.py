# ruff: noqa
# mypy: ignore-errors
# KPR inference heads, HL3; see NOTICE.md and LICENSE.
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class MultiStageFusion(nn.Module):
    """Feature Pyramid Network to resize features maps from various backbone stages and concat them together to form
    a single high resolution feature map. Inspired from HrNet implementation in https://github.com/CASIA-IVA-Lab/ISP-reID/blob/master/modeling/backbones/cls_hrnet.py"""

    def __init__(self, input_dim, output_dim, mode="bilinear", img_size=None, spatial_scale=-1):
        super(MultiStageFusion, self).__init__()
        if spatial_scale > 0:
            self.spatial_size = np.array(img_size) / spatial_scale
        else:
            self.spatial_size = None
        self.mode = mode
        self.dim_reduce = BeforePoolingDimReduceLayer(input_dim, output_dim)

    def forward(self, features_per_stage):
        if self.spatial_size is None:
            spatial_size = features_per_stage[0].size()[2:]
        else:
            spatial_size = self.spatial_size
        resized_feature_maps = [features_per_stage[0]]
        for i in range(1, len(features_per_stage)):
            resized_feature_maps.append(
                F.interpolate(
                    features_per_stage[i],
                    size=spatial_size,
                    mode=self.mode,
                    align_corners=True
                )
            )
        fused_features = torch.cat(resized_feature_maps, 1)
        fused_features = self.dim_reduce(fused_features)
        return fused_features

class BeforePoolingDimReduceLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BeforePoolingDimReduceLayer, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                input_dim, output_dim, 1, stride=1, padding=0
            )
        )
        layers.append(nn.BatchNorm2d(output_dim))
        layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self._init_params()

    def forward(self, x):
        return self.layers(x)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class AfterPoolingDimReduceLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p=None):
        super(AfterPoolingDimReduceLayer, self).__init__()
        # dim reduction used in ResNet and PCB
        layers = []
        layers.append(
            nn.Linear(
                input_dim, output_dim, bias=True
            )
        )
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout_p is not None:
            layers.append(nn.opout(p=dropout_p))

        self.layers = nn.Sequential(*layers)
        self._init_params()

    def forward(self, x):
        # Fixed inference ranks are checked by the public wrapper. Preserve
        # both upstream 2D/3D cases with arithmetic shape handling only.
        shape = x.shape
        x = self.layers(x.reshape(-1, shape[-1]))
        return x.reshape(*shape[:-1], -1)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class PixelToPartClassifier(nn.Module):
    def __init__(self, dim_reduce_output, parts_num):
        super(PixelToPartClassifier, self).__init__()
        self.bn = torch.nn.BatchNorm2d(dim_reduce_output)
        self.classifier = nn.Conv2d(in_channels=dim_reduce_output, out_channels=parts_num + 1, kernel_size=1, stride=1, padding=0)
        self._init_params()

    def forward(self, x):
        x = self.bn(x)
        return self.classifier(x)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class BNClassifier(nn.Module):
    # Source: https://github.com/upgirlnana/Pytorch-Person-REID-Baseline-Bag-of-Tricks
    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)  # BoF: this doesn't have a big impact on perf according to author on github
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self._init_params()

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class GlobalMaskWeightedPoolingHead(nn.Module):
    def __init__(self, depth, normalization='identity'):
        super().__init__()
        if normalization == 'identity':
            self.normalization = nn.Identity()
        elif normalization == 'batch_norm_3d':
            self.normalization = torch.nn.BatchNorm3d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        elif normalization == 'batch_norm_2d':
            self.normalization = torch.nn.BatchNorm2d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        elif normalization == 'batch_norm_1d':
            self.normalization = torch.nn.BatchNorm1d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        else:
            raise ValueError('normalization type {} not supported'.format(normalization))

    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)
        features = torch.unsqueeze(features, 1)
        parts_features = torch.mul(part_masks, features)

        N, M, _, _, _ = parts_features.size()
        parts_features = parts_features.flatten(0, 1)
        parts_features = self.normalization(parts_features)
        parts_features = self.global_pooling(parts_features)
        parts_features = parts_features.view(N, M, -1)
        return parts_features

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class GlobalAveragePoolingHead(GlobalMaskWeightedPoolingHead):
    global_pooling = nn.AdaptiveAvgPool2d((1, 1))

class GlobalWeightedAveragePoolingHead(GlobalMaskWeightedPoolingHead):
    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)
        features = torch.unsqueeze(features, 1)
        parts_features = torch.mul(part_masks, features)

        N, M, _, _, _ = parts_features.size()
        parts_features = parts_features.flatten(0, 1)
        parts_features = self.normalization(parts_features)
        parts_features = torch.sum(parts_features, dim=(-2, -1))
        part_masks_sum = torch.sum(part_masks.flatten(0, 1), dim=(-2, -1))
        part_masks_sum = torch.clamp(part_masks_sum, min=1e-6)
        parts_features_avg = torch.div(parts_features, part_masks_sum)
        parts_features = parts_features_avg.view(N, M, -1)
        return parts_features
