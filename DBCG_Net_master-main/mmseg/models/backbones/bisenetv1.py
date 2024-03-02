# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmseg.ops import resize
from ..builder import BACKBONES, build_backbone
import torch.nn.functional as F
import torch.fft as fft

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Filter(image, d, n):
    f = fft.fft2(image)
    fshift = fft.fftshift(f).to(device)
    fimg = torch.log(torch.abs(fshift))

    def make_transform_matrix(d):
        transform_matrix = torch.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, fimg.shape))
        for i in range(transform_matrix.shape[0]):
            for j in range(transform_matrix.shape[1]):
                def cal_distance(pa, pb):
                    from math import sqrt
                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis

                dis = cal_distance(center_point, (i, j))
                transform_matrix[i, j] = 1 / (1 + (dis / d) ** (2 * n))
        return transform_matrix

    d_matrix = make_transform_matrix(d).to(device)
    new_img = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fshift * d_matrix)))
    return new_img


class SpatialPath(BaseModule):
    """Spatial Path to preserve the spatial size of the original input image
    and encode affluent spatial information.

    Args:
        in_channels(int): The number of channels of input
            image. Default: 3.
        num_channels (Tuple[int]): The number of channels of
            each layers in Spatial Path.
            Default: (64, 64, 64, 128).
    Returns:
        x (torch.Tensor): Feature map for Feature Fusion Module.
    """

    def __init__(self,
                 in_channels=3,
                 num_channels=(64, 64, 64, 128),
                 conv_cfg=None,
                 norm_layer=nn.BatchNorm2d,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(SpatialPath, self).__init__(init_cfg=init_cfg)
        assert len(num_channels) == 4, 'Length of input channels \
                                        of Spatial Path must be 4!'
        inter_channels = 64
        self.conv7x7 = _ConvBNReLU(in_channels, inter_channels, 7, 2, 3, norm_layer=norm_layer)
        self.initation = InitialBlock(num_channels[0], norm_layer=nn.BatchNorm2d)
        self.CFF = CascadeFeatureFusion(num_channels[3], num_channels[0], num_channels[3])
        self.conv_low = nn.Sequential(
            nn.Conv2d(num_channels[3], num_channels[3], kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(num_channels[3])
        )
        self.conv_low_cls = nn.Conv2d(num_channels[3], 6, 1, bias=False)

    def forward(self, x1, x2):
        x1 = self.conv7x7(x1)
        x1 = self.initation(x1)
        x_out = self.CFF(x1, x2)
        x_out = self.conv_low(x_out)
        x_out_cls = self.conv_low_cls(x_out)
        return x_out, x_out_cls


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)
        # self.inplace_abn = InPlaceABNSync(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class InitialBlock(nn.Module):
    """ENet initial block"""

    def __init__(self, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(InitialBlock, self).__init__()
        self.main_branch = nn.Conv2d(64, out_channels, 3, 2, 1, bias=False)
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)
        self.bn = norm_layer(out_channels + 64)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

    def forward(self, x):
        x_conv = self.main_branch(x)
        x_pool = self.ext_branch(x)
        x = torch.cat([x_conv, x_pool], dim=1)
        x = self.bn(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class CascadeFeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            norm_layer(out_channels)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)

        return x


class AttentionRefinementModule(BaseModule):
    """Attention Refinement Module (ARM) to refine the features of each stage.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    Returns:
        x_out (torch.Tensor): Feature map of Attention Refinement Module.
    """

    def __init__(self,
                 in_channels,
                 out_channel,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(AttentionRefinementModule, self).__init__(init_cfg=init_cfg)
        self.conv_layer = ConvModule(
            in_channels=in_channels,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.atten_conv_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None), nn.Sigmoid())

    def forward(self, x):
        x = self.conv_layer(x)
        x_atten = self.atten_conv_layer(x)
        x_out = x * x_atten
        return x_out


class ContextPath(BaseModule):
    """Context Path to provide sufficient receptive field.

    Args:
        backbone_cfg:(dict): Config of backbone of
            Context Path.
        context_channels (Tuple[int]): The number of channel numbers
            of various modules in Context Path.
            Default: (128, 256, 512).
        align_corners (bool, optional): The align_corners argument of
            resize operation. Default: False.
    Returns:
        x_16_up, x_32_up (torch.Tensor, torch.Tensor): Two feature maps
            undergoing upsampling from 1/16 and 1/32 downsampling
            feature maps. These two feature maps are used for Feature
            Fusion Module and Auxiliary Head.
    """

    def __init__(self,
                 backbone_cfg,
                 context_channels=(128, 256, 512),
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(ContextPath, self).__init__(init_cfg=init_cfg)
        assert len(context_channels) == 3, 'Length of input channels \
                                           of Context Path must be 3!'

        self.backbone = build_backbone(backbone_cfg)

        self.align_corners = align_corners
        self.arm16 = AttentionRefinementModule(context_channels[1],
                                               context_channels[0])
        self.arm32 = AttentionRefinementModule(context_channels[2],
                                               context_channels[0])
        self.conv_head32 = ConvModule(
            in_channels=context_channels[0],
            out_channels=context_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv_head16 = ConvModule(
            in_channels=context_channels[0],
            out_channels=context_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gap_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                in_channels=context_channels[2],
                out_channels=context_channels[0],
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, x):
        x_4, x_8, x_16, x_32 = self.backbone(x)
        x_gap = self.gap_conv(x_32)

        x_32_arm = self.arm32(x_32)
        x_32_sum = x_32_arm + x_gap
        x_32_up = resize(input=x_32_sum, size=x_16.shape[2:], mode='nearest')
        x_32_up = self.conv_head32(x_32_up)

        x_16_arm = self.arm16(x_16)
        x_16_sum = x_16_arm + x_32_up
        x_16_up = resize(input=x_16_sum, size=x_8.shape[2:], mode='nearest')
        x_16_up = self.conv_head16(x_16_up)

        return x_16_up, x_32_up, x_4


class FeatureFusionModule(BaseModule):
    """Feature Fusion Module to fuse low level output feature of Spatial Path
    and high level output feature of Context Path.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    Returns:
        x_out (torch.Tensor): Feature map of Feature Fusion Module.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(FeatureFusionModule, self).__init__(init_cfg=init_cfg)
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_atten = nn.Sequential(
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg), nn.Sigmoid())

    def forward(self, x_sp, x_cp):
        x_concat = torch.cat([x_sp, x_cp], dim=1)
        x_fuse = self.conv1(x_concat)
        x_atten = self.gap(x_fuse)
        # Note: No BN and more 1x1 conv in paper.
        x_atten = self.conv_atten(x_atten)
        x_atten = x_fuse * x_atten
        x_out = x_atten + x_fuse
        return x_out


@BACKBONES.register_module()
class BiSeNetV1(BaseModule):
    """BiSeNetV1 backbone.

    This backbone is the implementation of `BiSeNet: Bilateral
    Segmentation Network for Real-time Semantic
    Segmentation <https://arxiv.org/abs/1808.00897>`_.

    Args:
        backbone_cfg:(dict): Config of backbone of
            Context Path.
        in_channels (int): The number of channels of input
            image. Default: 3.
        spatial_channels (Tuple[int]): Size of channel numbers of
            various layers in Spatial Path.
            Default: (64, 64, 64, 128).
        context_channels (Tuple[int]): Size of channel numbers of
            various modules in Context Path.
            Default: (128, 256, 512).
        out_indices (Tuple[int] | int, optional): Output from which stages.
            Default: (0, 1, 2).
        align_corners (bool, optional): The align_corners argument of
            resize operation in Bilateral Guided Aggregation Layer.
            Default: False.
        out_channels(int): The number of channels of output.
            It must be the same with `in_channels` of decode_head.
            Default: 256.
    """

    def __init__(self,
                 backbone_cfg,
                 in_channels=3,
                 spatial_channels=(64, 64, 64, 128),
                 context_channels=(128, 256, 512),
                 out_indices=(0, 1, 2),
                 align_corners=False,
                 out_channels=256,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(BiSeNetV1, self).__init__(init_cfg=init_cfg)
        assert len(spatial_channels) == 4, 'Length of input channels \
                                           of Spatial Path must be 4!'

        assert len(context_channels) == 3, 'Length of input channels \
                                           of Context Path must be 3!'

        self.out_indices = out_indices
        self.align_corners = align_corners
        self.context_path = ContextPath(backbone_cfg, context_channels,
                                        self.align_corners)
        self.spatial_path = SpatialPath(in_channels, spatial_channels)
        self.ffm = FeatureFusionModule(context_channels[1], out_channels)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

    def forward(self, x):
        # stole refactoring code from Coin Cheung, thanks
        x_context8, x_context16, c2 = self.context_path(x)
        x_spatial, x_cls = self.spatial_path(x, c2)
        x_fuse = self.ffm(x_spatial, x_context8)

        outs = [x_fuse, x_context8, x_context16]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs), x_cls
