# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      build_activation_layer, build_norm_layer)
from mmcv.runner import BaseModule

from mmseg.ops import resize
from ..builder import BACKBONES


class DetailBranch(BaseModule):
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
        super(DetailBranch, self).__init__(init_cfg=init_cfg)
        inter_channels = 64
        self.conv7x7 = _ConvBNReLU(in_channels, inter_channels, 7, 2, 3, norm_layer=norm_layer)
        self.initation = InitialBlock(num_channels[0], norm_layer=nn.BatchNorm2d)
        self.CFF = CascadeFeatureFusion(num_channels[3],num_channels[0], num_channels[3])
        self.conv_low = nn.Sequential(
            nn.Conv2d(num_channels[3], num_channels[3], kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(num_channels[3])
        )
        self.conv_low_cls = nn.Conv2d(num_channels[3], 8, 1, bias=False)

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
        # x = self.inplace_abn(x)

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
        # print("x_conv", x_conv.shape)
        # print("x_pool",x_pool.shape)
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
            nn.Conv2d(16, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )


    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)

        return x


class StemBlock(BaseModule):
    """Stem Block at the beginning of Semantic Branch.

    Args:
        in_channels (int): Number of input channels.
            Default: 3.
        out_channels (int): Number of output channels.
            Default: 16.
        conv_cfg (dict | None): Config of conv layers.
            Default: None.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Returns:
        x (torch.Tensor): First feature map in Semantic Branch.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=16,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(StemBlock, self).__init__(init_cfg=init_cfg)

        self.conv_first = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.convs = nn.Sequential(
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels // 2,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.pool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse_last = ConvModule(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv_first(x)
        x_left = self.convs(x)
        x_right = self.pool(x)
        x = self.fuse_last(torch.cat([x_left, x_right], dim=1))
        return x


class GELayer(BaseModule):
    """Gather-and-Expansion Layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        exp_ratio (int): Expansion ratio for middle channels.
            Default: 6.
        stride (int): Stride of GELayer. Default: 1
        conv_cfg (dict | None): Config of conv layers.
            Default: None.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Returns:
        x (torch.Tensor): Intermediate feature map in
            Semantic Branch.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 exp_ratio=6,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(GELayer, self).__init__(init_cfg=init_cfg)
        mid_channel = in_channels * exp_ratio
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if stride == 1:
            self.dwconv = nn.Sequential(
                # ReLU in ConvModule not shown in paper
                ConvModule(
                    in_channels=in_channels,
                    out_channels=mid_channel,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=in_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.shortcut = None
        else:
            self.dwconv = nn.Sequential(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=mid_channel,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                # ReLU in ConvModule not shown in paper
                ConvModule(
                    in_channels=mid_channel,
                    out_channels=mid_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=mid_channel,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
            )
            self.shortcut = nn.Sequential(
                DepthwiseSeparableConvModule(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    dw_norm_cfg=norm_cfg,
                    dw_act_cfg=None,
                    pw_norm_cfg=norm_cfg,
                    pw_act_cfg=None,
                ))

        self.conv2 = nn.Sequential(
            ConvModule(
                in_channels=mid_channel,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
            ))

        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv2(x)
        if self.shortcut is not None:
            shortcut = self.shortcut(identity)
            x = x + shortcut
        else:
            x = x + identity
        x = self.act(x)
        return x


class CEBlock(BaseModule):
    """Context Embedding Block for large receptive filed in Semantic Branch.

    Args:
        in_channels (int): Number of input channels.
            Default: 3.
        out_channels (int): Number of output channels.
            Default: 16.
        conv_cfg (dict | None): Config of conv layers.
            Default: None.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Returns:
        x (torch.Tensor): Last feature map in Semantic Branch.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=16,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(CEBlock, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            build_norm_layer(norm_cfg, self.in_channels)[1])
        self.conv_gap = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # Note: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvModule(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        identity = x
        x = self.gap(x)
        x = self.conv_gap(x)
        x = identity + x
        x = self.conv_last(x)
        return x


class SemanticBranch(BaseModule):
    """Semantic Branch which is lightweight with narrow channels and deep
    layers to obtain　high-level semantic context.

    Args:
        semantic_channels(Tuple[int]): Size of channel numbers of
            various stages in Semantic Branch.
            Default: (16, 32, 64, 128).
        in_channels (int): Number of channels of input image. Default: 3.
        exp_ratio (int): Expansion ratio for middle channels.
            Default: 6.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Returns:
        semantic_outs (List[torch.Tensor]): List of several feature maps
            for auxiliary heads (Booster) and Bilateral
            Guided Aggregation Layer.
    """

    def __init__(self,
                 semantic_channels=(16, 32, 64, 128),
                 in_channels=3,
                 exp_ratio=6,
                 init_cfg=None):
        super(SemanticBranch, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.semantic_channels = semantic_channels
        self.semantic_stages = []
        for i in range(len(semantic_channels)):
            stage_name = f'stage{i + 1}'
            self.semantic_stages.append(stage_name)
            if i == 0:
                self.add_module(
                    stage_name,
                    StemBlock(self.in_channels, semantic_channels[i]))
            elif i == (len(semantic_channels) - 1):
                self.add_module(
                    stage_name,
                    nn.Sequential(
                        GELayer(semantic_channels[i - 1], semantic_channels[i],
                                exp_ratio, 2),
                        GELayer(semantic_channels[i], semantic_channels[i],
                                exp_ratio, 1),
                        GELayer(semantic_channels[i], semantic_channels[i],
                                exp_ratio, 1),
                        GELayer(semantic_channels[i], semantic_channels[i],
                                exp_ratio, 1)))
            else:
                self.add_module(
                    stage_name,
                    nn.Sequential(
                        GELayer(semantic_channels[i - 1], semantic_channels[i],
                                exp_ratio, 2),
                        GELayer(semantic_channels[i], semantic_channels[i],
                                exp_ratio, 1)))

        self.add_module(f'stage{len(semantic_channels)}_CEBlock',
                        CEBlock(semantic_channels[-1], semantic_channels[-1]))
        self.semantic_stages.append(f'stage{len(semantic_channels)}_CEBlock')

    def forward(self, x):
        semantic_outs = []
        for stage_name in self.semantic_stages:
            semantic_stage = getattr(self, stage_name)
            x = semantic_stage(x)
            semantic_outs.append(x)
        return semantic_outs


class BGALayer(BaseModule):
    """Bilateral Guided Aggregation Layer to fuse the complementary information
    from both Detail Branch and Semantic Branch.

    Args:
        out_channels (int): Number of output channels.
            Default: 128.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        conv_cfg (dict | None): Config of conv layers.
            Default: None.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Returns:
        output (torch.Tensor): Output feature map for Segment heads.
    """

    def __init__(self,
                 out_channels=128,
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(BGALayer, self).__init__(init_cfg=init_cfg)
        self.out_channels = out_channels
        self.align_corners = align_corners
        self.detail_dwconv = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dw_norm_cfg=norm_cfg,
                dw_act_cfg=None,
                pw_norm_cfg=None,
                pw_act_cfg=None,
            ))
        self.detail_down = nn.Sequential(
            ConvModule(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.semantic_conv = nn.Sequential(
            ConvModule(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None))
        self.semantic_dwconv = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dw_norm_cfg=norm_cfg,
                dw_act_cfg=None,
                pw_norm_cfg=None,
                pw_act_cfg=None,
            ))
        self.conv = ConvModule(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            inplace=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x_d, x_s):
        detail_dwconv = self.detail_dwconv(x_d)
        detail_down = self.detail_down(x_d)
        semantic_conv = self.semantic_conv(x_s)
        semantic_dwconv = self.semantic_dwconv(x_s)
        semantic_conv = resize(
            input=semantic_conv,
            size=detail_dwconv.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv)
        fuse_2 = detail_down * torch.sigmoid(semantic_dwconv)
        fuse_2 = resize(
            input=fuse_2,
            size=fuse_1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = self.conv(fuse_1 + fuse_2)
        return output


@BACKBONES.register_module()
class BiSeNetV2(BaseModule):
    """BiSeNetV2: Bilateral Network with Guided Aggregation for
    Real-time Semantic Segmentation.

    This backbone is the implementation of
    `BiSeNetV2 <https://arxiv.org/abs/2004.02147>`_.

    Args:
        in_channels (int): Number of channel of input image. Default: 3.
        detail_channels (Tuple[int], optional): Channels of each stage
            in Detail Branch. Default: (64, 64, 128).
        semantic_channels (Tuple[int], optional): Channels of each stage
            in Semantic Branch. Default: (16, 32, 64, 128).
            See Table 1 and Figure 3 of paper for more details.
        semantic_expansion_ratio (int, optional): The expansion factor
            expanding channel number of middle channels in Semantic Branch.
            Default: 6.
        bga_channels (int, optional): Number of middle channels in
            Bilateral Guided Aggregation Layer. Default: 128.
        out_indices (Tuple[int] | int, optional): Output from which stages.
            Default: (0, 1, 2, 3, 4).
        align_corners (bool, optional): The align_corners argument of
            resize operation in Bilateral Guided Aggregation Layer.
            Default: False.
        conv_cfg (dict | None): Config of conv layers.
            Default: None.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 detail_channels=(64, 64, 128),
                 semantic_channels=(16, 32, 64, 128),
                 semantic_expansion_ratio=6,
                 bga_channels=128,
                 out_indices=(0, 1, 2, 3, 4),
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        if init_cfg is None:
            init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(
                    type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
            ]
        super(BiSeNetV2, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.detail_channels = detail_channels
        self.semantic_channels = semantic_channels
        self.semantic_expansion_ratio = semantic_expansion_ratio
        self.bga_channels = bga_channels
        self.align_corners = align_corners
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.detail = DetailBranch()
        self.semantic = SemanticBranch(self.semantic_channels,
                                       self.in_channels,
                                       self.semantic_expansion_ratio)
        self.bga = BGALayer(self.bga_channels, self.align_corners)

    def forward(self, x):
        #  stole refactoring code from Coin Cheung, thanks
        x_semantic_lst = self.semantic(x)
        x_detail, x_cls = self.detail(x, x_semantic_lst[0])
        x_head = self.bga(x_detail, x_semantic_lst[-1])
        outs = [x_head] + x_semantic_lst[:-1]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs), x_cls