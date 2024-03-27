# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:57:49 2019

@author: Fsl
"""
from scipy import ndimage
import torch
from torchvision import models
import torch.nn as nn
# from .resnet import resnet34
# from resnet import resnet34
# import resnet
from torch.nn import functional as F
# import torchsummary
from torch.nn import init
import numpy as np
from functools import partial
from thop import profile
up_kwargs = {'mode': 'bilinear', 'align_corners': True}
BatchNorm2d = nn.BatchNorm2d

class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//8,kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out
class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out
class AffinityAttention(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab
        return out
class AffinityAttention2(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention2, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(sab)
        out = sab + cab
        return out


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(size=scale_factor, mode='bilinear'), )

    def forward(self, input):
        return self.dsv(input)
def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, groups=group, bias=bias)
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # scalecoe = F.sigmoid(channel_att_sum)
        channel_att_sum = channel_att_sum.reshape(channel_att_sum.shape[0], 4, 4)# (_, 4, * / 4)
        avg_weight = torch.mean(channel_att_sum, dim=2).unsqueeze(2)
        avg_weight = avg_weight.expand(channel_att_sum.shape[0], 4, 4).reshape(channel_att_sum.shape[0], 16)
        scale = F.sigmoid(avg_weight).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale, scale
class Scale_atten_block(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(Scale_atten_block, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialAtten(gate_channels, gate_channels //reduction_ratio)

    def forward(self, x):
        x_out, ca_atten = self.ChannelGate(x)
        if not self.no_spatial:
            x_out, sa_atten = self.SpatialGate(x_out)

        return x_out, ca_atten, sa_atten
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class SpatialAtten(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=1):
        super(SpatialAtten, self).__init__()
        self.conv1 = BasicConv(in_size, out_size, kernel_size, stride=stride,
                               padding=(kernel_size-1) // 2, relu=True)
        self.conv2 = BasicConv(out_size, out_size, kernel_size=1, stride=stride,
                               padding=0, relu=True, bn=False)

    def forward(self, x):
        residual = x
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        spatial_att = F.sigmoid(x_out).unsqueeze(4).permute(0, 1, 4, 2, 3)
        spatial_att = spatial_att.expand(spatial_att.shape[0], 4, 4, spatial_att.shape[3], spatial_att.shape[4]).reshape(
                                        spatial_att.shape[0], 16, spatial_att.shape[3], spatial_att.shape[4])
        x_out = residual * spatial_att

        x_out += residual

        return x_out, spatial_att
class scale_atten_convblock(nn.Module):
    def __init__(self, in_size, out_size, stride=1, downsample=None, use_cbam=True, no_spatial=False, drop_out=False):
        super(scale_atten_convblock, self).__init__()
        # if stride != 1 or in_size != out_size:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(in_size, out_size,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )
        self.downsample = downsample
        self.stride = stride
        self.no_spatial = no_spatial
        self.dropout = drop_out

        self.relu = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(in_size, out_size)
        self.bn3 = nn.BatchNorm2d(out_size)

        if use_cbam:
            self.cbam = Scale_atten_block(in_size, reduction_ratio=4, no_spatial=self.no_spatial)  # out_size
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out, scale_c_atten, scale_s_atten = self.cbam(x)

            # scale_c_atten = nn.Sigmoid()(scale_c_atten)
            # scale_s_atten = nn.Sigmoid()(scale_s_atten)
            # scale_atten = channel_atten_c * spatial_atten_s

        # scale_max = torch.argmax(scale_atten, dim=1, keepdim=True)
        # scale_max_soft = get_soft_label(input_tensor=scale_max, num_class=8)
        # scale_max_soft = scale_max_soft.permute(0, 3, 1, 2)
        # scale_atten_soft = scale_atten * scale_max_soft

        out += residual
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out

class Attention_block1(nn.Module):
    def __init__(self, width=64):
        super(Attention_block1, self).__init__()

        self.up_kwargs = up_kwargs

        self.psi = nn.Sequential(
            nn.Conv2d(width, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

        in_channels = [64, 128, 256, 512]
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            # nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            # nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            # nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            # nn.ReLU(inplace=True)
        )
    def forward(self, x1_in, x2_in, x3_in, x4_in):
        inputs = [x1_in, x2_in, x3_in, x4_in]

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]),self.conv3(inputs[-3]),self.conv2(inputs[-4])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feats[-4] = F.interpolate(feats[-4], (h, w), **self.up_kwargs)

        x1 = feats[-1]
        x2 = feats[-2]
        x3 = feats[-3]
        x4 = feats[-4]

        psi = self.relu(x1 + x2 + x3 + x4)
        psi = self.psi(psi)

        return x1_in * psi
class Attention_block2(nn.Module):
    def __init__(self, width=128):
        super(Attention_block2, self).__init__()

        self.up_kwargs = up_kwargs

        self.psi = nn.Sequential(
            nn.Conv2d(width, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

        in_channels = [64, 128, 256, 512]
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            # nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            # nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            # nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        ch_out = width
        self.downsample = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=2,padding=1,bias=True),#kernel_size3/2, s=2/maxpool
            nn.BatchNorm2d(ch_out),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x1_in, x2_in, x3_in, x4_in):
        inputs = [x1_in, x2_in, x3_in, x4_in]

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4])]
        _, _, h, w = feats[-2].size()
        # feats[-1] = F.interpolate(feats[-1], (h, w), **self.up_kwargs)
        # feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-1] = self.downsample(feats[-1])
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feats[-4] = F.interpolate(feats[-4], (h, w), **self.up_kwargs)

        x1 = feats[-1]
        x2 = feats[-2]
        x3 = feats[-3]
        x4 = feats[-4]

        psi = self.relu(x1 + x2 + x3 + x4)
        psi = self.psi(psi)

        return x2_in * psi
class Attention_block3(nn.Module):
    def __init__(self, width=256):
        super(Attention_block3, self).__init__()

        self.up_kwargs = up_kwargs

        self.psi = nn.Sequential(
            nn.Conv2d(width, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

        in_channels = [64, 128, 256, 512]
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            # nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            # nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )

        ch_out = width
        self.downsample1 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=2,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=2,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x1_in, x2_in, x3_in, x4_in):
        inputs = [x1_in, x2_in, x3_in, x4_in]

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4])]
        _, _, h, w = feats[-3].size()
        # feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        # feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feats[-1] = self.downsample2(feats[-1])
        feats[-2] = self.downsample1(feats[-2])
        feats[-4] = F.interpolate(feats[-4], (h, w), **self.up_kwargs)
        # feat = torch.cat(feats, dim=1)
        x1 = feats[-1]
        x3 = feats[-3]
        x4 = feats[-4]
        x2 = feats[-2]

        psi = self.relu(x1 + x2 + x3 + x4)
        psi = self.psi(psi)

        return x3_in * psi

class RenderTrans(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(RenderTrans, self).__init__()
        self.upsample = upsample

        self.conv3x3 = nn.Conv2d(channels_high, channels_high, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_high)

        self.conv1x1 = nn.Conv2d(channels_low, channels_high, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_high)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_low, channels_high, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_high)
        else:
            self.conv_reduction = nn.Conv2d(channels_low, channels_high, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_high)

        self.str_conv3x3_1 = nn.Conv2d(channels_low, channels_high, kernel_size=3, stride=2, padding=1, bias=False)
        self.str_conv3x3_2 = nn.Conv2d(channels_high, channels_high, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_reduction_2 = nn.BatchNorm2d(channels_high)
        self.str_conv3x3_3 = nn.Conv2d(channels_high, channels_high, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_reduction_3 = nn.BatchNorm2d(channels_high)
        self.relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Conv2d(channels_high*2, channels_high, kernel_size=1, padding=0, bias=False)

    def forward(self, x_high, x_low, down=1):
        b, c, h, w = x_low.shape
        x_low_gp = nn.AvgPool2d(x_low.shape[2:])(x_low).view(len(x_low), c, 1, 1)
        x_low_gp = self.conv1x1(x_low_gp)
        x_low_gp = self.bn_low(x_low_gp)
        x_low_gp = self.relu(x_low_gp)

        x_high_mask = self.conv3x3(x_high)
        x_high_mask = self.bn_high(x_high_mask)

        x_att = x_high_mask * x_low_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.str_conv3x3(x_low)) + x_att)
                # self.conv_cat(torch.cat([self.bn_upsample(self.str_conv3x3(x_low)), x_att], dim=1))
        if down == 1:
            out = self.relu(
                self.bn_reduction(self.str_conv3x3_1(x_low)) + x_att)
        elif down == 2:
            out = self.relu(
                self.bn_reduction_2(self.str_conv3x3_2(self.relu(self.bn_reduction(self.str_conv3x3_1(x_low))))) + x_att)
        elif down == 3:
            out = self.relu(
                self.bn_reduction_3(self.str_conv3x3_3(self.relu(self.bn_reduction_2(self.str_conv3x3_2(self.relu(self.bn_reduction(self.str_conv3x3_1(x_low)))))))) + x_att)
                # # self.conv_cat(torch.cat([self.bn_reduction(self.str_conv3x3(x_low)), x_att], dim=1))
        return out

class RenderTrans2(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(RenderTrans2, self).__init__()
        self.upsample = upsample

        self.conv3x3 = nn.Conv2d(channels_high, channels_high, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_high)

        self.conv1x1 = nn.Conv2d(channels_low, channels_high, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_high)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_low, channels_high, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_high)
        else:
            self.conv_reduction = nn.Conv2d(channels_low, channels_high, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_high)

        self.str_conv3x3_1 = nn.Conv2d(channels_low, channels_high, kernel_size=3, stride=2, padding=1, bias=False)
        self.str_conv3x3_2 = nn.Conv2d(channels_high, channels_high, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_reduction_2 = nn.BatchNorm2d(channels_high)
        self.str_conv3x3_3 = nn.Conv2d(channels_high, channels_high, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_reduction_3 = nn.BatchNorm2d(channels_high)
        self.relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Conv2d(channels_high*2, channels_high, kernel_size=1, padding=0, bias=False)
        self.d1 = nn.Sequential(
            nn.Conv2d(channels_low, channels_high, 3, 1, 1),
            nn.BatchNorm2d(channels_high),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.d2 = nn.Sequential(
            nn.Conv2d(channels_high, channels_high, 3, 1, 1),
            nn.BatchNorm2d(channels_high),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.d3 = nn.Sequential(
            nn.Conv2d(channels_high, channels_high, 3, 1, 1),
            nn.BatchNorm2d(channels_high),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x_high, x_low, down=1):
        b, c, h, w = x_low.shape
        x_low_gp = nn.AvgPool2d(x_low.shape[2:])(x_low).view(len(x_low), c, 1, 1)
        x_low_gp = self.conv1x1(x_low_gp)
        x_low_gp = self.bn_low(x_low_gp)
        x_low_gp = self.relu(x_low_gp)

        x_high_mask = self.conv3x3(x_high)
        x_high_mask = self.bn_high(x_high_mask)

        x_att = x_high_mask * x_low_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.str_conv3x3(x_low)) + x_att)
                # self.conv_cat(torch.cat([self.bn_upsample(self.str_conv3x3(x_low)), x_att], dim=1))
        if down == 1:
            out = self.relu(
                self.bn_reduction(self.d1(x_low)) + x_att)
        elif down == 2:
            out = self.relu(
                self.bn_reduction_2(self.d2(self.d1(x_low))) + x_att)
        elif down == 3:
            out = self.relu(
                self.bn_reduction_3(self.d3(self.d1(self.d1(x_low)))) + x_att)
                # # self.conv_cat(torch.cat([self.bn_reduction(self.str_conv3x3(x_low)), x_att], dim=1))
        return out

class GroundTrans(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='dot', dimension=2, bn_layer=True):
        super(GroundTrans, self).__init__()
        assert dimension in [1, 2, 3]
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d

        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d

        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x_low, x_high):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """
        batch_size = x_low.size(0)
        g_x = self.g(x_high).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x_low.view(batch_size, self.in_channels, -1)
            phi_x = x_high.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x_low).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x_high).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x_low).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_high).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N
        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, int(x_low.size()[2]), int(x_low.size()[3]))

        z = self.W_z(y)
        return z
class MixtureOfSoftMax(nn.Module):
    """"https://arxiv.org/pdf/1711.03953.pdf"""
    def __init__(self, n_mix, d_k, attn_dropout=0.1):
        super(MixtureOfSoftMax, self).__init__()
        self.temperature = np.power(d_k, 0.5)
        self.n_mix = n_mix
        self.att_drop = attn_dropout
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.d_k = d_k
        if n_mix > 1:
            self.weight = nn.Parameter(torch.Tensor(n_mix, d_k))
            std = np.power(n_mix, -0.5)
            self.weight.data.uniform_(-std, std)

    def forward(self, qt, kt, vt):
        B, d_k, N = qt.size()
        m = self.n_mix
        assert d_k == self.d_k
        d = d_k // m
        if m > 1:
            bar_qt = torch.mean(qt, 2, True)
            pi = self.softmax1(torch.matmul(self.weight, bar_qt)).view(B*m, 1, 1)

        q = qt.view(B*m, d, N).transpose(1, 2)
        N2 = kt.size(2)
        kt = kt.view(B*m, d, N2)
        v = vt.transpose(1, 2)
        attn = torch.bmm(q, kt)
        attn = attn / self.temperature
        attn = self.softmax2(attn)
        attn = self.dropout(attn)
        if m > 1:
            attn = (attn * pi).view(B, m, N, N2).sum(1)
        output = torch.bmm(attn, v)
        return output, attn
class SelfTrans(nn.Module):
    def __init__(self, n_head, n_mix, d_model, d_k, d_v,
                 norm_layer=BatchNorm2d, kq_transform='conv', value_transform='conv',
                 pooling=True, concat=False, dropout=0.1):
        super(SelfTrans, self).__init__()

        self.n_head = n_head
        self.n_mix = n_mix
        self.d_k = d_k
        self.d_v = d_v

        self.pooling = pooling
        self.concat = concat

        if self.pooling:
            self.pool = nn.AvgPool2d(3, 2, 1, count_include_pad=False)
        if kq_transform == 'conv':
            self.conv_qs = nn.Conv2d(d_model, n_head * d_k, 1)
            nn.init.normal_(self.conv_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        elif kq_transform == 'ffn':
            self.conv_qs = nn.Sequential(
                nn.Conv2d(d_model, n_head * d_k, 3, padding=1, bias=False),
                norm_layer(n_head * d_k),
                nn.ReLU(True),
                nn.Conv2d(n_head * d_k, n_head * d_k, 1),
            )
            nn.init.normal_(self.conv_qs[-1].weight, mean=0, std=np.sqrt(1.0 / d_k))
        elif kq_transform == 'dffn':
            self.conv_qs = nn.Sequential(
                nn.Conv2d(d_model, n_head * d_k, 3, padding=4, dilation=4, bias=False),
                norm_layer(n_head * d_k),
                nn.ReLU(True),
                nn.Conv2d(n_head * d_k, n_head * d_k, 1),
            )
            nn.init.normal_(self.conv_qs[-1].weight, mean=0, std=np.sqrt(1.0 / d_k))
        else:
            raise NotImplemented

        self.conv_ks = self.conv_qs
        if value_transform == 'conv':
            self.conv_vs = nn.Conv2d(d_model, n_head * d_v, 1)
        else:
            raise NotImplemented

        nn.init.normal_(self.conv_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = MixtureOfSoftMax(n_mix=n_mix, d_k=d_k)

        self.conv = nn.Conv2d(n_head * d_v, d_model, 1, bias=False)
        self.norm_layer = norm_layer(d_model)

    def forward(self, x):
        residual = x
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        b_, c_, h_, w_ = x.size()
        if self.pooling:
            qt = self.conv_ks(x).view(b_ * n_head, d_k, h_ * w_)
            kt = self.conv_ks(self.pool(x)).view(b_ * n_head, d_k, h_ * w_ // 4)
            vt = self.conv_vs(self.pool(x)).view(b_ * n_head, d_v, h_ * w_ // 4)
        else:
            kt = self.conv_ks(x).view(b_ * n_head, d_k, h_ * w_)
            qt = kt
            vt = self.conv_vs(x).view(b_ * n_head, d_v, h_ * w_)

        output, attn = self.attention(qt, kt, vt)

        output = output.transpose(1, 2).contiguous().view(b_, n_head * d_v, h_, w_)

        output = self.conv(output)
        if self.concat:
            output = torch.cat((self.norm_layer(output), residual), 1)
        else:
            output = self.norm_layer(output) + residual
        return output
class FPT(nn.Module):
    def __init__(self, feature_dim, with_norm='none', upsample_method='bilinear'):
        super(FPT, self).__init__()
        self.feature_dim = feature_dim
        assert upsample_method in ['nearest', 'bilinear']

        def interpolate(input):
            return F.interpolate(input, scale_factor=2, mode=upsample_method,
                                 align_corners=False if upsample_method == 'bilinear' else None)

        self.fpn_upsample = interpolate
        assert with_norm in ['group_norm', 'batch_norm', 'none']
        if with_norm == 'batch_norm':
            norm = nn.BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(num_channels):
                return nn.GroupNorm(32, num_channels)

            norm = group_norm
        self.st_p5 = SelfTrans(n_head=1, n_mix=2, d_model=feature_dim, d_k=feature_dim, d_v=feature_dim)
        self.st_p4 = SelfTrans(n_head=1, n_mix=2, d_model=feature_dim, d_k=feature_dim, d_v=feature_dim)
        self.st_p3 = SelfTrans(n_head=1, n_mix=2, d_model=feature_dim, d_k=feature_dim, d_v=feature_dim)
        self.st_p2 = SelfTrans(n_head=1, n_mix=2, d_model=feature_dim, d_k=feature_dim, d_v=feature_dim)

        self.gt_p4_p5 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2,
                                    bn_layer=True)
        self.gt_p3_p4 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2,
                                    bn_layer=True)
        self.gt_p3_p5 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2,
                                    bn_layer=True)
        self.gt_p2_p3 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2,
                                    bn_layer=True)
        self.gt_p2_p4 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2,
                                    bn_layer=True)
        self.gt_p2_p5 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2,
                                    bn_layer=True)

        self.rt_p5_p4 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p5_p3 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p5_p2 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p4_p3 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p4_p2 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p3_p2 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        # drop_block = DropBlock2D(block_size=3, drop_prob=0.2)

        if with_norm != 'none':
            self.fpn_p5_1x1 = nn.Sequential(*[nn.Conv2d(2048, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p4_1x1 = nn.Sequential(*[nn.Conv2d(1024, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p3_1x1 = nn.Sequential(*[nn.Conv2d(512, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p2_1x1 = nn.Sequential(*[nn.Conv2d(256, feature_dim, 1, bias=False), norm(feature_dim)])

            self.fpt_p5 = nn.Sequential(
                *[nn.Conv2d(feature_dim * 5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpt_p4 = nn.Sequential(
                *[nn.Conv2d(feature_dim * 5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpt_p3 = nn.Sequential(
                *[nn.Conv2d(feature_dim * 5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpt_p2 = nn.Sequential(
                *[nn.Conv2d(feature_dim * 5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
        else:
            self.fpn_p5_1x1 = nn.Conv2d(256, feature_dim, 1)
            self.fpn_p4_1x1 = nn.Conv2d(128, feature_dim, 1)
            self.fpn_p3_1x1 = nn.Conv2d(64, feature_dim, 1)
            self.fpn_p2_1x1 = nn.Conv2d(64, feature_dim, 1)

            self.fpt_p5 = nn.Conv2d(feature_dim * 5, 256, 3, padding=1)
            self.fpt_p4 = nn.Conv2d(feature_dim * 5, 128, 3, padding=1)
            self.fpt_p3 = nn.Conv2d(feature_dim * 5, 64, 3, padding=1)
            self.fpt_p2 = nn.Conv2d(feature_dim * 5, 64, 3, padding=1)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, res2, res3, res4, res5):
        fpn_p5_1 = self.fpn_p5_1x1(res5)
        fpn_p4_1 = self.fpn_p4_1x1(res4)
        fpn_p3_1 = self.fpn_p3_1x1(res3)
        fpn_p2_1 = self.fpn_p2_1x1(res2)
        fpt_p5_out = torch.cat((self.st_p5(fpn_p5_1), self.rt_p5_p4(fpn_p5_1, fpn_p4_1),
                                self.rt_p5_p3(fpn_p5_1, fpn_p3_1, 2), self.rt_p5_p2(fpn_p5_1, fpn_p2_1, 3), fpn_p5_1),
                               1)
        fpt_p4_out = torch.cat((self.st_p4(fpn_p4_1), self.rt_p4_p3(fpn_p4_1, fpn_p3_1),
                                self.rt_p4_p2(fpn_p4_1, fpn_p2_1, 2), self.gt_p4_p5(fpn_p4_1, fpn_p5_1), fpn_p4_1), 1)
        fpt_p3_out = torch.cat((self.st_p3(fpn_p3_1), self.rt_p3_p2(fpn_p3_1, fpn_p2_1),
                                self.gt_p3_p4(fpn_p3_1, fpn_p4_1), self.gt_p3_p5(fpn_p3_1, fpn_p5_1), fpn_p3_1), 1)
        fpt_p2_out = torch.cat((self.st_p2(fpn_p2_1), self.gt_p2_p3(fpn_p2_1, fpn_p3_1),
                                self.gt_p2_p4(fpn_p2_1, fpn_p4_1), self.gt_p2_p5(fpn_p2_1, fpn_p5_1), fpn_p2_1), 1)
        fpt_p5 = self.fpt_p5(fpt_p5_out)
        fpt_p4 = self.fpt_p4(fpt_p4_out)
        fpt_p3 = self.fpt_p3(fpt_p3_out)
        fpt_p2 = self.fpt_p2(fpt_p2_out)
        '''
        fpt_p5 = drop_block(self.fpt_p5(fpt_p5_out))
        fpt_p4 = drop_block(self.fpt_p4(fpt_p4_out))
        fpt_p3 = drop_block(self.fpt_p3(fpt_p3_out))
        fpt_p2 = drop_block(self.fpt_p2(fpt_p2_out))
        '''
        return fpt_p2, fpt_p3, fpt_p4, fpt_p5
class FPT2(nn.Module):
    def __init__(self, feature_dim, with_norm='none', upsample_method='bilinear'):
        super(FPT2, self).__init__()
        self.feature_dim = feature_dim
        assert upsample_method in ['nearest', 'bilinear']

        def interpolate(input):
            return F.interpolate(input, scale_factor=2, mode=upsample_method,
                                 align_corners=False if upsample_method == 'bilinear' else None)

        self.fpn_upsample = interpolate
        assert with_norm in ['group_norm', 'batch_norm', 'none']
        if with_norm == 'batch_norm':
            norm = nn.BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(num_channels):
                return nn.GroupNorm(32, num_channels)

            norm = group_norm
        self.st_p5 = SelfTrans(n_head=1, n_mix=2, d_model=feature_dim, d_k=feature_dim, d_v=feature_dim)
        self.st_p4 = SelfTrans(n_head=1, n_mix=2, d_model=feature_dim, d_k=feature_dim, d_v=feature_dim)
        self.st_p3 = SelfTrans(n_head=1, n_mix=2, d_model=feature_dim, d_k=feature_dim, d_v=feature_dim)
        self.st_p2 = SelfTrans(n_head=1, n_mix=2, d_model=feature_dim, d_k=feature_dim, d_v=feature_dim)

        self.gt_p4_p5 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2,
                                    bn_layer=True)
        self.gt_p3_p4 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2,
                                    bn_layer=True)
        self.gt_p3_p5 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2,
                                    bn_layer=True)
        self.gt_p2_p3 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2,
                                    bn_layer=True)
        self.gt_p2_p4 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2,
                                    bn_layer=True)
        self.gt_p2_p5 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2,
                                    bn_layer=True)

        self.rt_p5_p4 = RenderTrans2(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p5_p3 = RenderTrans2(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p5_p2 = RenderTrans2(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p4_p3 = RenderTrans2(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p4_p2 = RenderTrans2(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p3_p2 = RenderTrans2(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        # drop_block = DropBlock2D(block_size=3, drop_prob=0.2)

        if with_norm != 'none':
            self.fpn_p5_1x1 = nn.Sequential(*[nn.Conv2d(2048, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p4_1x1 = nn.Sequential(*[nn.Conv2d(1024, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p3_1x1 = nn.Sequential(*[nn.Conv2d(512, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p2_1x1 = nn.Sequential(*[nn.Conv2d(256, feature_dim, 1, bias=False), norm(feature_dim)])

            self.fpt_p5 = nn.Sequential(
                *[nn.Conv2d(feature_dim * 5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpt_p4 = nn.Sequential(
                *[nn.Conv2d(feature_dim * 5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpt_p3 = nn.Sequential(
                *[nn.Conv2d(feature_dim * 5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpt_p2 = nn.Sequential(
                *[nn.Conv2d(feature_dim * 5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
        else:
            self.fpn_p5_1x1 = nn.Conv2d(256, feature_dim, 1)
            self.fpn_p4_1x1 = nn.Conv2d(128, feature_dim, 1)
            self.fpn_p3_1x1 = nn.Conv2d(64, feature_dim, 1)
            self.fpn_p2_1x1 = nn.Conv2d(64, feature_dim, 1)

            self.fpt_p5 = nn.Conv2d(feature_dim * 5, 256, 3, padding=1)
            self.fpt_p4 = nn.Conv2d(feature_dim * 5, 128, 3, padding=1)
            self.fpt_p3 = nn.Conv2d(feature_dim * 5, 64, 3, padding=1)
            self.fpt_p2 = nn.Conv2d(feature_dim * 5, 64, 3, padding=1)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, res2, res3, res4, res5):
        fpn_p5_1 = self.fpn_p5_1x1(res5)
        fpn_p4_1 = self.fpn_p4_1x1(res4)
        fpn_p3_1 = self.fpn_p3_1x1(res3)
        fpn_p2_1 = self.fpn_p2_1x1(res2)
        fpt_p5_out = torch.cat((self.st_p5(fpn_p5_1), self.rt_p5_p4(fpn_p5_1, fpn_p4_1),
                                self.rt_p5_p3(fpn_p5_1, fpn_p3_1, 2), self.rt_p5_p2(fpn_p5_1, fpn_p2_1, 3), fpn_p5_1),
                               1)
        fpt_p4_out = torch.cat((self.st_p4(fpn_p4_1), self.rt_p4_p3(fpn_p4_1, fpn_p3_1),
                                self.rt_p4_p2(fpn_p4_1, fpn_p2_1, 2), self.gt_p4_p5(fpn_p4_1, fpn_p5_1), fpn_p4_1), 1)
        fpt_p3_out = torch.cat((self.st_p3(fpn_p3_1), self.rt_p3_p2(fpn_p3_1, fpn_p2_1),
                                self.gt_p3_p4(fpn_p3_1, fpn_p4_1), self.gt_p3_p5(fpn_p3_1, fpn_p5_1), fpn_p3_1), 1)
        fpt_p2_out = torch.cat((self.st_p2(fpn_p2_1), self.gt_p2_p3(fpn_p2_1, fpn_p3_1),
                                self.gt_p2_p4(fpn_p2_1, fpn_p4_1), self.gt_p2_p5(fpn_p2_1, fpn_p5_1), fpn_p2_1), 1)
        fpt_p5 = self.fpt_p5(fpt_p5_out)
        fpt_p4 = self.fpt_p4(fpt_p4_out)
        fpt_p3 = self.fpt_p3(fpt_p3_out)
        fpt_p2 = self.fpt_p2(fpt_p2_out)
        '''
        fpt_p5 = drop_block(self.fpt_p5(fpt_p5_out))
        fpt_p4 = drop_block(self.fpt_p4(fpt_p4_out))
        fpt_p3 = drop_block(self.fpt_p3(fpt_p3_out))
        fpt_p2 = drop_block(self.fpt_p2(fpt_p2_out))
        '''
        return fpt_p2, fpt_p3, fpt_p4, fpt_p5

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage
class ASFF_ddw(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF_ddw, self).__init__()
        self.level = level
        self.dim = [256, 128, 64, 32]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(128, self.inter_dim, 3, 2)
            self.stride_level_2 = nn.Sequential(
                add_conv(64, self.inter_dim, 3, 2),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.stride_level_3 = nn.Sequential(
                add_conv(32, 64, 3, 2),
                add_conv(64, self.inter_dim, 3, 2),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.expand = add_conv(self.inter_dim, 256, 3, 1)  # ????

        elif level == 1:
            self.stride_level_0 = nn.Sequential(
                add_conv(256, self.inter_dim, 1, 1),
                nn.Upsample(size=(56, 80), mode='bilinear')
            )
            self.stride_level_2 = add_conv(64, self.inter_dim, 3, 2)
            self.stride_level_3 = nn.Sequential(
                add_conv(32, self.inter_dim, 3, 2),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.expand = add_conv(self.inter_dim, 64, 3, 1)

        elif level == 2:
            self.stride_level_0 = nn.Sequential(
                add_conv(256, self.inter_dim, 1, 1),
                nn.Upsample(size=(112, 160), mode='bilinear')
            )
            self.stride_level_1 = nn.Sequential(
                add_conv(128, self.inter_dim, 1, 1),
                nn.Upsample(size=(112, 160), mode='bilinear')
            )
            self.stride_level_3 = add_conv(32, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 32, 3, 1)

        elif level == 3:
            self.stride_level_0 = nn.Sequential(
                add_conv(256, self.inter_dim, 1, 1),
                nn.Upsample(size=(224, 320), mode='bilinear')
            )
            self.stride_level_1 = nn.Sequential(
                add_conv(128, self.inter_dim, 1, 1),
                nn.Upsample(size=(224, 320), mode='bilinear')
            )
            self.stride_level_2 = nn.Sequential(
                add_conv(64, self.inter_dim, 1, 1),
                nn.Upsample(size=(224, 320), mode='bilinear')
            )
            self.expand = add_conv(self.inter_dim, 32, 3, 1)

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2, x_level_3):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level == 2:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = x_level_2
            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level == 3:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
            level_3_resized = x_level_3

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        level_3_weight_v = self.weight_level_3(level_3_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:3, :, :] + \
                            level_3_resized * levels_weight[:, 3:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

class Scale_Aware(nn.Module):
    def __init__(self, in_channels):
        super(Scale_Aware, self).__init__()

        # self.bn = nn.ModuleList([nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels)])
        self.conv1x1 = nn.ModuleList(
            [nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0),
             nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0)])
        self.conv3x3_1 = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1),
             nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1)])
        self.conv3x3_2 = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels // 2, out_channels=2, dilation=1, kernel_size=3, padding=1),
             nn.Conv2d(in_channels=in_channels // 2, out_channels=2, dilation=1, kernel_size=3, padding=1)])
        self.conv_last = ConvBnRelu(in_planes=in_channels, out_planes=in_channels, ksize=1, stride=1, pad=0, dilation=1)

        self.relu = nn.ReLU()
    def forward(self, x_l, x_h):
        feat = torch.cat([x_l, x_h], dim=1)
        # feat=feat_cat.detach()
        feat = self.relu(self.conv1x1[0](feat))
        feat = self.relu(self.conv3x3_1[0](feat))
        att = self.conv3x3_2[0](feat)
        att = F.softmax(att, dim=1)

        att_1 = att[:, 0, :, :].unsqueeze(1)
        att_2 = att[:, 1, :, :].unsqueeze(1)

        fusion_1_2 = att_1 * x_l + att_2 * x_h
        return fusion_1_2


class GPG_3(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None,norm_layer=nn.BatchNorm2d):
        super(GPG_3, self).__init__()
        self.up_kwargs = up_kwargs
        

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(3*width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))
        
        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat)], dim=1)
        feat=self.conv_out(feat)
        return feat
class GPG_4(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None,norm_layer=nn.BatchNorm2d):
        super(GPG_4, self).__init__()
        self.up_kwargs = up_kwargs
        

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(2*width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))
        
        self.dilation1 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat)], dim=1)
        feat=self.conv_out(feat)
        return feat
class GPG_2(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None,norm_layer=nn.BatchNorm2d):
        super(GPG_2, self).__init__()
        self.up_kwargs = up_kwargs
        

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(
            nn.Conv2d(4*width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))
        
        self.dilation1 = nn.Sequential(SeparableConv2d(4*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(4*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(4*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(4*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]),self.conv3(inputs[-3]),self.conv2(inputs[-4])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feats[-4] = F.interpolate(feats[-4], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)
        feat=self.conv_out(feat)
        return feat
class GPG_3_my(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(GPG_3_my, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(
            nn.Conv2d(4 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))

        self.dilation1 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
        ch_out = width
        self.downsample = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, *inputs):

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4])]
        _, _, h, w = feats[-2].size()
        # feats[-1] = F.interpolate(feats[-1], (h, w), **self.up_kwargs)
        # feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-1] = self.downsample(feats[-1])
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feats[-4] = F.interpolate(feats[-4], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)],
                         dim=1)
        feat = self.conv_out(feat)
        return feat
class GPG_4_my(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(GPG_4_my, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(
            nn.Conv2d(4 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))

        self.dilation1 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
        ch_out = width
        self.downsample1 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=2,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, *inputs):

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4])]
        _, _, h, w = feats[-3].size()
        # feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        # feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feats[-1] = self.downsample2(feats[-1])
        feats[-2] = self.downsample1(feats[-2])
        feats[-4] = F.interpolate(feats[-4], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)],
                         dim=1)
        feat = self.conv_out(feat)
        return feat

class BaseNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BaseNetHead, self).__init__()
        if is_aux:
            self.conv_1x1_3x3=nn.Sequential(
                ConvBnRelu(in_planes, 64, 1, 1, 0,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False),
                ConvBnRelu(64, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False))
        else:
            self.conv_1x1_3x3=nn.Sequential(
                ConvBnRelu(in_planes, 32, 1, 1, 0,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False),
                ConvBnRelu(32, 32, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False))
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1_2 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1_2 = nn.Conv2d(32, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)
        fm = self.conv_1x1_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1_2(fm)
        return output
class SAPblock(nn.Module):
    def __init__(self, in_channels):
        super(SAPblock, self).__init__()
        self.conv3x3=nn.Conv2d(in_channels=in_channels, out_channels=in_channels,dilation=1,kernel_size=3, padding=1)
        
        self.bn=nn.ModuleList([nn.BatchNorm2d(in_channels),nn.BatchNorm2d(in_channels),nn.BatchNorm2d(in_channels)]) 
        self.conv1x1=nn.ModuleList([nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0),
                                    nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0)])
        self.conv3x3_1=nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,dilation=1,kernel_size=3, padding=1),
                                      nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,dilation=1,kernel_size=3, padding=1)])
        self.conv3x3_2=nn.ModuleList([nn.Conv2d(in_channels=in_channels//2, out_channels=2,dilation=1,kernel_size=3, padding=1),
                                      nn.Conv2d(in_channels=in_channels//2, out_channels=2,dilation=1,kernel_size=3, padding=1)])
        self.conv_last=ConvBnRelu(in_planes=in_channels,out_planes=in_channels,ksize=1,stride=1,pad=0,dilation=1)



        self.gamma = nn.Parameter(torch.zeros(1))
    
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):

        x_size= x.size()

        branches_1=self.conv3x3(x)
        branches_1=self.bn[0](branches_1)

        branches_2=F.conv2d(x,self.conv3x3.weight,padding=2,dilation=2)#share weight
        branches_2=self.bn[1](branches_2)

        branches_3=F.conv2d(x,self.conv3x3.weight,padding=4,dilation=4)#share weight
        branches_3=self.bn[2](branches_3)

        feat=torch.cat([branches_1,branches_2],dim=1)
        # feat=feat_cat.detach()
        feat=self.relu(self.conv1x1[0](feat))
        feat=self.relu(self.conv3x3_1[0](feat))
        att=self.conv3x3_2[0](feat)
        att = F.softmax(att, dim=1)
        
        att_1=att[:,0,:,:].unsqueeze(1)
        att_2=att[:,1,:,:].unsqueeze(1)

        fusion_1_2=att_1*branches_1+att_2*branches_2



        feat1=torch.cat([fusion_1_2,branches_3],dim=1)
        # feat=feat_cat.detach()
        feat1=self.relu(self.conv1x1[0](feat1))
        feat1=self.relu(self.conv3x3_1[0](feat1))
        att1=self.conv3x3_2[0](feat1)
        att1 = F.softmax(att1, dim=1)
        
        att_1_2=att1[:,0,:,:].unsqueeze(1)
        att_3=att1[:,1,:,:].unsqueeze(1)


        ax=self.relu(self.gamma*(att_1_2*fusion_1_2+att_3*branches_3)+(1-self.gamma)*x)
        ax=self.conv_last(ax)

        return ax
class DecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d,scale=2,relu=True,last=False):
        super(DecoderBlock, self).__init__()
       

        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
       
        self.sap=SAPblock(in_planes)
        self.scale=scale
        self.last=last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.last==False:
            x = self.conv_3x3(x)
            # x=self.sap(x)
        if self.scale>1:
            x=F.interpolate(x,scale_factor=self.scale,mode='bilinear',align_corners=True)
        x=self.conv_1x1(x)
        return x

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x
    
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x
    
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)

        return inputs

class Local_Channel(nn.Module):
    def __init__(self, in_channel):
        super(Local_Channel, self).__init__()
        self.attn = nn.Sequential(GlobalAvgPool2d(), nn.Conv2d(in_channel, in_channel, 1), nn.Sigmoid())
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        attn_map = self.attn(x)
        return x * (1 - self.gamma) + attn_map * x * self.gamma, attn_map

class Local_Spatial(nn.Module):
    def __init__(self, in_channel, mid_channel):
        super(Local_Spatial, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, mid_channel, 1)
        self.branch1 = nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, 1)
        self.branch2 = nn.Conv2d(mid_channel, mid_channel, 3, 1, 2, 2)
        self.branch3 = nn.Conv2d(mid_channel, mid_channel, 3, 1, 3, 3)
        self.attn = nn.Conv2d(3 * mid_channel, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        mid = self.conv1x1(x)
        branch1 = self.branch1(mid)
        branch2 = self.branch2(mid)
        branch3 = self.branch3(mid)
        branch123 = torch.cat([branch1, branch2, branch3], dim=1)
        attn_map = self.attn(branch123)
        return x * (1 - self.gamma) + attn_map * x * self.gamma, attn_map

nonlinearity = partial(F.relu, inplace=True)

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up3(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2, X3):
        x1 = self.up(x1)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1, X3], dim=1)
        return self.conv(x)

class CBAM_Module(nn.Module):
    def __init__(self, channels=512, reduction=2):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial Attention module
        x = module_input * x
        module_input = x
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x

class CBAM_Module2(nn.Module):
    def __init__(self, channels=512, reduction=2):
        super(CBAM_Module2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial Attention module
        x = module_input * x + module_input
        module_input = x
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x + module_input
        return x

class Bridge(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, in_channels_3, mid_channels):
        super(Bridge, self).__init__()
        self.mid_channels = mid_channels
        self.conv_qk1 = nn.Conv2d(in_channels_1, mid_channels, 1, 1, 0)
        self.conv_qk2 = nn.Conv2d(in_channels_2, mid_channels, 1, 1, 0)
        self.conv_qk3 = nn.Conv2d(in_channels_3, mid_channels, 1, 1, 0)

        self.conv_v1 = nn.Conv2d(in_channels_1, mid_channels, 1, 1, 0)
        self.conv_v2 = nn.Conv2d(in_channels_2, mid_channels, 1, 1, 0)
        self.conv_v3 = nn.Conv2d(in_channels_3, mid_channels, 1, 1, 0)

        self.conv_out1 = nn.Conv2d(2 * mid_channels + in_channels_1, in_channels_1, 1, 1, 0)
        self.conv_out2 = nn.Conv2d(2 * mid_channels + in_channels_2, in_channels_2, 1, 1, 0)
        self.conv_out3 = nn.Conv2d(2 * mid_channels + in_channels_3, in_channels_3, 1, 1, 0)

    def forward(self, f1, f2, f3):
        batch_size = f1.size(0)
        qk1 = self.conv_qk1(f1).view(batch_size, self.mid_channels, -1)
        qk2 = self.conv_qk2(f2).view(batch_size, self.mid_channels, -1)
        qk3 = self.conv_qk3(f3).view(batch_size, self.mid_channels, -1)

        v1 = self.conv_v1(f1).view(batch_size, self.mid_channels, -1)
        v2 = self.conv_v2(f2).view(batch_size, self.mid_channels, -1)
        v3 = self.conv_v3(f3).view(batch_size, self.mid_channels, -1)

        sim12 = torch.matmul(qk1.permute(0, 2, 1), qk2)
        sim23 = torch.matmul(qk2.permute(0, 2, 1), qk3)
        sim31 = torch.matmul(qk3.permute(0, 2, 1), qk1)

        attn12 = F.softmax(sim12, dim=-1)
        attn21 = F.softmax(sim12.permute(0, 2, 1), dim=-1)
        attn23 = F.softmax(sim23, dim=-1)
        attn32 = F.softmax(sim23.permute(0, 2, 1), dim=-1)
        attn31 = F.softmax(sim31, dim=-1)
        attn13 = F.softmax(sim31.permute(0, 2, 1), dim=-1)

        y12 = torch.matmul(v1, attn12).contiguous()
        y13 = torch.matmul(v1, attn13).contiguous()
        y21 = torch.matmul(v2, attn21).contiguous()
        y23 = torch.matmul(v2, attn23).contiguous()
        y31 = torch.matmul(v3, attn31).contiguous()
        y32 = torch.matmul(v3, attn32).contiguous()

        y12 = y12.view(batch_size, self.mid_channels, int(f2.size()[2]), int(f2.size()[3]))
        y13 = y13.view(batch_size, self.mid_channels, int(f3.size()[2]), int(f3.size()[3]))
        y21 = y21.view(batch_size, self.mid_channels, int(f1.size()[2]), int(f1.size()[3]))
        y23 = y23.view(batch_size, self.mid_channels, int(f3.size()[2]), int(f3.size()[3]))
        y31 = y31.view(batch_size, self.mid_channels, int(f1.size()[2]), int(f1.size()[3]))
        y32 = y32.view(batch_size, self.mid_channels, int(f2.size()[2]), int(f2.size()[3]))

        out1 = self.conv_out1(torch.cat([f1, y31, y21], dim=1))
        out2 = self.conv_out2(torch.cat([f2, y12, y32], dim=1))
        out3 = self.conv_out3(torch.cat([f3, y23, y13], dim=1))

        return out1, out2, out3

class Bridge4(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, in_channels_3, in_channels_4, mid_channels):
        super(Bridge4, self).__init__()
        self.mid_channels = mid_channels
        self.conv_qk1 = nn.Conv2d(in_channels_1, mid_channels, 1, 1, 0)
        self.conv_qk2 = nn.Conv2d(in_channels_2, mid_channels, 1, 1, 0)
        self.conv_qk3 = nn.Conv2d(in_channels_3, mid_channels, 1, 1, 0)
        self.conv_qk4 = nn.Conv2d(in_channels_4, mid_channels, 1, 1, 0)

        self.conv_v1 = nn.Conv2d(in_channels_1, mid_channels, 1, 1, 0)
        self.conv_v2 = nn.Conv2d(in_channels_2, mid_channels, 1, 1, 0)
        self.conv_v3 = nn.Conv2d(in_channels_3, mid_channels, 1, 1, 0)
        self.conv_v4 = nn.Conv2d(in_channels_4, mid_channels, 1, 1, 0)

        self.conv_out1 = nn.Conv2d(3 * mid_channels + in_channels_1, in_channels_1, 1, 1, 0)
        self.conv_out2 = nn.Conv2d(3 * mid_channels + in_channels_2, in_channels_2, 1, 1, 0)
        self.conv_out3 = nn.Conv2d(3 * mid_channels + in_channels_3, in_channels_3, 1, 1, 0)
        self.conv_out4 = nn.Conv2d(3 * mid_channels + in_channels_4, in_channels_4, 1, 1, 0)


    def forward(self, f1, f2, f3, f4):
        batch_size = f1.size(0)
        qk1 = self.conv_qk1(f1).view(batch_size, self.mid_channels, -1)
        qk2 = self.conv_qk2(f2).view(batch_size, self.mid_channels, -1)
        qk3 = self.conv_qk3(f3).view(batch_size, self.mid_channels, -1)
        qk4 = self.conv_qk4(f4).view(batch_size, self.mid_channels, -1)

        v1 = self.conv_v1(f1).view(batch_size, self.mid_channels, -1)
        v2 = self.conv_v2(f2).view(batch_size, self.mid_channels, -1)
        v3 = self.conv_v3(f3).view(batch_size, self.mid_channels, -1)
        v4 = self.conv_v4(f4).view(batch_size, self.mid_channels, -1)


        sim12 = torch.matmul(qk1.permute(0, 2, 1), qk2)
        sim23 = torch.matmul(qk2.permute(0, 2, 1), qk3)
        sim31 = torch.matmul(qk3.permute(0, 2, 1), qk1)
        sim31 = torch.matmul(qk3.permute(0, 2, 1), qk1)

        attn12 = F.softmax(sim12, dim=-1)
        attn21 = F.softmax(sim12.permute(0, 2, 1), dim=-1)
        attn23 = F.softmax(sim23, dim=-1)
        attn32 = F.softmax(sim23.permute(0, 2, 1), dim=-1)
        attn31 = F.softmax(sim31, dim=-1)
        attn13 = F.softmax(sim31.permute(0, 2, 1), dim=-1)

        y12 = torch.matmul(v1, attn12).contiguous()
        y13 = torch.matmul(v1, attn13).contiguous()
        y21 = torch.matmul(v2, attn21).contiguous()
        y23 = torch.matmul(v2, attn23).contiguous()
        y31 = torch.matmul(v3, attn31).contiguous()
        y32 = torch.matmul(v3, attn32).contiguous()

        y12 = y12.view(batch_size, self.mid_channels, int(f2.size()[2]), int(f2.size()[3]))
        y13 = y13.view(batch_size, self.mid_channels, int(f3.size()[2]), int(f3.size()[3]))
        y21 = y21.view(batch_size, self.mid_channels, int(f1.size()[2]), int(f1.size()[3]))
        y23 = y23.view(batch_size, self.mid_channels, int(f3.size()[2]), int(f3.size()[3]))
        y31 = y31.view(batch_size, self.mid_channels, int(f1.size()[2]), int(f1.size()[3]))
        y32 = y32.view(batch_size, self.mid_channels, int(f2.size()[2]), int(f2.size()[3]))

        out1 = self.conv_out1(torch.cat([f1, y31, y21], dim=1))
        out2 = self.conv_out2(torch.cat([f2, y12, y32], dim=1))
        out3 = self.conv_out3(torch.cat([f3, y23, y13], dim=1))

        return out1, out2, out3

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)





class msca_net(nn.Module):
    def __init__(self, classes=2, channels=3, ccm=True, norm_layer=nn.BatchNorm2d, is_training=True, expansion=2,
                 base_channel=32):
        super(msca_net, self).__init__()
        self.backbone = models.resnet34(pretrained=True)
        # self.backbone =resnet34(pretrained=False)
        self.expansion = expansion
        self.base_channel = base_channel
        if self.expansion == 4 and self.base_channel == 64:
            expan = [512, 1024, 2048]
            spatial_ch = [128, 256]
        elif self.expansion == 4 and self.base_channel == 32:
            expan = [256, 512, 1024]
            spatial_ch = [32, 128]
            conv_channel_up = [256, 384, 512]
        elif self.expansion == 2 and self.base_channel == 32:
            expan = [128, 256, 512]
            spatial_ch = [64, 64]
            conv_channel_up = [128, 256, 512]

        conv_channel = expan[0]

        self.is_training = is_training
        # self.sap = SAPblock(expan[-1])

        # self.decoder5 = DecoderBlock(expan[-1], expan[-2], relu=False, last=True)  # 256
        # self.decoder4 = DecoderBlock(expan[-2], expan[-3], relu=False)  # 128
        # self.decoder3 = DecoderBlock(expan[-3], spatial_ch[-1], relu=False)  # 64
        # self.decoder2 = DecoderBlock(spatial_ch[-1], spatial_ch[-2])  # 32

        bilinear =True
        factor = 2
        self.up1 = Up(768, 512 // factor, bilinear)
        self.up2 = Up(384, 256 // factor, bilinear)
        self.up3 = Up(192, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.main_head = BaseNetHead(64, classes, 2,
                                     is_aux=False, norm_layer=norm_layer)

        # self.relu = nn.ReLU()

        # self.fpt = FPT(feature_dim=4)

        filters = [64, 64, 128, 256]
        self.out_size = (112, 160)
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=64, scale_factor=self.out_size)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=64, scale_factor=self.out_size)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=64, scale_factor=self.out_size)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=64, kernel_size=1)

        self.sw1 = Scale_Aware(in_channels=64)
        self.sw2 = Scale_Aware(in_channels=64)
        self.sw3 = Scale_Aware(in_channels=64)

        self.affinity_attention = AffinityAttention2(512)
        self.cbam = CBAM_Module2()
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.bridge = Bridge(64, 128, 256, 64)
    def forward(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)  # 1/2  64

        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)  # 1/4   64
        c3 = self.backbone.layer2(c2)  # 1/8   128
        c4 = self.backbone.layer3(c3)  # 1/16   256
        c5 = self.backbone.layer4(c4)  # 1/32   512
        # d_bottom=self.bottom(c5)

        # m1, m2, m3, m4 = self.fpt(c1, c2, c3, c4)
        m2, m3, m4 = self.bridge(c2, c3, c4)

        # c5 = self.sap(c5)
        attention = self.affinity_attention(c5)
        cbam_attn = self.cbam(c5)
        # l_channel, _ = self.l_channel(c5)
        # l_spatial, _ = self.l_spatial(c5)
        c5 = self.gamma1 * attention + self.gamma2 * cbam_attn + self.gamma3 * c5#多种并行方式， 用不用bn relu, 用不用scale aware

        # d5=d_bottom+c5           #512

        # d4 = self.relu(self.decoder5(c5) + m4)  # 256
        # d3 = self.relu(self.decoder4(d4) + m3)  # 128
        # d2 = self.relu(self.decoder3(d3) + m2)  # 64
        # d1 = self.decoder2(d2) + m1  # 32
        d4 = self.up1(c5, m4)
        d3 = self.up2(d4, m3)
        d2 = self.up3(d3, m2)
        d1 = self.up4(d2, c1)

        dsv4 = self.dsv4(d4)
        dsv3 = self.dsv3(d3)
        dsv2 = self.dsv2(d2)
        dsv1 = self.dsv1(d1)

        dsv43 = self.sw1(dsv4, dsv3)
        dsv432 = self.sw2(dsv43, dsv2)
        dsv4321 = self.sw3(dsv432, dsv1)

        main_out = self.main_head(dsv4321)

        final = F.sigmoid(main_out)

        return final

class msca_net_with_heatmap_output(nn.Module):
    def __init__(self, classes=2, channels=3, ccm=True, norm_layer=nn.BatchNorm2d, is_training=True, expansion=2,
                 base_channel=32):
        super(msca_net_with_heatmap_output, self).__init__()
        self.backbone = models.resnet34(pretrained=True)
        # self.backbone =resnet34(pretrained=False)
        self.expansion = expansion
        self.base_channel = base_channel
        if self.expansion == 4 and self.base_channel == 64:
            expan = [512, 1024, 2048]
            spatial_ch = [128, 256]
        elif self.expansion == 4 and self.base_channel == 32:
            expan = [256, 512, 1024]
            spatial_ch = [32, 128]
            conv_channel_up = [256, 384, 512]
        elif self.expansion == 2 and self.base_channel == 32:
            expan = [128, 256, 512]
            spatial_ch = [64, 64]
            conv_channel_up = [128, 256, 512]

        conv_channel = expan[0]

        self.is_training = is_training
        # self.sap = SAPblock(expan[-1])

        # self.decoder5 = DecoderBlock(expan[-1], expan[-2], relu=False, last=True)  # 256
        # self.decoder4 = DecoderBlock(expan[-2], expan[-3], relu=False)  # 128
        # self.decoder3 = DecoderBlock(expan[-3], spatial_ch[-1], relu=False)  # 64
        # self.decoder2 = DecoderBlock(spatial_ch[-1], spatial_ch[-2])  # 32

        bilinear =True
        factor = 2
        self.up1 = Up(768, 512 // factor, bilinear)
        self.up2 = Up(384, 256 // factor, bilinear)
        self.up3 = Up(192, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.main_head = BaseNetHead(64, classes, 2,
                                     is_aux=False, norm_layer=norm_layer)

        # self.relu = nn.ReLU()

        # self.fpt = FPT(feature_dim=4)

        filters = [64, 64, 128, 256]
        self.out_size = (112, 160)
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=64, scale_factor=self.out_size)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=64, scale_factor=self.out_size)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=64, scale_factor=self.out_size)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=64, kernel_size=1)

        self.sw1 = Scale_Aware(in_channels=64)
        self.sw2 = Scale_Aware(in_channels=64)
        self.sw3 = Scale_Aware(in_channels=64)

        self.affinity_attention = AffinityAttention2(512)
        self.cbam = CBAM_Module2()
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.bridge = Bridge(64, 128, 256, 64)
    def forward(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)  # 1/2  64

        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)  # 1/4   64
        c3 = self.backbone.layer2(c2)  # 1/8   128
        c4 = self.backbone.layer3(c3)  # 1/16   256
        c5 = self.backbone.layer4(c4)  # 1/32   512
        # d_bottom=self.bottom(c5)

        # m1, m2, m3, m4 = self.fpt(c1, c2, c3, c4)
        m2, m3, m4 = self.bridge(c2, c3, c4)

        # c5 = self.sap(c5)
        attention = self.affinity_attention(c5)
        cbam_attn = self.cbam(c5)
        # l_channel, _ = self.l_channel(c5)
        # l_spatial, _ = self.l_spatial(c5)
        c5 = self.gamma1 * attention + self.gamma2 * cbam_attn + self.gamma3 * c5#多种并行方式， 用不用bn relu, 用不用scale aware

        # d5=d_bottom+c5           #512

        # d4 = self.relu(self.decoder5(c5) + m4)  # 256
        # d3 = self.relu(self.decoder4(d4) + m3)  # 128
        # d2 = self.relu(self.decoder3(d3) + m2)  # 64
        # d1 = self.decoder2(d2) + m1  # 32
        d4 = self.up1(c5, m4)
        d3 = self.up2(d4, m3)
        d2 = self.up3(d3, m2)
        d1 = self.up4(d2, c1)

        dsv4 = self.dsv4(d4)
        dsv3 = self.dsv3(d3)
        dsv2 = self.dsv2(d2)
        dsv1 = self.dsv1(d1)

        dsv43 = self.sw1(dsv4, dsv3)
        dsv432 = self.sw2(dsv43, dsv2)
        dsv4321 = self.sw3(dsv432, dsv1)

        main_out = self.main_head(dsv4321)

        final = F.sigmoid(main_out)

        att_cacs_map = dsv4321.cpu().detach().numpy().astype(np.float) #Change to the features you want to visualize
        att_cacs_map = np.mean(att_cacs_map, axis=1)
        att_cacs_map = ndimage.interpolation.zoom(att_cacs_map, [1.0, 224 / att_cacs_map.shape[1],
                                                              320 / att_cacs_map.shape[2]], order=1)   # [1, 1024, 224, 320]

        return final, att_cacs_map
