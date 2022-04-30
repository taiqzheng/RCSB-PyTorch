import torch
import numpy as np
import torch.nn as nn
from torch.fft import fft2, fftshift, ifft2, ifftshift
#from util.utils import *
import torch.nn.functional as F
from option import get_option

opt = get_option()


class Frequency_Edge_Module(nn.Module):
    def __init__(self, radius, channel):
        super(Frequency_Edge_Module, self).__init__()
        self.radius = radius
        self.UAM = UnionAttentionModule(channel, only_channel_tracing=True)

        # DWS + DWConv
        self.DWSConv = DWSConv(channel, 1, kernel=3, padding=1, kernels_per_layer=1)
        self.DWConv1 = nn.Sequential(
            DWConv(channel, channel, kernel=1, padding=0, dilation=1),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.DWConv2 = nn.Sequential(
            DWConv(channel, channel, kernel=3, padding=1, dilation=1),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.DWConv3 = nn.Sequential(
            DWConv(channel, channel, kernel=3, padding=3, dilation=3),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.DWConv4 = nn.Sequential(
            DWConv(channel, channel, kernel=3, padding=5, dilation=5),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.conv = BasicConv2d(channel, 1, 1)

    def distance(self, i, j, imageSize, r):
        dis = np.sqrt((i - imageSize / 2) ** 2 + (j - imageSize / 2) ** 2)
        if dis < r:
            return 1.0
        else:
            return 0

    def mask_radial(self, img, r):
        batch, channels, rows, cols = img.shape
        mask = torch.zeros((rows, cols), dtype=torch.float32)
        for i in range(rows):
            for j in range(cols):
                mask[i, j] = self.distance(i, j, imageSize=rows, r=r)
        return mask

    def forward(self, x):
        """
        Input:
            The first encoder block representation: (B, C, H, W)
        Returns:
            Edge refined representation: X + edge (B, C, H, W)
        """
        x_fft = fft2(x, dim=(-2, -1))
        x_fft = fftshift(x_fft)

        # Mask -> low, high separate
        mask = self.mask_radial(img=x, r=self.radius).cuda()
        high_frequency = x_fft * (1 - mask)
        x_fft = ifftshift(high_frequency)
        x_fft = ifft2(x_fft, dim=(-2, -1))
        x_H = torch.abs(x_fft)
        # print('x_H.shape = {}'.format(x_H.shape))

        # x_H, _ = self.UAM.Channel_Tracer(x_H)
        # edge_maks = self.DWSConv(x_H)
        # skip = edge_maks.clone()

        # edge_maks = torch.cat([self.DWConv1(edge_maks), self.DWConv2(edge_maks),
        #                        self.DWConv3(edge_maks), self.DWConv4(edge_maks)], dim=1) + skip
        # edge = torch.relu(self.conv(edge_maks))
        # print('edge.shape = {}'.format(edge.shape))

        # x = x + edge  # Feature + Masked Edge information
        x = x + x_H
        edge = x_H
        return x, edge

class GlobalAvgPool(nn.Module):
    def __init__(self, flatten=False):
        super(GlobalAvgPool, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class UnionAttentionModule(nn.Module):
    def __init__(self, n_channels, only_channel_tracing=False):
        super(UnionAttentionModule, self).__init__()
        self.GAP = GlobalAvgPool()
        self.confidence_ratio = opt.gamma
        self.bn = nn.BatchNorm2d(n_channels)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.Dropout3d(self.confidence_ratio)
        )
        self.channel_q = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.channel_k = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.channel_v = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)

        self.fc = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                            padding=0, bias=False)

        if only_channel_tracing == False:
            self.spatial_q = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                       padding=0, bias=False)
            self.spatial_k = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                       padding=0, bias=False)
            self.spatial_v = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                       padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def masking(self, x, mask):
        mask = mask.squeeze(3).squeeze(2)
        threshold = torch.quantile(mask, self.confidence_ratio, dim=-1, keepdim=True)
        mask[mask <= threshold] = 0.0
        mask = mask.unsqueeze(2).unsqueeze(3)
        mask = mask.expand(-1, x.shape[1], x.shape[2], x.shape[3]).contiguous()
        masked_x = x * mask

        return masked_x

    def Channel_Tracer(self, x):
        avg_pool = self.GAP(x)
        x_norm = self.norm(avg_pool)

        q = self.channel_q(x_norm).squeeze(-1)
        k = self.channel_k(x_norm).squeeze(-1)
        v = self.channel_v(x_norm).squeeze(-1)

        # softmax(Q*K^T)
        QK_T = torch.matmul(q, k.transpose(1, 2))
        alpha = F.softmax(QK_T, dim=-1)

        # a*v
        att = torch.matmul(alpha, v).unsqueeze(-1)
        att = self.fc(att)
        att = self.sigmoid(att)

        output = (x * att) + x
        alpha_mask = att.clone()

        return output, alpha_mask

    def forward(self, x):
        X_c, alpha_mask = self.Channel_Tracer(x)
        X_c = self.bn(X_c)
        x_drop = self.masking(X_c, alpha_mask)

        q = self.spatial_q(x_drop).squeeze(1)
        k = self.spatial_k(x_drop).squeeze(1)
        v = self.spatial_v(x_drop).squeeze(1)

        # softmax(Q*K^T)
        QK_T = torch.matmul(q, k.transpose(1, 2))
        alpha = F.softmax(QK_T, dim=-1)

        output = torch.matmul(alpha, v).unsqueeze(1) + v.unsqueeze(1)

        return output

class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.selu(x)

        return x


class DWConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, dilation, padding):
        super(DWConv, self).__init__()
        self.out_channel = out_channel
        self.DWConv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, padding=padding, groups=in_channel,
                                dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.DWConv(x)
        out = self.selu(self.bn(x))

        return out


class DWSConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, padding, kernels_per_layer):
        super(DWSConv, self).__init__()
        self.out_channel = out_channel
        self.DWConv = nn.Conv2d(in_channel, in_channel * kernels_per_layer, kernel_size=kernel, padding=padding,
                                groups=in_channel, bias=False)
        self.bn = nn.BatchNorm2d(in_channel * kernels_per_layer)
        self.selu = nn.SELU()
        self.PWConv = nn.Conv2d(in_channel * kernels_per_layer, out_channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.DWConv(x)
        x = self.selu(self.bn(x))
        out = self.PWConv(x)
        out = self.selu(self.bn2(out))

        return out