"""Split-Attention"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, ReLU
from torch.nn.modules.utils import _pair


class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 groups=1, bias=True,
                 radix=2, reduction_factor=4, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        # 最小为32, 32是ResNext的分支数
        inter_channels = max(channels * radix // reduction_factor, 32)  # 中间层输出通道数
        self.dropblock_prob = dropblock_prob
        self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding,
                           groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)

        # 参数inplace = True:
        # inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
        # inplace：can optionally do the operation in -place.Default: False
        # 注： 产生的计算结果不会有影响。利用in - place计算可以节省内（显）存，同时还可以省去反复申请和释放内存的时间。但是会对原变量覆盖，只要不带来错误就用。
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if self.dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)  # (batch, channels*radix, x, x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            # splited是一个tuple，每个元素shape为(batch, int(rchannel//self.radix)，h, w)
            # 因为rchannel = channels*radix， 所以int(rchannel//self.radix) ==> channels
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)  # (batch, channels ，h, w)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)  # (batch, channels，1, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)  # (batch, inter_channels，1, 1)
        atten = self.fc2(gap)  # (batch, channels*radix，1, 1)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)  # (batch, channels*radix，1, 1)

        if self.radix > 1:
            # 将atten划分为self.radix
            # attens是一个tuple，每个元素shape为(batch, int(rchannel//self.radix)，h, w)
            # 因为rchannel = channels*radix， 所以int(rchannel//self.radix) ==> channels
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x  # (batch, channels*radix，h, w), radix = 1 ==> # (batch, channels，h, w)

        # 调用contiguous()之后，PyTorch会开辟一块新的内存空间存放变换之后的数据，并会真正改变Tensor的内容，按照变换之后的顺序存放数据
        return out.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        assert radix > 0, cardinality > 0
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        # 输入shape为: (batch, channels*radix，1, 1)，
        # channels表示输入到Split Attention模块时，每一个radix在【K个Cardinality】上的总通道数
        batch = x.size(0)
        if self.radix > 1:
            # x ==> (batch, self.cardinality, self.radix, -1) ==> (batch, self.radix, self.cardinality, -1)
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            # 在self.radix维度上进行softmax
            x = F.softmax(x, dim=1)  # (batch, self.radix, self.cardinality, -1)
            x = x.reshape(batch, -1)  # (batch, self.radix, self.cardinality, -1)
        else:
            x = torch.sigmoid(x)  # (batch, channels*radix，1, 1)
        return x


class Splat(nn.Module):
    """Split-Attention Module
        """
    def __init__(self, channels, radix, cardinality, reduction_factor=4):
        super(Splat, self).__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.channels = channels
        # 最小不能小于32, 32是ResNext的分支数
        # inter_channels = max(in_channels * radix // reduction_factor, 32)
        inter_channels = max(channels * radix // reduction_factor, 32)
        # self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        #  is false
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, cardinality)

    def forward(self, x):
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            # 将x划分为self.radix份
            # splited是一个tuple，每个元素shape为(batch, int(rchannel//self.radix)，h, w)
            # 因为rchannel = channels*radix， 所以int(rchannel//self.radix) ==> channels
            splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited)  # (batch, int(rchannel//self.radix)，h, w)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)  # (batch, int(rchannel//self.radix)，1, 1) ==> (batch, channels，1, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)  # (batch, inter_channels，1, 1)

        atten = self.fc2(gap)  # (batch, channels*radix，1, 1)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)  # (batch, channels*radix，1, 1)

        if self.radix > 1:
            # 将atten划分为self.radix
            # attens是一个tuple，每个元素shape为(batch, int(rchannel//self.radix)，h, w)
            # 因为rchannel = channels*radix， 所以int(rchannel//self.radix) ==> channels
            attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x  # (batch, channels*radix，h, w), radix = 1 ==> # (batch, channels，h, w)

        # 调用contiguous()之后，PyTorch会开辟一块新的内存空间存放变换之后的数据，并会真正改变Tensor的内容，按照变换之后的顺序存放数据
        return out.contiguous()


if __name__ == '__main__':
    X = torch.zeros(2, 128, 8, 8)
    # 需要满足in_channels = channels * radix, channels代表每一个radix的通道数。
    model = Splat(64, 2, 2)
    y = model(X)
    print(y.shape)

