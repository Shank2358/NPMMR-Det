import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class route_func(nn.Module):
    r"""CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf
    Args:
        c_in (int): Number of channels in the input image
        num_experts (int): Number of experts for mixture. Default: 1
    """

    def __init__(self, c_in, num_experts):
        super(route_func, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(c_in, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class CondConv2d(nn.Module):
    r"""CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts for mixture. Default: 1
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(CondConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, c_in, h, w = x.size()
        k, c_out, c_in, kh, kw = self.weight.size()
        x = x.view(1, -1, h, w)
        weight = self.weight.view(k, -1)

        combined_weight = torch.mm(routing_weight, weight).view(-1, c_in, kh, kw)

        if self.bias is not None:
            combined_bias = torch.mm(routing_weight, self.bias).view(-1)
            output = F.conv2d(
                x, weight=combined_weight, bias=combined_bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv2d(
                x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)

        output = output.view(b, c_out, output.size(-2), output.size(-1))
        return output