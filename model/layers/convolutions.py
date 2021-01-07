from .activations import *
from dcn_v2 import DCN
from ..plugandplay.DynamicConv import Dynamic_conv2d
from ..plugandplay.CondConv import CondConv2d, route_func
from ..layers.directional_dynamic_convolution import *

norm_name = {"bn": nn.BatchNorm2d}
activate_name = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "relu6": nn.ReLU6,
    "Mish": Mish,
    "Swish": Swish,
    "MEMish": MemoryEfficientMish,
    "MESwish": MemoryEfficientSwish,
    "FReLu": FReLU
}

class Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride, pad, groups=1, dila=1, norm=None, activate=None):
        super(Convolutional, self).__init__()
        self.norm = norm
        self.activate = activate
        self.__conv = nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size,
                                stride=stride, padding=pad, bias=not norm, groups=groups, dilation=dila)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)
        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "relu6":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "Mish":
                self.__activate = Mish()
            if activate == "Swish":
                self.__activate = Swish()
            if activate == "MEMish":
                self.__activate = MemoryEfficientMish()
            if activate == "MESwish":
                self.__activate = MemoryEfficientSwish()
            if activate == "FReLu":
                self.__activate = FReLU()

    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x

class Separable_Conv(nn.Module):
    def __init__(self, filters_in, filters_out, stride, norm="bn", activate="relu6"):
        super(Separable_Conv, self).__init__()

        self.__dw = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=3,
                                  stride=stride, pad=1, groups=filters_in, norm=norm, activate=activate)

        self.__pw = Convolutional(filters_in=filters_in, filters_out=filters_out, kernel_size=1,
                                  stride=1, pad=0, norm=norm, activate=activate)

    def forward(self, x):
        return self.__pw(self.__dw(x))


class Deformable_Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride, pad, groups=1, norm=None, activate=None):
        super(Deformable_Convolutional, self).__init__()
        self.norm = norm
        self.activate = activate
        self.__dcn = DCN(filters_in, filters_out, kernel_size=kernel_size, stride=stride, padding=pad, deformable_groups=groups).cuda()
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)
        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "relu6":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "Mish":
                self.__activate = Mish()
            if activate == "Swish":
                self.__activate = Swish()
            if activate == "MEMish":
                self.__activate = MemoryEfficientMish()
            if activate == "MESwish":
                self.__activate = MemoryEfficientSwish()
            if activate == "FReLu":
                self.__activate = FReLU()

    def forward(self, x):
        x = self.__dcn(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x

class Separable_Conv_dila(nn.Module):
    def __init__(self, filters_in, filters_out, stride, pad, dila):
        super(Separable_Conv_dila, self).__init__()

        self.__dw = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=3, stride=stride,
                                  pad=pad, groups=filters_in, dila=dila, norm="bn", activate="relu6")
        #self.__se=SELayer(filters_in)
        self.__pw = Convolutional(filters_in=filters_in, filters_out=filters_out, kernel_size=1, stride=1,
                                  pad=0, norm="bn", activate="relu6")

    def channel_shuffle(self, features, groups=2):
        batchsize, num_channels, height, width = features.data.size()
        assert (num_channels % groups == 0)
        channels_per_group = num_channels // groups
        # reshape
        features = features.view(batchsize, groups, channels_per_group, height, width)
        features = torch.transpose(features, 1, 2).contiguous()
        # flatten
        features = features.view(batchsize, -1, height, width)
        return features

    def forward(self, x):
        #return self.__pw(self.__se(self.__dw(x)))
        out = self.__pw(self.__dw(x))
        #out = self.channel_shuffle(out)
        return out


class Cond_Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride=1, pad=0, dila=1, groups=1, bias=True, num_experts=1, norm=None, activate=None):

        super(Cond_Convolutional, self).__init__()
        self.norm = norm
        self.activate = activate
        self.__conv = CondConv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size,
                                 stride=stride, padding=pad, dilation=dila, groups=groups, bias=bias, num_experts=num_experts)
        self.__routef = route_func(filters_in, num_experts)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)
        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "relu6":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "Mish":
                self.__activate = Mish()
            if activate == "Swish":
                self.__activate = Swish()
            if activate == "MEMish":
                self.__activate = MemoryEfficientMish()
            if activate == "MESwish":
                self.__activate = MemoryEfficientSwish()
            if activate == "FReLu":
                self.__activate = FReLU()

    def forward(self, x):
        routef = self.__routef(x)
        x = self.__conv(x,routef)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x


class Dynamic_Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride=1, pad=0, dila=1, groups=1, bias=True, K=4, temperature=34, norm=None, activate=None):

        super(Dynamic_Convolutional, self).__init__()
        self.norm = norm
        self.activate = activate
        self.__conv = Dynamic_conv2d(in_planes=filters_in, out_planes=filters_out, kernel_size=kernel_size,
                                     ratio=0.25, stride=stride, padding=pad, dilation=dila, groups=groups, bias=bias, K=K, temperature=temperature, init_weight=True)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)
        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "relu6":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "Mish":
                self.__activate = Mish()
            if activate == "Swish":
                self.__activate = Swish()
            if activate == "MEMish":
                self.__activate = MemoryEfficientMish()
            if activate == "MESwish":
                self.__activate = MemoryEfficientSwish()
            if activate == "FReLu":
                self.__activate = FReLU()

    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x

class Directional_Dynamic_Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride=1, pad=0, dila=1, groups=1, bias=True, type='tri', temperature=34, norm=None, activate=None):

        super(Directional_Dynamic_Convolutional, self).__init__()
        self.norm = norm
        self.activate = activate

        if type =='tri':
            self.__conv = Directional_Dynamic_Conv2d_Triangle(in_planes=filters_in, out_planes=filters_out, kernel_size=kernel_size, ratio=0.25, stride=stride,
                                         padding=pad, dilation=dila, groups=groups, bias=bias, temperature=temperature, init_weight=True)
        if type =='tri_sw':
            self.__conv = Directional_Dynamic_Conv2d_Triangle_SW(in_planes=filters_in, out_planes=filters_out, kernel_size=kernel_size, ratio=0.25, stride=stride,
                                         padding=pad, dilation=dila, groups=groups, bias=bias, temperature=temperature, init_weight=True)
        if type =='rect_sw':
            self.__conv = Directional_Dynamic_Conv2d_SW(in_planes=filters_in, out_planes=filters_out, kernel_size=kernel_size, ratio=0.25, stride=stride,
                                         padding=pad, dilation=dila, groups=groups, bias=bias, temperature=temperature, init_weight=True)
        if type == 'rect':
            self.__conv = Dynamic_conv2d(in_planes=filters_in, out_planes=filters_out, kernel_size=kernel_size,
                                     ratio=0.25, stride=stride, padding=pad, dilation=dila, groups=groups, bias=bias, K=K, temperature=temperature, init_weight=True)

        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)
        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "relu6":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "Mish":
                self.__activate = Mish()
            if activate == "Swish":
                self.__activate = Swish()
            if activate == "MEMish":
                self.__activate = MemoryEfficientMish()
            if activate == "MESwish":
                self.__activate = MemoryEfficientSwish()
            if activate == "FReLu":
                self.__activate = FReLU()

    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x