import torch
import numpy as np
import torch.nn as nn
from ..layers.convolutions import Convolutional
from ..layers.conv_blocks import Residual_block, Residual_block_CSP

class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        """
        x1 means previous output; x2 means current output
        """
        out = torch.cat((x2, x1), dim=1)
        return out

class CSPDarknet53(nn.Module):

    def __init__(self, pre_weight=None):
        super(CSPDarknet53, self).__init__()
        self.__conv = Convolutional(filters_in=3, filters_out=32, kernel_size=3, stride=1, pad=1, norm='bn',
                                    activate='MEMish')

        self.__conv_5_0 = Convolutional(filters_in=32, filters_out=64, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='MEMish')#
        self.__conv_5_0_0 = Convolutional(filters_in=64, filters_out=64, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='MEMish')#
        self.__route_5_0_0 = Route() ## self.__conv_5_0
        self.__conv_5_0_1 = Convolutional(filters_in=128, filters_out=64, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')#
        self.__rb_5_0 = Residual_block_CSP(filters_in=64)
        self.__conv_5_0_2 = Convolutional(filters_in=64, filters_out=64, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')#
        self.__route_5_0_1 = Route() ### self.__conv_5_0
        self.__conv_5_0_3 = Convolutional(filters_in=128, filters_out=64, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')


        self.__conv_5_1 = Convolutional(filters_in=64, filters_out=128, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__conv_5_1_0 = Convolutional(filters_in=128, filters_out=64, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')
        self.__route_5_1_0 = Route() ## self.__conv_5_1 128+64
        self.__conv_5_1_1 = Convolutional(filters_in=128+64, filters_out=64, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')
        self.__rb_5_1_0 = Residual_block_CSP(filters_in=64)
        self.__rb_5_1_1 = Residual_block_CSP(filters_in=64)
        self.__conv_5_1_2 = Convolutional(filters_in=64, filters_out=64, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')
        self.__route_5_1_1 = Route() ## self.__conv_5_1_0 64+64
        self.__conv_5_1_3 = Convolutional(filters_in=64+64, filters_out=128, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')


        self.__conv_5_2 = Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__conv_5_2_0 = Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')
        self.__route_5_2_0 = Route() ## self.__conv_5_2 128+256
        self.__conv_5_2_1 = Convolutional(filters_in=128 + 256, filters_out=128, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')
        self.__rb_5_2_0 = Residual_block_CSP(filters_in=128)
        self.__rb_5_2_1 = Residual_block_CSP(filters_in=128)
        self.__rb_5_2_2 = Residual_block_CSP(filters_in=128)
        self.__rb_5_2_3 = Residual_block_CSP(filters_in=128)
        self.__rb_5_2_4 = Residual_block_CSP(filters_in=128)
        self.__rb_5_2_5 = Residual_block_CSP(filters_in=128)
        self.__rb_5_2_6 = Residual_block_CSP(filters_in=128)
        self.__rb_5_2_7 = Residual_block_CSP(filters_in=128)
        self.__conv_5_2_2 = Convolutional(filters_in=128, filters_out=128, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')
        self.__route_5_2_1 = Route() ## self.__conv_5_2_0 128+128
        self.__conv_5_2_3 = Convolutional(filters_in=128+128, filters_out=256, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')
        

        self.__conv_5_3 = Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__conv_5_3_0 = Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')
        self.__route_5_3_0 = Route() ## self.__conv_5_3 256+512
        self.__conv_5_3_1 = Convolutional(filters_in=256 + 512, filters_out=256, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')
        self.__rb_5_3_0 = Residual_block_CSP(filters_in=256)
        self.__rb_5_3_1 = Residual_block_CSP(filters_in=256)
        self.__rb_5_3_2 = Residual_block_CSP(filters_in=256)
        self.__rb_5_3_3 = Residual_block_CSP(filters_in=256)
        self.__rb_5_3_4 = Residual_block_CSP(filters_in=256)
        self.__rb_5_3_5 = Residual_block_CSP(filters_in=256)
        self.__rb_5_3_6 = Residual_block_CSP(filters_in=256)
        self.__rb_5_3_7 = Residual_block_CSP(filters_in=256)
        self.__conv_5_3_2 = Convolutional(filters_in=256, filters_out=256, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')
        self.__route_5_3_1 = Route() ## self.__conv_5_3_0 256+256
        self.__conv_5_3_3 = Convolutional(filters_in=256+256, filters_out=512, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')


        self.__conv_5_4 = Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__conv_5_4_0 = Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')
        self.__route_5_4_0 = Route() ## self.__conv_5_4 512+1024
        self.__conv_5_4_1 = Convolutional(filters_in=512 + 1024, filters_out=512, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')
        self.__rb_5_4_0 = Residual_block_CSP(filters_in=512)
        self.__rb_5_4_1 = Residual_block_CSP(filters_in=512)
        self.__rb_5_4_2 = Residual_block_CSP(filters_in=512)
        self.__rb_5_4_3 = Residual_block_CSP(filters_in=512)
        self.__conv_5_4_2 = Convolutional(filters_in=512, filters_out=512, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')
        self.__route_5_4_1 = Route() ## self.__conv_5_4_0 512+512
        self.__conv_5_4_3 = Convolutional(filters_in=512+512, filters_out=1024, kernel_size=1, stride=1, pad=0, norm='bn',
                                        activate='leaky')

        self.__initialize_weights()

        if pre_weight:
            self.load_darknet_weights(pre_weight)


    def __initialize_weights(self):
        print("**" * 10, "Initing darknet weights", "**" * 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                print("initing {}".format(m))


    def load_darknet_weights(self, weight_file, cutoff=52):
        "https://github.com/ultralytics/yolov3/blob/master/models.py"

        print("**"*25 + "\nload darknet weights : ", weight_file)

        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                if count == cutoff:
                    break
                count += 1

                conv_layer = m._Convolutional__conv
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                    print("loading weight {}".format(bn_layer))
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

                print("loading weight {}".format(conv_layer))


    def forward(self, x):
        conv = self.__conv(x)

        conv_5_0 = self.__conv_5_0(conv)
        conv_5_0_0 = self.__conv_5_0_0(conv_5_0)
        route_5_0_0 = self.__route_5_0_0(conv_5_0_0, conv_5_0)
        conv_5_0_1 = self.__conv_5_0_1(route_5_0_0)
        rb_5_0 = self.__rb_5_0(conv_5_0_1)
        conv_5_0_2 = self.__conv_5_0_2(rb_5_0)
        route_5_0_1 = self.__route_5_0_1(conv_5_0_2, conv_5_0_0)
        conv_5_0_3 = self.__conv_5_0_3(route_5_0_1)

        conv_5_1 = self.__conv_5_1(conv_5_0_3)
        conv_5_1_0 = self.__conv_5_1_0(conv_5_1)
        route_5_1_0 = self.__route_5_1_0(conv_5_1_0, conv_5_1)
        conv_5_1_1 = self.__conv_5_1_1(route_5_1_0)
        rb_5_1_0 = self.__rb_5_1_0(conv_5_1_1)
        rb_5_1_1 = self.__rb_5_1_1(rb_5_1_0)
        conv_5_1_2 = self.__conv_5_1_2(rb_5_1_1)
        route_5_1_1 = self.__route_5_1_1(conv_5_1_2, conv_5_1_0)
        conv_5_1_3 = self.__conv_5_1_3(route_5_1_1)

        conv_5_2 = self.__conv_5_2(conv_5_1_3)
        conv_5_2_0 = self.__conv_5_2_0(conv_5_2)
        route_5_2_0 = self.__route_5_2_0(conv_5_2_0, conv_5_2)
        conv_5_2_1 = self.__conv_5_2_1(route_5_2_0)
        rb_5_2_0 = self.__rb_5_2_0(conv_5_2_1)
        rb_5_2_1 = self.__rb_5_2_1(rb_5_2_0)
        rb_5_2_2 = self.__rb_5_2_2(rb_5_2_1)
        rb_5_2_3 = self.__rb_5_2_3(rb_5_2_2)
        rb_5_2_4 = self.__rb_5_2_4(rb_5_2_3)
        rb_5_2_5 = self.__rb_5_2_5(rb_5_2_4)
        rb_5_2_6 = self.__rb_5_2_6(rb_5_2_5)
        rb_5_2_7 = self.__rb_5_2_7(rb_5_2_6)
        conv_5_2_2 = self.__conv_5_2_2(rb_5_2_7)
        route_5_2_1 = self.__route_5_2_1(conv_5_2_2, conv_5_2_0)
        conv_5_2_3 = self.__conv_5_2_3(route_5_2_1)

        conv_5_3 = self.__conv_5_3(conv_5_2_3)
        conv_5_3_0 = self.__conv_5_3_0(conv_5_3)
        route_5_3_0 = self.__route_5_3_0(conv_5_3_0, conv_5_3)
        conv_5_3_1 = self.__conv_5_3_1(route_5_3_0)
        rb_5_3_0 = self.__rb_5_3_0(conv_5_3_1)
        rb_5_3_1 = self.__rb_5_3_1(rb_5_3_0)
        rb_5_3_2 = self.__rb_5_3_2(rb_5_3_1)
        rb_5_3_3 = self.__rb_5_3_3(rb_5_3_2)
        rb_5_3_4 = self.__rb_5_3_4(rb_5_3_3)
        rb_5_3_5 = self.__rb_5_3_5(rb_5_3_4)
        rb_5_3_6 = self.__rb_5_3_6(rb_5_3_5)
        rb_5_3_7 = self.__rb_5_3_7(rb_5_3_6)
        conv_5_3_2 = self.__conv_5_3_2(rb_5_3_7)
        route_5_3_1 = self.__route_5_3_1(conv_5_3_2, conv_5_3_0)
        conv_5_3_3 = self.__conv_5_3_3(route_5_3_1)

        conv_5_4 = self.__conv_5_4(conv_5_3_3)
        conv_5_4_0 = self.__conv_5_4_0(conv_5_4)
        route_5_4_0 = self.__route_5_4_0(conv_5_4_0, conv_5_4)
        conv_5_4_1 = self.__conv_5_4_1(route_5_4_0)
        rb_5_4_0 = self.__rb_5_4_0(conv_5_4_1)
        rb_5_4_1 = self.__rb_5_4_1(rb_5_4_0)
        rb_5_4_2 = self.__rb_5_4_2(rb_5_4_1)
        rb_5_4_3 = self.__rb_5_4_3(rb_5_4_2)
        conv_5_4_2 = self.__conv_5_4_2(rb_5_4_3)
        route_5_4_1 = self.__route_5_4_1(conv_5_4_2, conv_5_4_0)
        conv_5_4_3 = self.__conv_5_4_3(route_5_4_1)
  
        return conv_5_2_3, conv_5_3_3, conv_5_4_3