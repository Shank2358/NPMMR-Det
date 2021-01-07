"""ResNet variants"""
import math
import torch
import torch.nn as nn
from model.backbones.splat import SplAtConv2d, DropBlock2D

# 全局平均池化层
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


# ResNest块
class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # 类属性
    expansion = 4  # 决定了self.conv3的输出通道数out_channels = planes*expansion

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, is_first=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality  # 中间层输出通道数
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)  # 每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半
        self.avd_first = avd_first  # 决定了池化层添加在self.conv1层后，还是添加在self.conv2层后

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=1,
                groups=cardinality, bias=False,
                radix=radix, norm_layer=norm_layer,
                dropblock_prob=dropblock_prob)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=1,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):  # x shape: (batch, inplanes, h, w)
        residual = x

        out = self.conv1(x)  # out shape: (batch, group_width, h, w)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)  # 池化层。长宽减半  ==> (batch, group_width, h/2, w/2)

        # 进行池化后：h_, w_ = h/2, w/2，否则h_, w_ = h, w
        out = self.conv2(out)  # out shape: (batch, group_width, h_, w_)
        if self.radix == 0:
            # 单层conv后接 BN, Dropout, relu
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)  # out shape: (batch, planes * self.expansion, h_, w_)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        # 对输入x进行形状改变，使之与out shape相同
        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        # 在relu之间进行叠加
        return self.relu(out)  # out shape: (batch, planes * self.expansion, h_, w_)

# X = torch.zeros(2, 128, 8, 8)
# model = Bottleneck(128, 128, norm_layer=nn.BatchNorm2d)
# for name, blk in model.named_children():
#     X = blk(X)
#     print(name, 'output shape:', X.shape)
# 输出：
# conv1 output shape: torch.Size([2, 128, 8, 8])
# bn1 output shape: torch.Size([2, 128, 8, 8])
# conv2 output shape: torch.Size([2, 128, 8, 8])
# conv3 output shape: torch.Size([2, 512, 8, 8])
# bn3 output shape: torch.Size([2, 512, 8, 8])
# relu output shape: torch.Size([2, 512, 8, 8])

######## todo:bottleneck_width ? 所有resnest参数值都为bottleneck_width=64 ############

# ResNest模型
class ResNet(nn.Module):
    """ResNet Variants

        Parameters
        ----------
        block : Block
            Class for the residual block. Options are BasicBlockV1, BottleneckV1.
        layers : list of int
            Numbers of layers in each block
        classes : int, default 1000
            Number of classification classes.
        dilated : bool, default False
            Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
            typically used in Semantic Segmentation.
        norm_layer : object
            Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
            for Synchronized Cross-GPU BachNormalization).

        Reference:

            - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

            - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
        """

    def __init__(self, block, layers, in_channels=3, radix=1, groups=1, bottleneck_width=64,
                 num_classes=1000, deep_stem=False, stem_width=64, avg_down=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, dropblock_prob=0,
                 last_gamma=False, norm_layer=nn.BatchNorm2d):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        super(ResNet, self).__init__()

        # ResNet第一个模块
        if deep_stem:
            # 3个3x3卷积层替代1个7x7卷积层
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, stem_width, kernel_size=3, stride=2, padding=1, bias=False),  # 长宽减半
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, self.inplanes, kernel_size=3, padding=1, bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=1, bias=False)  # 长宽减半

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 长宽减半

        # ResNet的后四个模块，每个模块都是ResNet残差块，这里换成了ResNest块-Bottleneck
        # 经过每个残差块后，长宽减半（除第一个残差块外），通道加倍
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, dropblock_prob=dropblock_prob)

        # 最后，与GoogLeNet一样，加入全局平均池化层后接上全连接层输出。
        self.avgpool = GlobalAvgPool2d()  # 全局平均池化层
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)

    # ResNet残差网络
    def _make_layer(self, block, planes, num_blocks, stride=1, norm_layer=None, dropblock_prob=0.0, is_first=True):
        downsample = None
        # Bottleneck网络输出通道数： planes * expansion
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 此时需要下采样，使得通道数一致
            down_layer = []
            if self.avg_down:
                # ceil_mode=True: 计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作
                # count_include_pad=False：计算平均池化时，将不包括padding的0
                down_layer.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
                down_layer.append(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, bias=False))
            else:
                down_layer.append(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))
            down_layer.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layer)

        layers = []
        # 每个模块在第一个残差块里将通道数变为planes * block.expansion，并将高和宽减半  ==> 由downsample层完成
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            radix=self.radix, cardinality=self.cardinality,
                            bottleneck_width=self.bottleneck_width,
                            avd=self.avd, avd_first=self.avd_first,
                            is_first=is_first, norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                            last_gamma=self.last_gamma))

        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            # 该block输出通道数out_channels = planes * block.expansion
            layers.append(block(self.inplanes, planes, stride=1, downsample=None,
                          radix=self.radix, cardinality=self.cardinality,
                          bottleneck_width=self.bottleneck_width,
                          avd=self.avd, avd_first=self.avd_first,
                          is_first=False, norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                          last_gamma=self.last_gamma))

        return nn.Sequential(*layers)

    def forward(self, x):  # x shape: (b, c, h, w)
        x = self.conv1(x)  # x shape: (b, self.inplanes, h//2, w//2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # x shape: (b, self.inplanes, h//4, w//4)

        # 经过每个残差块后，长宽减半（除第一个残差块外），通道加倍（除第一个残差块外）
        x = self.layer1(x)  # x shape: (b, 256, h//4, w//4)         , 256 = 64 * block.expansion
        x = self.layer2(x)  # x shape: (b, 512, h//8, w//8)         , 512 = 128 * block.expansion
        x = self.layer3(x)  # x shape: (b, 1024, h//16, w//16)      , 1024 = 256 * block.expansion
        x = self.layer4(x)  # x shape: (b, 2048, h//32, w//32)      , 2048 = 512 * block.expansion

        x = self.avgpool(x)  # x shape: (b, 2048)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)  # x shape: (b, num_classes)
        return x


if __name__ == '__main__':
    X = torch.zeros(2, 3, 512, 512)
    # resnest50
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                       radix=2, groups=1, bottleneck_width=64, num_classes=10,
                       deep_stem=True, stem_width=32, avg_down=True,
                       avd=True, avd_first=False)
    print('input shape:', X.shape)
    for name, blk in model.named_children():
        X = blk(X)
        print(name, 'output shape:', X.shape)

# stem_width=32时，输出：
# input shape: torch.Size([2, 3, 64, 64])
# conv1 output shape: torch.Size([2, 64, 32, 32])
# bn1 output shape: torch.Size([2, 64, 32, 32])
# relu output shape: torch.Size([2, 64, 32, 32])
# maxpool output shape: torch.Size([2, 64, 16, 16])
# layer1 output shape: torch.Size([2, 256, 16, 16])
# layer2 output shape: torch.Size([2, 512, 8, 8])
# layer3 output shape: torch.Size([2, 1024, 4, 4])
# layer4 output shape: torch.Size([2, 2048, 2, 2])
# avgpool output shape: torch.Size([2, 2048])
# fc output shape: torch.Size([2, 10])

# stem_width=64时，输出：
# input shape: torch.Size([2, 3, 64, 64])
# conv1 output shape: torch.Size([2, 128, 32, 32])
# bn1 output shape: torch.Size([2, 128, 32, 32])
# relu output shape: torch.Size([2, 128, 32, 32])
# maxpool output shape: torch.Size([2, 128, 16, 16])
# layer1 output shape: torch.Size([2, 256, 16, 16])   ==>   从这层开始输出通道与上边例子统一了
# layer2 output shape: torch.Size([2, 512, 8, 8])
# layer3 output shape: torch.Size([2, 1024, 4, 4])
# layer4 output shape: torch.Size([2, 2048, 2, 2])
# avgpool output shape: torch.Size([2, 2048])
# fc output shape: torch.Size([2, 10])


# stem_width=32时，输入tensor中h, w < 32时，输出：
# input shape: torch.Size([2, 3, 16, 16])
# conv1 output shape: torch.Size([2, 64, 8, 8])
# bn1 output shape: torch.Size([2, 64, 8, 8])
# relu output shape: torch.Size([2, 64, 8, 8])
# maxpool output shape: torch.Size([2, 64, 4, 4])
# layer1 output shape: torch.Size([2, 256, 4, 4])
# layer2 output shape: torch.Size([2, 512, 2, 2])
# layer3 output shape: torch.Size([2, 1024, 1, 1])
# layer4 output shape: torch.Size([2, 2048, 1, 1])  # 这里长宽没有减半了
# avgpool output shape: torch.Size([2, 2048])
# fc output shape: torch.Size([2, 10])
