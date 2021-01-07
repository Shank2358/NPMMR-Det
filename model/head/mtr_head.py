import torch.nn as nn
import torch
from dcn_v2 import DCNv2

class MTR_Head1(nn.Module):
    def __init__(self, filters_in, anchor_num, fo_class, temp=False):
        super(MTR_Head1, self).__init__()
        self.fo_class = fo_class
        self.anchor_num = anchor_num
        self.temp = temp

        self.__conv_conf = nn.Conv2d(in_channels=filters_in, out_channels=self.anchor_num * 1, kernel_size=1, stride=1,
                                     padding=0)

        # self.__conv_offset_mask1 = Convolutional(filters_in, self.anchor_num*4, kernel_size=1, stride=1, pad=0)
        self.__conv_offset_mask = nn.Conv2d(in_channels=filters_in, out_channels=3 * 9, kernel_size=1, stride=1,
                                            padding=0, bias=True)

        self.__dconv_loc = DCNv2(filters_in, filters_in, kernel_size=3, stride=1, padding=1)
        self.__bnloc = nn.BatchNorm2d(filters_in)
        self.__reluloc = nn.LeakyReLU(inplace=True)
        self.__dconv_locx = nn.Conv2d(filters_in, self.anchor_num * 4, kernel_size=1, stride=1, padding=0)

        self.__dconv_cla = DCNv2(filters_in, filters_in, kernel_size=3, stride=1, padding=1)
        self.__bncla = nn.BatchNorm2d(filters_in)
        self.__relucla = nn.LeakyReLU(inplace=True)
        self.__dconv_clax = nn.Conv2d(filters_in, self.anchor_num * self.fo_class, kernel_size=1, stride=1, padding=0)

        self.init_offset()

    def init_offset(self):
        self.__conv_offset_mask.weight.data.zero_()
        self.__conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        out_conf = self.__conv_conf(x)

        out_offset_mask = self.__conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out_offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        # print(offset.shape)
        # if self.temp == True:
        # mask = torch.sigmoid(mask*edge)
        # else:
        out_loc = self.__dconv_locx(self.__reluloc(self.__bnloc(self.__dconv_loc(x, offset, mask))))
        out_cla = self.__dconv_clax(self.__relucla(self.__bncla(self.__dconv_cla(x, offset, mask))))

        out_loc1 = out_loc.view(x.shape[0], self.anchor_num, 4, x.shape[2], x.shape[3]).cuda()
        out_conf1 = out_conf.view(x.shape[0], self.anchor_num, 1, x.shape[2], x.shape[3]).cuda()
        out_cla1 = out_cla.view(x.shape[0], self.anchor_num, self.fo_class, x.shape[2], x.shape[3]).cuda()
        out = torch.cat((out_loc1, out_conf1, out_cla1), 2).cuda()
        return out

class MTR_Head2(nn.Module):
    def __init__(self, nC, anchors, stride):
        super(MTR_Head2, self).__init__()
        self.__anchors = anchors
        self.__nA = len(anchors)
        self.__nC = nC
        self.__stride = stride

    def forward(self, p):
        p = p.permute(0, 3, 4, 1, 2)
        p_de = self.__decode(p.clone())
        return (p, p_de)

    def __decode(self, p):
        batch_size, output_size = p.shape[:2]
        device = p.device
        stride = self.__stride
        anchors = (1.0 * self.__anchors).to(device)
        conv_raw_dxdy = p[:, :, :, :, 0:2]
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        conv_raw_conf = p[:, :, :, :, 4:5]
        conv_raw_prob = p[:, :, :, :, 5:]
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)
        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

        return pred_bbox.view(-1, 5 + self.__nC) if not self.training else pred_bbox
