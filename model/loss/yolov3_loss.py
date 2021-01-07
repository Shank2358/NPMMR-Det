import sys
sys.path.append("../utils")
import torch
import torch.nn as nn
from utils import utils_basic as tools

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)

        return loss

class YoloV3Loss(nn.Module):
    def __init__(self, anchors, strides, iou_threshold_loss=0.5):
        super(YoloV3Loss, self).__init__()
        self.__iou_threshold_loss = iou_threshold_loss
        self.__strides = strides

    def forward(self, p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes):
        strides = self.__strides
        loss_s, loss_s_giou, loss_s_conf, loss_s_cls = self.__cal_loss_per_layer(p[0], p_d[0], label_sbbox,
                                                               sbboxes, strides[0])
        loss_m, loss_m_giou, loss_m_conf, loss_m_cls = self.__cal_loss_per_layer(p[1], p_d[1], label_mbbox,
                                                               mbboxes, strides[1])
        loss_l, loss_l_giou, loss_l_conf, loss_l_cls = self.__cal_loss_per_layer(p[2], p_d[2], label_lbbox,
                                                               lbboxes, strides[2])
        loss = loss_l + loss_m + loss_s
        loss_giou = loss_s_giou + loss_m_giou + loss_l_giou
        loss_conf = loss_s_conf + loss_m_conf + loss_l_conf
        loss_cls = loss_s_cls + loss_m_cls + loss_l_cls

        return loss, loss_giou, loss_conf, loss_cls


    def __cal_loss_per_layer(self, p, p_d, label, bboxes, stride):

        BCE = nn.BCEWithLogitsLoss(reduction="none")
        FOCAL = FocalLoss(gamma=2, alpha=1.0, reduction="none")

        batch_size, grid = p.shape[:2]
        img_size = stride * grid
        p_conf = p[..., 4:5]
        p_cls = p[..., 5:]
        p_d_xywh = p_d[..., :4]
        label_xywh = label[..., :4]
        label_obj_mask = label[..., 4:5]
        label_cls = label[..., 6:]
        label_mix = label[..., 5:6]
        # loss giou
        giou = tools.GIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)
        loss_giou = label_obj_mask * (1.0 - giou) * label_mix
        # loss confidence
        iou = tools.iou_xywh_torch(p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        iou_max = iou.max(-1, keepdim=True)[0]
        label_noobj_mask = (1.0 - label_obj_mask) * (iou_max < self.__iou_threshold_loss).float()
        loss_conf = (label_obj_mask * FOCAL(input=p_conf, target=label_obj_mask) +
                    label_noobj_mask * FOCAL(input=p_conf, target=label_obj_mask)) * label_mix
        # loss classes
        loss_cls = label_obj_mask * BCE(input=p_cls, target=label_cls) * label_mix
        loss_giou = (torch.sum(loss_giou)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size
        loss = loss_giou + loss_conf + loss_cls
        return loss, loss_giou, loss_conf, loss_cls

