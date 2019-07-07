from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat
import torch.nn.functional as F


# Focal Loss
# https://blog.csdn.net/gentleman_qin/article/details/87343004
def _neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = pred.lt(1).float()

    # 这个是论文中作者加的，作用是配合 heatmap 来让物体中心一定范围内的点的惩罚
    # 小一些，因为可以看出除了中心点之外，其他的都被归类为负样本，而对于 heatmap 内的
    # 点，它们也可以算作是“小中心”
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    
    gamma = 2

    pos_loss = torch.pow(1 - pred, gamma) * torch.log(pred) * pos_inds
    neg_loss = neg_weights * torch.pow(pred, gamma) * torch.log(1 - pred) * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)


# regr => [N, max_objs, 2]
# gt_regr => [N, max_objs, 2]
# mask => [N, max_objs]
def _reg_loss(regr, gt_regr, mask):
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()
  
  regr = regr * mask
  gt_regr = gt_regr * mask
  # https://www.zhihu.com/question/58200555
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss


class RegLoss(nn.Module):
  def __init__(self):
    super(RegLoss, self).__init__()

  # output => [N, 2, 128, 128]
  # mask => [N, max_objs]
  # ind => [N, max_objs] 中心点在一维特征图第几个像素位置
  def forward(self, output, mask, ind, target):
    # [N, max_objs, 2]
    pred = _tranpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss


class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

  
class NormRegL1Loss(nn.Module):
  pass


class RegWeightedL1Loss(nn.Module):
  pass