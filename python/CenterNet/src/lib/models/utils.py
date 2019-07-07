from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _sigmoid(x):
    y = torch.clamp(torch.sigmoid(x), min=1e-4, max=1-1e-4)
    return y


# feat => [N, 128 * 128, 2]
# ind => [N, max_objs]
def _gather_feat(feat, ind):
    dim = feat.size(2)
    # [N, max_objs, 1] => [N, max_objs, 2]
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # [N, max_objs, 2]
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


# feat => [N, 2, 128, 128]
# ind => [N, max_objs]
def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat