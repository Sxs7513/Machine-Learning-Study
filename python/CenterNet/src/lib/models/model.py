from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn

_model_factory = {
    "dla": get_dla_dcn,
}


def create_model(arch, heads, head_conv):
    num_layers = int(arch[(arch.find('_') + 1):]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    return model


def load_model():
    return


def save_model():
    return