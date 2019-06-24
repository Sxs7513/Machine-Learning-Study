from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

def create_model():
    num_layers = int(arch[(arch.find('_') + 1):]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    return


def load_model():
    return


def save_model():
    return