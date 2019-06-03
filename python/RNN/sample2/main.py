# https://github.com/L1aoXingyu/Char-RNN-PyTorch/blob/master/main.py

from data import TextDataset, TextConverter

import numpy as np
import torch
from mxtorch import meter
from mxtorch.trainer import Trainer, ScheduledOptim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from config import opt
from data import TextDataset, TextConverter


def get_data(convert):
    dataset = TextDataset(opt.txt, opt.len, convert.text_to_arr)
    return DataLoader(dataset, opt.batch_size, shuffle=True, num_workers=opt.num_workers)


def get_model(convert):
    model = getattr(models, opt.model)(
        convert.vocab_size,
        opt.embed_dim,
        opt.hidden_size,
        opt.num_layers,
        opt.dropout
    )
    if opt.use_gpu:
        model = model.cuda()
    return model


def get_loss(score, label):
    return nn.CrossEntropyLoss()(score, label.view(-1))


def get_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    return ScheduledOptim(optimizer)


class CharRNNTrainer(Trainer):
    def __init__(self, convert):
        self.convert = convert
        
        model = get_model(convert)
        criterion = get_loss
        optimizer = get_optimizer(model)
        super().__init__(model, criterion, optimizer)


def train(**kwargs):
    opt._parse(kwargs)
    # torch.cuda.set_device(opt.ctx)
    convert = TextConverter(opt.txt, max_vocab=opt.max_vocab)
    train_data = get_data(convert)