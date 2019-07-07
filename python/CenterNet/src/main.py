# https://arxiv.org/abs/1904.07850
# http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Deep_Layer_Aggregation_CVPR_2018_paper.pdf
# https://www.zhihu.com/question/57493889

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory



def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    # print(opt)

    # TODO logger

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    # TODO load model

    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')
    # TODO val

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size, 
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)



if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
