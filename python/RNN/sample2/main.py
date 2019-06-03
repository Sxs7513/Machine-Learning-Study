# https://github.com/L1aoXingyu/Char-RNN-PyTorch/blob/master/main.py

from data import TextDataset, TextConverter

from torch.utils.data import DataLoader

from config import opt


def get_data(convert):
    dataset = TextDataset(opt.txt, opt.len, convert.text_to_arr)
    return DataLoader(dataset, opt.batch_size, shuffle=True, num_workers=opt.num_workers)


class CharRNNTrainer(Trainer):
    def __init__(self, convert):
        self.convert = convert
        


def train(**kwargs):
    opt._parse(kwargs)
    # torch.cuda.set_device(opt.ctx)
    convert = TextConverter(opt.txt, max_vocab=opt.max_vocab)
    train_data = get_data(convert)