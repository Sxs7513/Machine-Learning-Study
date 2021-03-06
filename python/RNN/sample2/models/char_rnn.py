import torch
from torch import nn
from torch.autograd import Variable

from config import opt

class CharRNN(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # https://www.cnblogs.com/duye/p/10590146.html
        # 词嵌入层
        self.word_to_vec = nn.Embedding(num_classes, embed_dim)
        # GRU 层
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers, dropout)
        # 线性层，最后输出预测的字符
        self.project = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, hs=None):
        batch = x.shape[0]
        if hs is None:
            hs = Variable(
                torch.zeros(self.num_layers, batch, self.hidden_size)
            )
            if opt.use_gpu:
                hs = hs.cuda()
        # [batch_size, seq_len, embed_dim]
        word_embed = self.word_to_vec(x)
        # [seq_len, batch_size, embed_dim]
        word_embed = word_embed.permute(1, 0, 2)
        # out => [seq_len, batch_size, hidden_size]
        out, h0 = self.rnn(word_embed, hs)
        le, mb, hd = out.shape
        out = out.view(le * mb, hd)
        out = self.project(out)
        out = out.view(le, mb, -1)
        out = out.permute(1, 0, 2).contiguous()  # (batch, len, hidden)
        return out.view(-1, out.shape[2]), h0
