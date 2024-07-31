import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
import zarr
import copy
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(vocab_size, d_model)
        self.d_model = d_model

    # input x: (batch_size, seq_len)
    # output: (batch_size, seq_len, d_model)
    def forward(self, x):
        out =  self.lut(x) 
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            -(math.log(10000.0) / d_model))  # (d_model/2)

        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, d_model)
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, d_model)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    # input x: (batch_size, seq_len, d_model)
    # output: (batch_size, seq_len, d_model)
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        # input: (batch_size, seq_len, d_model)
        self.linears = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # (batch_size, seq_len, d_model) => (batch_size, h, seq_len, d_k)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = self.attention(query,
                                 key,
                                 value,
                                 mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None, dropout=None): 
        d_k = query.size(-1)
        # (batch_size, h, seq_len, d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # (batch_size, h, seq_len, seq_len)
        p_attn = nn.functional.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        # (batch_size, h, seq_len, d_k) * (batch_size, h, seq_len, seq_len)
        # => (batch_size, h, seq_len, d_k)
        return torch.matmul(p_attn, value), p_attn 

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Decoder(nn.Module):
    def __init__(self, embedding, positional_encoding, layer, N):
        super(Decoder, self).__init__()
        self.embedding = embedding
        self.positional_encoding = positional_encoding
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

if __name__ == '__main__':
    decoder = Decoder(DecoderLayer(16, MultiHeadedAttention(4, 16), PositionwiseFeedForward(16, 128), 0.1), 6)
    x = torch.randn(2, 10, 16)
    mask = None
    y = decoder(x, mask=mask)
    print(y.shape)

        