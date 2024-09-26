

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributed.autograd import context
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k

src_len = 5  # length of source
tgt_len = 5  # length of target

## 模型参数
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

src_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
tgt_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

# 创建词汇表
def yield_tokens(data_iter, tokenizer):
    for src_sentence, tgt_sentence in data_iter:
        yield tokenizer(src_sentence)
        yield tokenizer(tgt_sentence)

# 下载和加载 Multi30k 数据集
train_iter = Multi30k(split='train', language_pair=('en', 'fr'))

# 创建词汇表
src_vocab = build_vocab_from_iterator(yield_tokens(train_iter, src_tokenizer), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
src_vocab.set_default_index(src_vocab["<unk>"])

# 创建目标词汇表
tgt_vocab = build_vocab_from_iterator(yield_tokens(train_iter, tgt_tokenizer), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
tgt_vocab.set_default_index(tgt_vocab["<unk>"])



def mak_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    return  torch.LongTensor(input_batch),torch


class ScaledDotProductAttention(nn.Module):
    def __init(self):
        super(ScaledDotProductAttention,self).__init__()
    def forward(self, Q, K: torch.Tensor, V, attn_mask):
        """

        :param Q: [batch_size, n_heads,len_q,d_k] [批次大小， 多头数量， 查询队列长度, 键特征长度]
        :param K: [batch_size, n_heads, len_k, d_k] [批次大小， 多头数量， 键长度, 键特征长度]
        :param V:[批次大小， 多头数量， 键长度, 值特征长度]
        :param attn_mask: [batch_size, n_heads, len_q, len_k]
        :return: context, attn
        """
        scores =  torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(K.shape[-1])  # [batch_size, n_heads, len_q, len_k]
        scores.masked_fill(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores) # [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention,self).__init__()
        self.W_Q = nn.Linear(d_model, d_k*n_heads)
        self.W_K = nn.Linear(d_model, d_k*n_heads)
        self.W_V = nn.Linear(d_model, d_v*n_heads)
        self.linear = nn.Linear(n_heads*d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size, len_q, d_model]
        :param K: [batch_size, len_k, d_model]
        :param V: [batch_size, len_v, d_model]
        :param attn_mask: [batch_size, len_q, len_k]
        :return:
        """
        residual, batch_size = Q, Q.size(0)

        # 先映射，后分头
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2) # [batch_size, n_heads, len_q, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2) # [batch_size, n_heads, len_k, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2) # [batch_size, n_heads, len_v, d_v]

        # 复制attention mask到每个头
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # [batch_size, n_heads, len_q, len_k]

        # 通过scaled dot-product attention
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1,2).contiguous().view(batch_size, -1, n_heads*d_v) # [batch_size, len_q, n_heads*d_v]
        output = self.linear(context) # [batch_size, len_q, d_model]
        return self.layer_norm(output + residual), attn



# TODO: Position-wise Feedforward Networks
class PositionwiseFeedforward(nn.Module):
    def __init__(self):
        super(PositionwiseFeedforward,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1,2)))  # [batch_size, d_ff, len_q]
        output = self.conv2(output).transpose(1,2) # [batch_size, len_q, d_model]
        return self.layer_norm(output + residual)

def get_attn_pad_mask(seq_q, seq_k):
    """
    :param seq_q: [batch_size, len_q]
    :param seq_k: [batch_size, len_k]
    :return: pad_mask: [batch_size, len_q, len_k]
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask =  seq_k.data.eq(0).unsqueeze(1) # [batch_size, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k) # [batch_size, len_q, len_k]

class PositionalEncoding(nn.Module):
    def __init__(self,d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)





