from collections.abc import dict_values

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributed.autograd import context
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k


d_model = 0
d_k = 0
d_v = 0
n_heads = 0

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
        :param attn_mask:？
        :return:
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




