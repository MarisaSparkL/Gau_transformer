#---------------------------------------------------
import pandas as pd
from collections import Counter
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from    torch import autograd, einsum
import os
from tqdm import tqdm
import copy
from einops import rearrange

import collections
import torchtext
import random
from torchtext.vocab import vocab, GloVe
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import device

import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchtext import data
import math
import time
from torch.autograd import Variable
from rotary_embedding_torch import RotaryEmbedding

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder
    
class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device = device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim = -1)
        return emb * self.scale

# scalenorm

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# T5 relative positional bias

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale

# class

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        #x:[4,1024,128]
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        #out:[4,1024,4,128]
        return out.unbind(dim = -2)

# activation functions

class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

class FLASH(nn.Module):
    def __init__(
        self,
        *,
        dim,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 2.,
        causal = False,
        dropout = 0.,
        rotary_pos_emb = None,
        norm_klass = nn.LayerNorm,
        shift_tokens = False,
        laplace_attn_fn = False,
        reduce_group_non_causal_attn = True
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor) #
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        self.attn_fn = ReLUSquared() 

        # positional embeddings

        self.rotary_pos_emb = rotary_pos_emb
        self.rel_pos_bias = T5RelativePositionBias(query_key_dim ** 0.5, causal = causal)

        # norm

        self.norm = norm_klass(dim)
        self.dropout = nn.Dropout(dropout)

        # whether to reduce groups in non causal linear attention

        self.reduce_group_non_causal_attn = reduce_group_non_causal_attn

        # projections

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )

        self.qk_offset_scale = OffsetScale(query_key_dim, heads = 4)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(
        self,
        x,
        *,
        mask = None
    ):
        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)
        """

        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size
        # prenorm

        normed_x = self.norm(x) #normed_x [4,1024,512]

        # do token shift - a great, costless trick from an independent AI researcher in Shenzhen

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim = -1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
            normed_x = torch.cat((x_shift, x_pass), dim = -1)

        # initial projections

        v, gate = self.to_hidden(normed_x).chunk(2, dim = -1) #[4,1024,1024]
        qk = self.to_qk(normed_x) #[4,1024,128]

        # offset and scale

        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk) #[4,1024,128]

        # mask out linear attention keys

        if exists(mask): #no mask in prediction
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # rotate queries and keys

        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k)) #[4,1024,128]

        # padding for groups

        padding = padding_to_multiple_of(n, g)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (quad_q, quad_k, lin_q, lin_k, v))

            mask = default(mask, torch.ones((b, n), device = device, dtype = torch.bool))
            mask = F.pad(mask, (0, padding), value = False)

        # group along sequence

        quad_q, quad_k, lin_q, lin_k, v = map(lambda t: rearrange(t, 'b (n g) d -> b n g d', g = self.group_size), (quad_q, quad_k, lin_q, lin_k, v))
        # quad_q [4,4,256,128]
        # v [4,4,256,1024]

        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j = g)

        # calculate quadratic attention output

        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g #[4,4,256,256]

        sim = sim + self.rel_pos_bias(sim)

        attn = self.attn_fn(sim)  # ReLUSquared
        attn = self.dropout(attn) #[4,4,256,256]

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        quad_out = einsum('... i j, ... j d -> ... i d', attn, v) #[4,4,256,1024]

        # calculate linear attention output

        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g #[4,4,128,1024]
            
            # exclusive cumulative sum along group dimension
            lin_kv = lin_kv.cumsum(dim = 1) #lin_kv [4,4,128,1024]
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value = 0.)
            lin_out = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q) #[4,4,256,1024]
        else:
            context_einsum_eq = 'b d e' if self.reduce_group_non_causal_attn else 'b g d e'
            lin_kv = einsum(f'b g n d, b g n e -> {context_einsum_eq}', lin_k, v) / n
            lin_out = einsum(f'b g n d, {context_einsum_eq} -> b g n e', lin_q, lin_kv)

        # fold back groups into full sequence, and excise out padding

        quad_attn_out, lin_attn_out = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out, lin_out)) #[4,1024,1024]

        # gate

        out = gate * (quad_attn_out + lin_attn_out) #[4,1024,1024]

        # projection out and residual

        return self.to_out(out) + x

# FLASH Transformer

class FLASHTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 2.,
        causal = False,
        attn_dropout = 0.,
        norm_type = 'layernorm',
        shift_tokens = True,
        laplace_attn_fn = False,
        reduce_group_non_causal_attn = True
    ):
        super().__init__()
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.abs_pos_emb = ScaledSinuEmbedding(dim)
        self.group_size = group_size

        rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J

        self.layers = nn.ModuleList([FLASH(dim = dim, group_size = group_size, query_key_dim = query_key_dim, expansion_factor = expansion_factor, causal = causal, dropout = attn_dropout, rotary_pos_emb = rotary_pos_emb, norm_klass = norm_klass, shift_tokens = shift_tokens, reduce_group_non_causal_attn = reduce_group_non_causal_attn, laplace_attn_fn = laplace_attn_fn) for _ in range(depth)])

        config = Config()

        self.to_logits = nn.Sequential(
            #nn.LayerNorm(dim),
            nn.Linear(config.pad_size * dim, config.num_classes)
        )

    def forward(
        self,
        x,
        *,
        mask = None
    ):
        #x [4,1024]
        x = self.token_emb(x)
        #x [4,1024,512]
        x = self.abs_pos_emb(x) + x

        for flash in self.layers:
            # mask is None in evaluation
            x = flash(x, mask = mask)
        x = x.view(x.size(0), -1)

        return self.to_logits(x)



class Config(object):
    """配置参数"""
    def __init__(self):
        self.model_name = 'Transformer'
        self.embedding_pretrained = None  # 预训练词向量
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5 # 随机失活
        self.num_classes = 2  # 类别数
        self.num_epochs = 200  # epoch数
        self.batch_size = 20  # mini-batch大小
        self.pad_size = 500   # 每句话处理成的长度(短填长切)
        self.n_vocab = None#这里需要读取数据的部分进行赋值
        self.learning_rate = 5e-4  # 学习率
        self.embed = 300  # 词向量维度
        self.dim_model = 300
        self.hidden = 1024
        self.last_hidden = 512
        self.num_gau = 4
        self.checkpoint_path = './model.ckpt'
        self.query_key_dim = 300

torch.manual_seed(1234)

class ImdbDataset(Dataset):
    def __init__(
        self, folder_path="./aclImdb", is_train=True, is_small=False
    ) -> None:
        super().__init__()
        self.data, self.labels = self.read_dataset(folder_path, is_train, is_small)

    # 读取数据
    def read_dataset(
        self,
        folder_path,
        is_train,
        small
    ):
        data, labels = [], []
        for label in ("pos", "neg"):
            folder_name = os.path.join(
                folder_path, "train" if is_train else "test", label
            )
            for file in tqdm(os.listdir(folder_name)):
                with open(os.path.join(folder_name, file), "rb") as f:
                    text = f.read().decode("utf-8").replace("\n", "").lower()
                    data.append(text)
                    labels.append(1 if label == "pos" else 0)
        # random.shuffle(data)
        # random.shuffle(labels)
        # 小样本训练，仅用于本机验证
        
        return data, labels
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], int(self.labels[index])

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels


def get_tokenized(data):
    """获取数据集的词元列表"""

    def tokenizer(text):
        return [tok.lower() for tok in text.split(" ")]

    return [tokenizer(review) for review in data]


def get_vocab(data):
    """获取数据集的词汇表"""
    tokenized_data = get_tokenized(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # 将min_freq设置为5，确保仅包括至少出现5次的单词
    vocab_freq = {"<UNK>": 0, "<PAD>": 1}
    # 添加满足词频条件的单词到词汇表，并分配索引
    for word, freq in counter.items():
        if freq >= 5:
            vocab_freq[word] = len(vocab_freq)

    # 构建词汇表对象并返回
    return vocab(vocab_freq)


def preprocess_imdb(train_data, vocab,config):
    """数据预处理，将数据转换成神经网络的输入形式"""
    max_l = config.pad_size  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [1] * (max_l - len(x))

    labels = train_data.get_labels()
    tokenized_data = get_tokenized(train_data.get_data())
    vocab_dict = vocab.get_stoi()
    features = torch.tensor(
        [pad([vocab_dict.get(word, 0) for word in words]) for words in tokenized_data]
    )
    labels = torch.tensor([label for label in labels])
    return features, labels

def load_data(config):
    """加载数据集"""
    train_data = ImdbDataset(folder_path="./aclImdb", is_train=True)
    test_data = ImdbDataset(folder_path="./aclImdb", is_train=False)
    # print("输出第一句话：")
    # print(train_data.__getitem__(1))
    vocab = get_vocab(train_data.get_data())
    train_set = TensorDataset(*preprocess_imdb(train_data, vocab,config))
    # print("输出第一句话字典编码表示以及序列长度：")
    # print(train_set.__getitem__(1),train_set.__getitem__(1)[0].shape)
       
    test_set = TensorDataset(*preprocess_imdb(test_data, vocab,config))
    # print(f"训练集大小{train_set.__len__()}")
    # print(f"测试集大小{test_set.__len__()}")
    # print(f"词表中单词个数:{len(vocab)}")
    train_iter = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    test_iter = DataLoader(test_set, config.batch_size)
    return train_iter, test_iter, vocab

# 预先定义配置
config = Config()
train_data,test_data,vocabs_size = load_data(config)#加载数据
config.n_vocab = len(vocabs_size) + 1#补充词表大小，词表一定要多留出来一个
model = FLASHTransformer(
    num_tokens = config.n_vocab,
    dim = 300,
    depth = 4,
    causal = True,
    attn_dropout = 0.5,
    group_size = 100,
    shift_tokens = True,
    laplace_attn_fn = False
)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()#多分类的任务
batch_size=config.batch_size

#记录模型参数量
params = list(model.parameters())
k = 0
for i in params:
    l = 1
    print("该层的结构：" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("该层参数和：" + str(l))
    k = k + l
print("总参数数量和：" + str(k))

# 记录训练过程的数据
epoch_loss_values = []
metric_values = []
best_acc = 0.0
for epoch in range(config.num_epochs):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    # training
    model.train()
    for i,train_idx in enumerate(tqdm(train_data)):
        features, labels = train_idx
        features = features.cuda()
        labels = labels.cuda()
        
        optimizer.zero_grad() 
        outputs = model(features) 
        
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step() 
        
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss.item()
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_data)):
            features, labels = batch
            features = features.cuda()
            labels = labels.cuda()
            outputs = model(features)

            loss = criterion(outputs, labels) 
            
            _, val_pred = torch.max(outputs, 1) 
            val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
            val_loss += loss.item()
    print(f'训练信息：[{epoch+1:03d}/{config.num_epochs:03d}] Train Acc: {train_acc/25000:3.5f} Loss: {train_loss/len(train_data):3.5f} | Val Acc: {val_acc/25000:3.5f} loss: {val_loss/len(test_data):3.5f}')
    epoch_loss_values.append(train_loss/len(train_data))
    metric_values.append(val_acc/25000)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), config.checkpoint_path)
        print(f'saving model with acc {best_acc/25000:.5f}')
    print(f'best model with acc {best_acc/25000:.5f}') 
        
torch.save(model.state_dict(), 'model_flash.pt')   
        