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

class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2
    
class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)
    
class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out

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

my_mask = torch.ones(500, 500).triu(1)
for i in range(500) :
    for j in range(500) :
        if abs(i - j) > 100:
            my_mask[i,j] = 1
my_mask = my_mask.unsqueeze(0).expand(20, -1, -1)
#print(my_mask)

my_mask2 = torch.ones(500, 500).tril(1)
for i in range(500) :
    for j in range(500) :
        if abs(i - j) >= 100:
            my_mask2[i,j] = 0 
my_mask2 = my_mask2.unsqueeze(0).expand(20, -1, -1)
#print(my_mask2)

class GAU(nn.Module):
    def __init__(
        self,
        *,
        dim,
        query_key_dim = 128,
        expansion_factor = 2.,
        add_residual = True,
        causal = False,
        dropout = 0.,
        norm_klass = nn.LayerNorm
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)

        self.norm = norm_klass(dim)
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        self.query_key_dim = query_key_dim

        self.attn_fn = ReLUSquared() 

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )

        self.offsetscale = OffsetScale(query_key_dim, heads = 2)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.add_residual = add_residual
        self.rotary_pos_emb = RotaryEmbedding(dim = min(32, self.query_key_dim))
        self.rel_pos_bias = T5RelativePositionBias(query_key_dim ** 0.5, causal = causal)
        self.my_mask = my_mask
        self.my_mask2 = my_mask2

    def forward(
        self,
        x,
        mask = None
    ):
        seq_len, device = x.shape[-2], x.device
        #print("===============================")
        # print("x")
        # print(x.shape) 
        # x [20,500,300]
        normed_x = self.norm(x)
        # print("normed_x")
        # print(normed_x.shape)
        # normed_x [20,500,300]

#do token shifts
        x_shift, x_pass = normed_x.chunk(2, dim = -1)
        # print("x_shift")
        # print(x_shift.shape)
        # x_shift [20,500,150]
        # print("x_pass")
        # print(x_pass.shape)
        # x_pass [20,500,150]
        x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
        normed_x = torch.cat((x_shift, x_pass), dim = -1)
        # print("normed_x")
        # print(normed_x.shape)
        # normed_x [20,500,300]
        

        v, gate = self.to_hidden(normed_x).chunk(2, dim = -1) #v, gate [500,600]

        gate_mask_list = []
        for batch in range(20):
            t_gate = gate[batch]
            t_gate = t_gate.cpu()
            t_gate = t_gate.numpy()
            gate_max = np.max(t_gate)
            trim_thresh = 1
            trim_thresh_start = int(np.floor(gate_max))
            #print("max thresh ",trim_thresh_start)
            for threshold in range(trim_thresh_start,0,-1):
                bool_matrix = np.abs(t_gate) >= threshold
                count = np.sum(bool_matrix)
                #print("count ",count)
                if count > (300000 * 0.3):
                    trim_thresh = threshold
                    break
            #print("trim_thresh ",trim_thresh)
            counts = np.zeros((100, 120))
            for i in range(100):
                for j in range(120):
                    sub_matrix = t_gate[5*i:(5*i + 4), 5*j:(5*j + 4)]
                    bool_matrix = np.abs(sub_matrix) >= trim_thresh
                    count = np.sum(bool_matrix)
                    counts[i][j] = count
            #np.savetxt('counts.txt', counts, fmt='%d')
            thresh2 = int(np.max(counts))
            for t in range(thresh2,0,-1):
                bool_matrix = np.abs(counts) >= t
                c = np.sum(bool_matrix)
                if c > (12000 * 0.3):
                    thresh2 = t
                    #print("thresh2 c ",c)
                    break
            #print("thresh2 ",thresh2)
            gate_mask = np.zeros((500, 600))
            for i in range(100):
                for j in range(120):
                    if(counts[i][j] >= thresh2):
                        gate_mask[i*5:i*5+4,5*j:(5*j + 4)] = 1
                    else:
                        gate_mask[i*5:i*5+4,5*j:(5*j + 4)] = 0.25
            #gate_mask = np.where(counts == 1, np.ones((5, 5)), np.full((5, 5), 1 >> 2))
            #np.savetxt('gate_mask.txt', gate_mask, fmt='%f')
            gate_mask_list.append(gate_mask)
        gate_mask = np.stack(gate_mask_list, axis=0)
        gate_mask = gate_mask.astype(np.float32)
        gate_mask = torch.from_numpy(gate_mask)
        gate_mask = gate_mask.cuda()

        # t_gate = gate.cpu()
        # gate1 = t_gate.numpy()
        # gate1 = gate1[0]
        # print("gate1 shape")
        # print(gate1.shape)
        # np.savetxt('gate1.txt', gate1, fmt='%f')
        #return

        qk = self.to_qk(normed_x) #qk [500,128]
        q, k = self.offsetscale(qk) #q, k [500,128]

        q, k = map(self.rotary_pos_emb.rotate_queries_or_keys, (q, k)) 

        sim = einsum('b i d, b j d -> b i j', q, k)
        sim = sim + self.rel_pos_bias(sim)

        attn = self.attn_fn(sim / seq_len)
        attn = self.dropout(attn) #attn [500,500]

        my_mask = self.my_mask.type(torch.bool).to(attn.device)
        # t_mask = self.my_mask.cpu()
        # mask1 = t_mask.numpy()
        # np.savetxt('mask1.txt', mask1[0], fmt='%f') 
        # #print('prev attn')
        # #print(attn)
        # print("attn shape")
        # print(attn.shape)
        # t_attn = attn.cpu()
        # array1 = t_attn.numpy()
        # array1 = array1[0]
        # print("array shape")
        # print(array1.shape)
        # np.savetxt('array1.txt', array1, fmt='%f')
        attn = attn * self.my_mask2.to(attn.device)
        #attn = attn.masked_fill(my_mask, 0.)
        # #print('post attn')
        # #print(attn)
        # t_attn = attn.cpu()
        # array2 = t_attn.numpy()
        # array2 = array2[0]
        # np.savetxt('array2.txt', array2, fmt='%f')
        # return

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 j')
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((seq_len, seq_len), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        out = einsum('b i j, b j d -> b i d', attn, v)

        out = gate_mask * out

        out = out * gate #out [500,600]

        out = self.to_out(out) #out [500,300]

        if self.add_residual:
            out = out + x

        return out



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
        self.checkpoint_path = '../model.ckpt'
        self.query_key_dim = 300

torch.manual_seed(1234)

class ImdbDataset(Dataset):
    def __init__(
        self, folder_path="../aclImdb", is_train=True, is_small=False
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
    train_data = ImdbDataset(folder_path="../aclImdb", is_train=True)
    test_data = ImdbDataset(folder_path="../aclImdb", is_train=False)

    vocab = get_vocab(train_data.get_data())
    train_set = TensorDataset(*preprocess_imdb(train_data, vocab,config))
       
    test_set = TensorDataset(*preprocess_imdb(test_data, vocab,config))

    train_iter = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    test_iter = DataLoader(test_set, config.batch_size)
    return train_iter, test_iter, vocab

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout, config.device)
        self.gau = GAU(dim = 300, query_key_dim = 128, expansion_factor = 2., add_residual= True , causal = True, dropout = 0.5, norm_klass = nn.LayerNorm)
        self.gaus = nn.ModuleList([
            copy.deepcopy(self.gau)
            for _ in range(config.num_gau)])

        self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)

        out = self.postion_embedding(out)
        for gau in self.gaus:
            out = gau(out)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.fc1(out)
        return out



# 预先定义配置
config = Config()
train_data,test_data,vocabs_size = load_data(config)#加载数据
config.n_vocab = len(vocabs_size) + 1#补充词表大小，词表一定要多留出来一个
model = Model(config)#调用transformer的编码器


#load Model 
stat_dict = torch.load('../models_save/gau_best.pt')
model.load_state_dict({k.replace('net.',''):v for k,v in stat_dict.items()})

model.cuda()

optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()#多分类的任务
batch_size=config.batch_size

#记录模型参数量
# params = list(model.parameters())
# k = 0
# for i in params:
#     l = 1
#     print("该层的结构：" + str(list(i.size())))
#     for j in i.size():
#         l *= j
#     print("该层参数和：" + str(l))
#     k = k + l
# print("总参数数量和：" + str(k))

val_acc = 0.0
val_loss = 0.0

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
        # if i >= 125:
        #     break

print(f'Val Acc: {val_acc/25000:3.5f} loss: {val_loss/len(test_data):3.5f}')


        