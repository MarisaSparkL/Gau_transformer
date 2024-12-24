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

gau_scales = [[torch.ones(300),torch.ones(300),torch.ones(600)],
            [torch.ones(300),torch.ones(300),torch.ones(600)],
            [torch.ones(300),torch.ones(300),torch.ones(600)],
            [torch.ones(300),torch.ones(300),torch.ones(600)]
            ]
# 0: [1200,3,1] 1:[128,3,1] 2:[300,6,1]
gau_clips = [[torch.ones(1200,3),torch.ones(128,3),torch.ones(300,6)],
            [torch.ones(1200,3),torch.ones(128,3),torch.ones(300,6)],
            [torch.ones(1200,3),torch.ones(128,3),torch.ones(300,6)],
            [torch.ones(1200,3),torch.ones(128,3),torch.ones(300,6)]
            ]

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

    def forward(
        self,
        x,
        weight_scale,
        mask = None
    ):
        save_log = False
        seq_len, device = x.shape[-2], x.device
        normed_x = self.norm(x)
        x_shift, x_pass = normed_x.chunk(2, dim = -1)
        x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
        qk_s = weight_scale[0]
        hidden_s = weight_scale[1]
        out_s = weight_scale[2]

        normed_x.cuda()
        normed_x = torch.cat((x_shift, x_pass), dim = -1)

        if save_log:
            t_normed_x = normed_x[0].cpu()
            t_normed_x = t_normed_x.numpy()
            np.savetxt('t_normed_x.txt', t_normed_x, fmt='%f')

        normed_x_qk = normed_x / qk_s.cuda()
        normed_x_hidden = normed_x / hidden_s.cuda()

        if save_log:
            t_normed_x_qk = normed_x_qk[0].cpu()
            t_normed_x_qk = t_normed_x_qk.numpy()
            np.savetxt('t_normed_x_qk.txt', t_normed_x_qk, fmt='%f')

        if save_log:
            t_normed_x_hidden = normed_x_hidden[0].cpu()
            t_normed_x_hidden = t_normed_x_hidden.numpy()
            np.savetxt('t_normed_x_hidden.txt', t_normed_x_hidden, fmt='%f')

        if save_log:
            qk_origin = self.to_qk(normed_x)
            t_qk_origin = qk_origin[1].cpu()
            t_qk_origin = t_qk_origin.numpy()
            np.savetxt('t_qk_origin.txt', t_qk_origin, fmt='%f')

        if save_log:
            v_o, g_o = self.to_hidden(normed_x).chunk(2, dim = -1) #v, gate [500,600]
            v_origin = v_o[1].cpu()
            v_origin = v_origin.numpy()
            np.savetxt('v_origin.txt', v_origin, fmt='%f')

        origin_qk_weight = self.to_qk[0].weight.data
        origin_hidden_weight = self.to_hidden[0].weight.data
        origin_out_weight = self.to_out[0].weight.data

        qk_weight = origin_qk_weight
        hidden_weight = origin_hidden_weight
        out_weight = origin_out_weight

        if save_log:
            t_qk_weight_origin = qk_weight.cpu()
            t_qk_weight_origin = t_qk_weight_origin.numpy()
            np.savetxt('t_qk_weight_origin.txt', t_qk_weight_origin, fmt='%f')
            t_hidden_weight_origin = hidden_weight.cpu()
            t_hidden_weight_origin = t_hidden_weight_origin.numpy()
            np.savetxt('t_hidden_weight_origin.txt', t_hidden_weight_origin, fmt='%f')

        qk_weight = qk_weight * qk_s.cuda()
        hidden_weight = hidden_weight * hidden_s.cuda()
        out_weight = out_weight * out_s.cuda()

        self.to_qk[0].weight.data = qk_weight
        self.to_hidden[0].weight.data = hidden_weight

        if save_log:
            t_qk_weight_after = self.to_qk[0].weight.data.cpu()
            t_qk_weight_after = t_qk_weight_after.numpy()
            np.savetxt('t_qk_weight_after.txt', t_qk_weight_after, fmt='%f')
            t_hidden_weight_after = self.to_hidden[0].weight.data.cpu()
            t_hidden_weight_after = t_hidden_weight_after.numpy()
            np.savetxt('t_hidden_weight_after.txt', t_hidden_weight_after, fmt='%f')
        
        # quantize activation
        # q_row_max = torch.max(torch.abs(q), dim=2).values
        # k_row_max = torch.max(torch.abs(k), dim=2).values
        # for batch in range(20):
        #     for i in range(500):
        #         q[batch,i,:] = q[batch,i,:] / q_row_scale[batch,i]
        #         k[batch,i,:] = k[batch,i,:] / k_row_scale[batch,i]
        # q = torch.clamp(q, min=-127, max=127)
        # k = torch.clamp(k, min=-127, max=127)
        # q = q.to(torch.int8)
        # k = k.to(torch.int8)
        # q = q.to(torch.float32)
        # k = k.to(torch.float32)

        v, gate = self.to_hidden(normed_x_hidden).chunk(2, dim = -1) #v, gate [500,600]
        qk = self.to_qk(normed_x_qk) #qk [500,128]

        if save_log:
            v_after = v[1].cpu()
            v_after = v_after.numpy()
            np.savetxt('v_after.txt', v_after, fmt='%f')
            t_qk_after = qk[1].cpu()
            t_qk_after = t_qk_after.numpy()
            np.savetxt('t_qk_after.txt', t_qk_after, fmt='%f')

        q, k = self.offsetscale(qk) #q, k [500,128]
        q, k = map(self.rotary_pos_emb.rotate_queries_or_keys, (q, k)) 
        sim = einsum('b i d, b j d -> b i j', q, k)
        sim = sim + self.rel_pos_bias(sim)
        attn = self.attn_fn(sim / seq_len)
        attn = self.dropout(attn) #attn [500,500]
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 j')
            attn = attn.masked_fill(~mask, 0.)
        if self.causal:
            causal_mask = torch.ones((seq_len, seq_len), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = out * gate #out [500,600]
        out_out = out / out_s.cuda()

        out1 = self.to_out(out)
        
        if save_log:
            out_origin = out1[0].cpu()
            out_origin = out_origin.numpy()
            np.savetxt('out_origin.txt', out_origin, fmt='%f')

        self.to_out[0].weight.data = out_weight
        out = self.to_out(out_out) #out [500,300]

        if save_log:
            out_after = out[0].cpu()
            out_after = out_after.numpy()
            np.savetxt('out_after.txt', out_after, fmt='%f')    

        if self.add_residual:
            out = out + x

        self.to_qk[0].weight.data = origin_qk_weight
        self.to_hidden[0].weight.data = origin_hidden_weight
        self.to_out[0].weight.data = origin_out_weight

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
        self.checkpoint_path = './model.ckpt'
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
        i = 0
        for gau in self.gaus:
            out = gau(out,gau_scales[i])
            i = i + 1
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

# 预先定义配置
config = Config()
train_data,test_data,vocabs_size = load_data(config)#加载数据
config.n_vocab = len(vocabs_size) + 1#补充词表大小，词表一定要多留出来一个
model = Model(config)#调用transformer的编码器
loaded_awq_0 = torch.load('/root/gau/Gau_transformer/models_save/awq_results_0.pt')

gau_scales[0][0] = loaded_awq_0['scale'][0][2]
gau_scales[0][1] = loaded_awq_0['scale'][1][2]
gau_scales[0][2] = loaded_awq_0['scale'][2][2]

gau_clips[0][0] = loaded_awq_0['clip'][0][1]
gau_clips[0][1] = loaded_awq_0['clip'][1][1]
gau_clips[0][2] = loaded_awq_0['clip'][2][1]

def modify_weight(model,level):
    clip_hidden = gau_clips[level][0].cuda()
    clip_qk = gau_clips[level][1].cuda()
    clip_out = gau_clips[level][2].cuda()
    layer = model.gaus[level]
    layer_hidden_data = layer.to_hidden[0].weight.data
    layer_qk_data = layer.to_qk[0].weight.data
    layer_out_data = layer.to_out[0].weight.data
    #modify to_hidden
    clipped_layer_hidden_data = layer_hidden_data
    for row in range(1200):
        for group in range(3):
            clipped_layer_hidden_data[row][group*100 : (group * 100 + 100)] = torch.clamp(clipped_layer_hidden_data[row][group*100 : (group * 100 + 100)], min= -clip_hidden[row][group], max=clip_hidden[row][group])
    model.gaus[level].to_hidden[0].weight.data = clipped_layer_hidden_data

    #modify to_qk
    clipped_layer_qk_data = layer_qk_data
    print('origin qk')
    print(clipped_layer_qk_data)
    for row in range(128):
        for group in range(3):
            clipped_layer_qk_data[row][group*100 : (group * 100 + 100)] = torch.clamp(clipped_layer_qk_data[row][group*100 : (group * 100 + 100)], min= -clip_qk[row][group], max=clip_qk[row][group])
    model.gaus[level].to_qk[0].weight.data = clipped_layer_qk_data

    #modify to_out
    clipped_layer_out_data = layer_out_data
    for row in range(300):
        for group in range(6):
            clipped_layer_out_data[row][group*100 : (group * 100 + 100)] = torch.clamp(clipped_layer_out_data[row][group*100 : (group * 100 + 100)], min= -clip_out[row][group], max=clip_out[row][group])
    model.gaus[level].to_out[0].weight.data = clipped_layer_out_data



#load Model 
stat_dict = torch.load('../models_save/gau_best.pt')
model.load_state_dict({k.replace('net.',''):v for k,v in stat_dict.items()})
model.cuda()
model.eval() # set the model to evaluation mode

modify_weight(model,0)

optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()#多分类的任务
batch_size=config.batch_size

val_acc = 0.0
val_loss = 0.0




# print(loaded_awq_0['scale'][0][2].shape)
# print(loaded_awq_0['scale'][0][2])

# print(loaded_awq_0['scale'])
# print(loaded_awq_0['clip'])
# print('##########################')
# print(loaded_awq_0['clip'][0][1].shape)
# print(loaded_awq_0['clip'][1][1].shape)
# print(loaded_awq_0['clip'][2][1].shape)
# scale 0:to_qk 1:to_hidden 2:to_out
# weight to_hidden: [1200,300] to_qk: [128,300] to_out: [300,600]
# clip 0:to_hidden 1:to_qk 2:to_out
# 0: [1200,3,1] 1:[128,3,1] 2:[300,6,1]

end_point = 1250

with torch.no_grad():
    for i, batch in enumerate(tqdm(test_data)):
        features, labels = batch
        features = features.cuda()
        
        labels = labels.cuda()
        outputs = model(features)

        # output_origin = outputs[0].cpu()
        # output_origin = output_origin.numpy()
        # np.savetxt('output_after.txt', output_origin, fmt='%f')
        loss = criterion(outputs, labels)
        
        _, val_pred = torch.max(outputs, 1) 
        val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
        val_loss += loss.item()
        # print(i)
        # print(val_acc)
        # print(val_loss)
        if i == end_point:
            break
print(f'Val Acc: {val_acc/25000:3.5f} loss: {val_loss/len(test_data):3.5f}')


        