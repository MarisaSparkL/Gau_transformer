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

#GAU定义相关
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

gau_clips = [[torch.ones(1200),torch.ones(128),torch.ones(300)],
            [torch.ones(1200),torch.ones(128),torch.ones(300)],
            [torch.ones(1200),torch.ones(128),torch.ones(300)],
            [torch.ones(1200),torch.ones(128),torch.ones(300)]
            ]

weight_quant_params = [
    [torch.ones(1200),torch.ones(128),torch.ones(300)],
    [torch.ones(1200),torch.ones(128),torch.ones(300)],
    [torch.ones(1200),torch.ones(128),torch.ones(300)],
    [torch.ones(1200),torch.ones(128),torch.ones(300)]
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
        quant_params,
        mask = None
    ):
        with torch.no_grad():
            qk_s = weight_scale[0]
            hidden_s = weight_scale[1]
            out_s = weight_scale[2]
            
            hidden_params = quant_params[0].cuda()
            qk_params = quant_params[1].cuda()
            out_params = quant_params[2].cuda()

            seq_len, device = x.shape[-2], x.device
            normed_x = self.norm(x)
            # print('normed_x')
            # print(normed_x)
            
            # print('normed_x_qk')
            # print(normed_x_qk)
            # print('normed_x_hidden')
            # print(normed_x_hidden)

            x_shift, x_pass = normed_x.chunk(2, dim = -1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)

            normed_x.cuda()
            normed_x = torch.cat((x_shift, x_pass), dim = -1)

            normed_x_qk = normed_x / (qk_s.view(1,-1).cuda())
            normed_x_hidden = normed_x / (hidden_s.view(1,-1).cuda())

            # #quantize_normed_x
            # normed_x_row_max = torch.max(torch.abs(normed_x), dim=2).values
            # normed_x_row_scale = normed_x_row_max / 127
            # for batch in range(20):
            #     for i in range(500):
            #         normed_x[batch,i,:] = normed_x[batch,i,:] / normed_x_row_scale[batch,i]
            # normed_x = torch.clamp(normed_x, min=-127, max=127)
            # normed_x = normed_x.to(torch.int8)
            # normed_x = normed_x.to(torch.float32)

            #quantize_normed_x_qk
            normed_x_qk_row_max = torch.max(torch.abs(normed_x_qk), dim=2).values
            normed_x_qk_row_scale = normed_x_qk_row_max / 15
            for batch in range(20):
                for i in range(500):
                    normed_x_qk[batch,i,:] = normed_x_qk[batch,i,:] / normed_x_qk_row_scale[batch,i]
            normed_x_qk = torch.clamp(normed_x_qk, min=-15, max=15)
            normed_x_qk = normed_x_qk.to(torch.int8)
            normed_x_qk = normed_x_qk.to(torch.float32)

            #quantize_normed_x_hidden
            normed_x_hidden_row_max = torch.max(torch.abs(normed_x_hidden), dim=2).values
            normed_x_hidden_row_scale = normed_x_hidden_row_max / 15
            for batch in range(20):
                for i in range(500):
                    normed_x_hidden[batch,i,:] = normed_x_hidden[batch,i,:] / normed_x_hidden_row_scale[batch,i]
            normed_x_hidden = torch.clamp(normed_x_hidden, min=-15, max=15)
            normed_x_hidden = normed_x_hidden.to(torch.int8)
            normed_x_hidden = normed_x_hidden.to(torch.float32)

            v_gate = self.to_hidden(normed_x_hidden)
            # print('v_gate')
            # print(v_gate)
            #v_gate反量化
            for batch in range(20):
                v_gate_quant_matrix = torch.outer(normed_x_hidden_row_scale[batch], hidden_params)
                v_gate[batch] = v_gate[batch] * v_gate_quant_matrix

            v, gate = v_gate.chunk(2, dim = -1) #v, gate [500,600]
            qk = self.to_qk(normed_x_qk) #qk [500,128]
            # print('qk')
            # print(qk)
            #qk反量化
            for batch in range(20):
                qk_quant_matrix = torch.outer(normed_x_qk_row_scale[batch], qk_params)
                qk[batch] = qk[batch] * qk_quant_matrix

            q, k = self.offsetscale(qk) #q, k [500,128]
            q, k = map(self.rotary_pos_emb.rotate_queries_or_keys, (q, k)) 

            # quantize activation
            q_row_max = torch.max(torch.abs(q), dim=2).values
            k_row_max = torch.max(torch.abs(k), dim=2).values
            q_row_scale = q_row_max / 15
            k_row_scale = k_row_max / 15
            for batch in range(20):
                for i in range(500):
                    q[batch,i,:] = q[batch,i,:] / q_row_scale[batch,i]
                    k[batch,i,:] = k[batch,i,:] / k_row_scale[batch,i]
            q = torch.clamp(q, min=-15, max=15)
            k = torch.clamp(k, min=-15, max=15)
            q = q.to(torch.int8)
            k = k.to(torch.int8)
            q = q.to(torch.float32)
            k = k.to(torch.float32)

            sim = einsum('b i d, b j d -> b i j', q, k)
            sim = sim.to(torch.int32)
            sim = sim.to(torch.float32)

            # print('sim')
            # print(sim)

            #sim反量化
            for batch in range(20):
                sim_quant_matrix = torch.outer(q_row_scale[batch], k_row_scale[batch])
                sim[batch] = sim[batch] * sim_quant_matrix

            sim = sim + self.rel_pos_bias(sim)
            attn = self.attn_fn(sim / seq_len)
            attn = self.dropout(attn) #attn [500,500]
            if exists(mask):
                mask = rearrange(mask, 'b j -> b 1 j')
                attn = attn.masked_fill(~mask, 0.)
            if self.causal:
                causal_mask = torch.ones((seq_len, seq_len), dtype = torch.bool, device = device).triu(1)
                attn = attn.masked_fill(causal_mask, 0.)

            #attn quantize
            attn_row_max = torch.max(torch.abs(attn), dim=2).values
            v_col_max = torch.max(torch.abs(v), dim=1).values

            attn_row_scale = attn_row_max / 15
            v_col_scale = v_col_max / 15

            for batch in range(20):
                for i in range(500):
                    attn[batch,i,:] = attn[batch,i,:] / attn_row_scale[batch,i]
                for i in range(600):
                    v[batch,:,i] = v[batch,:,i] / v_col_scale[batch,i]

            attn = torch.clamp(attn, min=-15, max=15)
            v = torch.clamp(v, min=-15, max=15)
            attn = attn.to(torch.int8)
            v = v.to(torch.int8)
            attn = attn.to(torch.float32)
            v = v.to(torch.float32)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = out.to(torch.int32)
            out = out.to(torch.float32)

            #out反量化
            for batch in range(20):
                out_quant_matrix = torch.outer(attn_row_scale[batch], v_col_scale[batch])
                out[batch] = out[batch] * out_quant_matrix

            out = out * gate #out [500,600]
            out = out / (out_s.view(1,-1).cuda())

            #out重量化
            out_row_max = torch.max(torch.abs(out), dim=2).values
            out_row_scale = out_row_max / 15
            for batch in range(20):
                for i in range(500):
                    out[batch,i,:] = out[batch,i,:] / out_row_scale[batch,i]
            out = torch.clamp(out, min=-15, max=15)
            out = out.to(torch.int8)
            out = out.to(torch.float32)

            out = self.to_out(out) #out [500,300]
            #to_out反量化
            for batch in range(20):
                out_quant_matrix = torch.outer(out_row_scale[batch], out_params)
                out[batch] = out[batch] * out_quant_matrix

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
        self.checkpoint_path = './model.ckpt'
        self.query_key_dim = 300

#读取数据集相关
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
            out = gau(out,gau_scales[i],weight_quant_params[i])
            i = i + 1
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.fc1(out)
        return out

def modify_layer_weight_channel(model,level):
    weight_scale = gau_scales[level]
    clip_hidden = gau_clips[level][0].cuda()
    clip_qk = gau_clips[level][1].cuda()
    clip_out = gau_clips[level][2].cuda()
    layer = model.gaus[level]
    layer_hidden_data = layer.to_hidden[0].weight.data.cuda()
    layer_qk_data = layer.to_qk[0].weight.data.cuda()
    layer_out_data = layer.to_out[0].weight.data.cuda()
    #modify scale
    qk_s = weight_scale[0]
    hidden_s = weight_scale[1]
    out_s = weight_scale[2]

    layer_hidden_data = layer_hidden_data * hidden_s.cuda()
    layer_qk_data = layer_qk_data * qk_s.cuda()
    layer_out_data = layer_out_data * out_s.cuda()

    #modify to_hidden clip
    for row in range(1200):
        layer_hidden_data[row] = torch.clamp(layer_hidden_data[row], min= -torch.max(clip_hidden[row]), max=torch.max(clip_hidden[row]))
    
    #modify to_qk clip
    for row in range(128):
        layer_qk_data[row] = torch.clamp(layer_qk_data[row], min= -torch.max(clip_qk[row]), max=torch.max(clip_qk[row]))
    
    #modify to_out clip
    for row in range(300):
        layer_out_data[row] = torch.clamp(layer_out_data[row], min= -torch.max(clip_out[row]), max=torch.max(clip_out[row]))
    
    #quantize weight
    for row in range(1200):
        t = layer_hidden_data[row]
        t_max = torch.max(abs(t))
        t_quant_scale = t_max / 15
        t = t / t_quant_scale
        t = t.to(torch.int8)
        t = t.to(torch.float32)
        layer_hidden_data[row] = t
        weight_quant_params[level][0][row] = t_quant_scale

    for row in range(128):
        t = layer_qk_data[row]
        t_max = torch.max(abs(t))
        t_quant_scale = t_max / 15
        t = t / t_quant_scale
        t = t.to(torch.int8)
        t = t.to(torch.float32)
        layer_qk_data[row] = t
        weight_quant_params[level][1][row] = t_quant_scale

    for row in range(300):
        t = layer_out_data[row]
        t_max = torch.max(abs(t))
        t_quant_scale = t_max / 15
        t = t / t_quant_scale
        t = t.to(torch.int8)
        t = t.to(torch.float32)
        layer_out_data[row] = t
        weight_quant_params[level][2][row] = t_quant_scale
    
    #save modified data
    model.gaus[level].to_hidden[0].weight.data = layer_hidden_data
    model.gaus[level].to_qk[0].weight.data = layer_qk_data
    model.gaus[level].to_out[0].weight.data = layer_out_data


loaded_awq_0 = torch.load('/root/gau/Gau_transformer/models_save/awq_results_w4_0.pt')
loaded_awq_1 = torch.load('/root/gau/Gau_transformer/models_save/awq_results_w4_1.pt')
loaded_awq_2 = torch.load('/root/gau/Gau_transformer/models_save/awq_results_w4_2.pt')
loaded_awq_3 = torch.load('/root/gau/Gau_transformer/models_save/awq_results_w4_3.pt')

# gau_scales[0][0] = loaded_awq_0['scale'][0][2]
# gau_scales[0][1] = loaded_awq_0['scale'][1][2]
# gau_scales[0][2] = loaded_awq_0['scale'][2][2]

gau_clips[0][0] = loaded_awq_0['clip'][0][1]
gau_clips[0][1] = loaded_awq_0['clip'][1][1]
gau_clips[0][2] = loaded_awq_0['clip'][2][1]

# gau_scales[1][0] = loaded_awq_1['scale'][0][2]
# gau_scales[1][1] = loaded_awq_1['scale'][1][2]
# gau_scales[1][2] = loaded_awq_1['scale'][2][2]

gau_clips[1][0] = loaded_awq_1['clip'][0][1]
gau_clips[1][1] = loaded_awq_1['clip'][1][1]
gau_clips[1][2] = loaded_awq_1['clip'][2][1]

# gau_scales[2][0] = loaded_awq_2['scale'][0][2]
# gau_scales[2][1] = loaded_awq_2['scale'][1][2]
# gau_scales[2][2] = loaded_awq_2['scale'][2][2]

gau_clips[2][0] = loaded_awq_2['clip'][0][1]
gau_clips[2][1] = loaded_awq_2['clip'][1][1]
gau_clips[2][2] = loaded_awq_2['clip'][2][1]

# gau_scales[3][0] = loaded_awq_3['scale'][0][2]
# gau_scales[3][1] = loaded_awq_3['scale'][1][2]
# gau_scales[3][2] = loaded_awq_3['scale'][2][2]

gau_clips[3][0] = loaded_awq_3['clip'][0][1]
gau_clips[3][1] = loaded_awq_3['clip'][1][1]
gau_clips[3][2] = loaded_awq_3['clip'][2][1]

print('=========================')
print(gau_scales[0][0])
print(gau_clips[0][0])

# 预先定义配置
config = Config()
train_data,test_data,vocabs_size = load_data(config)#加载数据
config.n_vocab = len(vocabs_size) + 1#补充词表大小，词表一定要多留出来一个
model = Model(config)#调用transformer的编码器

#load Model 
stat_dict = torch.load('../models_save/gau_best.pt')
model.load_state_dict({k.replace('net.',''):v for k,v in stat_dict.items()})
model.cuda()
model.eval() # set the model to evaluation mode

modify_layer_weight_channel(model,0)
modify_layer_weight_channel(model,1)
modify_layer_weight_channel(model,2)
modify_layer_weight_channel(model,3)

optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()#多分类的任务
batch_size=config.batch_size

val_acc = 0.0
val_loss = 0.0

end_point = 1125

with torch.no_grad():
    for i, batch in enumerate(tqdm(test_data)):
        if i < 1000:
            continue
        features, labels = batch
        features = features.cuda()
        
        labels = labels.cuda()
        outputs = model(features)

        # print('outputs')
        # print(outputs)
        # print('labels')
        # print(labels)

        # output_origin = outputs[0].cpu()
        # output_origin = output_origin.numpy()
        # np.savetxt('output_after.txt', output_origin, fmt='%f')
        loss = criterion(outputs, labels)
        
        _, val_pred = torch.max(outputs, 1) 
        val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
        val_loss += loss.item()
        print(i)
        print(val_acc)
        print(val_loss)
        if i == end_point:
            break
print(f'Val Acc: {val_acc/2500:3.5f} loss: {val_loss/len(test_data):3.5f}')