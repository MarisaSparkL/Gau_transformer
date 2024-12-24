import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict
from typing import List
#---------------------------------------------------
import pandas as pd
from collections import Counter
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch import autograd, einsum
import os
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

from .pre_quant import run_awq, apply_awq
from .quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)

from .auto_scale import auto_scale_block, apply_scale
from .auto_clip import auto_clip_block, apply_clip

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
        seq_len, device = x.shape[-2], x.device
        normed_x = self.norm(x)
        #do token shifts
        x_shift, x_pass = normed_x.chunk(2, dim = -1)
        x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
        qk_s = weight_scale[0]
        hidden_s = weight_scale[1]
        out_s = weight_scale[2]
        # qk_s.cuda()
        # hidden_s.cuda()
        # out_s.cuda()
        normed_x.cuda()
        normed_x = torch.cat((x_shift, x_pass), dim = -1)
        normed_x_qk = normed_x.div_(qk_s.view(1,-1).cuda())
        normed_x_hidden = normed_x.div_(hidden_s.view(1,-1).cuda())

        v, gate = self.to_hidden(normed_x).chunk(2, dim = -1) #v, gate [500,600]
        qk = self.to_qk(normed_x) #qk [500,128]
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
        out_out = out.div_(out_s.view(1,-1).cuda())
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
        self.load_quant = False
        self.no_zero_point = False
        self.q_group_size = 100
        self.w_bit = 8
        self.run_awq = True
        self.dump_awq = '/root/gau/Gau_transformer/models_save/gau_best_awq_data'
        self.load_awq = '/root/gau/Gau_transformer/models_save/gau_best_awq.pt'
        self.model_path = '/root/gau/Gau_transformer/models_save/gau_best.pt'

#读取数据集相关
torch.manual_seed(1234)

gau_scales = [[torch.ones(300),torch.ones(300),torch.ones(600)],
            [torch.ones(300),torch.ones(300),torch.ones(600)],
            [torch.ones(300),torch.ones(300),torch.ones(600)],
            [torch.ones(300),torch.ones(300),torch.ones(600)]
            ]
# gau_scales.cuda()

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
        # out = torch.mean(out, 1)
        out = self.fc1(out)
        return out

# 预先定义配置
config = Config()

# train_data,test_data,vocabs_size = load_data(config)#加载数据
# config.n_vocab = len(vocabs_size) + 1 #补充词表大小，词表一定要多留出来一个
config.n_vocab = 46152

model = Model(config)#调用transformer的编码器

#load Model 
stat_dict = torch.load('/root/gau/Gau_transformer/models_save/gau_best.pt')
model.load_state_dict({k.replace('net.',''):v for k,v in stat_dict.items()})

q_config = {
    "zero_point": not config.no_zero_point,  # by default True
    "q_group_size": config.q_group_size,  # whether to use group quantization
}

from .awq_utils import get_calib_dataset
from .awq_utils import get_op_name
from .awq_utils import append_str_prefix

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def get_blocks(model):
    layers = model.gaus
    return layers

def get_scale_and_clip(model,layers,inps,level):    

    awq_results = {
        "scale": [],
        "clip": [],
    }

    torch.cuda.empty_cache()
    layer = layers[level]
    layer = layer.cuda()
    named_linears = get_named_linears(layer)

    # firstly, get input features of all linear layers
    def cache_input_hook(m, x, y, name, feat_dict):
        x = x[0]
        x = x.detach().cpu()
        feat_dict[name].append(x)

    input_feat = defaultdict(list)
    handles = []
    for name in named_linears:
        handles.append(
            named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
            )
        )
    gc.collect()
    torch.cuda.empty_cache()
    inps = layer(inps,gau_scales[level])
    for h in handles:
        h.remove()
    # now solve for scaling and clipping
    input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
    torch.cuda.empty_cache()
    scales_list = auto_scale_block(
        layer,
        w_bit=8,
        q_config=q_config,
        input_feat=input_feat,
    )
    awq_results["scale"] += append_str_prefix(
        scales_list, get_op_name(model, layer) + "."
    )
    # Clear GPU memory
    torch.cuda.empty_cache()
    clip_list = auto_clip_block(
        layer,
        w_bit=8,
        q_config=q_config,
        input_feat=input_feat,
    )
    apply_clip(layer, clip_list)
    # append prefix to make names global
    awq_results["clip"] += append_str_prefix(
        clip_list, get_op_name(model, layer) + "."
    )
    layer = layer.cpu()
    del input_feat
    gc.collect()
    torch.cuda.empty_cache()

    dict_str = '\n'.join(f'{k}: {v}' for k, v in awq_results.items())
    with open('awq_results_dict.txt', 'w') as f:
        f.write(dict_str)
    torch.cuda.empty_cache()
    return awq_results

def main():
    config = Config()
    config.n_vocab = 46152
    model = Model(config)#调用transformer的编码器
    stat_dict = torch.load('/root/gau/Gau_transformer/models_save/gau_best.pt')
    model.load_state_dict({k.replace('net.',''):v for k,v in stat_dict.items()})
    model.eval()
    model.cuda()
    layers = get_blocks(model)
    samples = get_calib_dataset(
        n_samples=16, block_size=16
    )
    samples = samples.cuda()
    layers[0] = layers[0].cuda()

    inps = []
    # get input and kwargs to layer 0
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, weight_scale = None, **kwargs):
            inps.append(inp)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples)
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    gc.collect()
    torch.cuda.empty_cache()
    awq_results_0 = get_scale_and_clip(model,layers,inps,0)
    awq_results_1 = get_scale_and_clip(model,layers,inps,1)
    awq_results_2 = get_scale_and_clip(model,layers,inps,2)
    awq_results_3 = get_scale_and_clip(model,layers,inps,3)

    torch.save(awq_results_0, '/root/gau/Gau_transformer/models_save/awq_results_0.pt')
    torch.save(awq_results_1, '/root/gau/Gau_transformer/models_save/awq_results_1.pt')
    torch.save(awq_results_2, '/root/gau/Gau_transformer/models_save/awq_results_2.pt')
    torch.save(awq_results_3, '/root/gau/Gau_transformer/models_save/awq_results_3.pt')

    #scale_list,clip_list = get_scale_and_clip(model)

if __name__ == "__main__":
    main()