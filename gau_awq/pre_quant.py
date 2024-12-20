import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict
from typing import List

from .auto_scale import auto_scale_block, apply_scale
from .auto_clip import auto_clip_block, apply_clip

__all__ = ["run_awq"]


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def get_blocks(model):
    layers = model.gaus
    return layers

@torch.no_grad()
def run_awq(
    model,
    w_bit,
    q_config,
    n_samples=32,
    seqlen=32,
    auto_scale=True,
    mse_range=True,
):
    from .awq_utils import get_calib_dataset
    from .awq_utils import get_op_name
    from .awq_utils import append_str_prefix

    layers = get_blocks(model)

    samples = get_calib_dataset(
        n_samples=n_samples, block_size=seqlen
    )

    
    samples = samples[0].cuda()
    # samples = torch.cat(samples, dim=0)

    inps = []

    layers[0] = layers[0].cuda()

    # get input and kwargs to layer 0
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
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

    # layers[0] = layers[0].cpu()

    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {
        "scale": [],
        "clip": [],
    }

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)
        print("--------------------")
        print(named_linears)

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
        # get output as next layer's input
        # print("###########")
        # print(layer)
        # print(inps.shape)
        inps = layer(inps)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        # print("input_feat")
        # print(input_feat)
        # # 将张量转换为字符串
        # str_dict = {k: ' '.join(map(str, v.tolist())) for k, v in input_feat.items()}
        # # 保存字典为文本文件
        # with open('input_feat_tensor_dict.txt', 'w') as f:
        #     for key, value in str_dict.items():
        #         f.write(f"{key}: {value}\n")
        # return
        # print("#########")
        # for k, v in input_feat.items():
        #     print(k)
        #     print(v.shape)

        # Clear GPU memory
        torch.cuda.empty_cache()

        if (
            auto_scale
        ):  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(
                layer,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            # apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model, layer) + "."
            )

        # Clear GPU memory
        torch.cuda.empty_cache()

        if mse_range:
            clip_list = auto_clip_block(
                layer,
                w_bit=w_bit,
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

    return awq_results


def apply_awq(model, awq_results):
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])
