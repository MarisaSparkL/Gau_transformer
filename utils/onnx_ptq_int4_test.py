import onnx
import numpy as np
import onnxruntime as rt
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset
import collections
from torchtext.vocab import vocab, GloVe
from onnx import numpy_helper


model_path = '../models_save/ptq_imdb_gau_onnx.onnx'

# 验证模型合法性
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

#修改参数
all_initializer = onnx_model.graph.initializer

# weights, names = [], []
# for t in all_initializer:
# 	weights.append(numpy_helper.to_array(t))
# 	names.append(t.name)

# target_initializer = 'onnx::MatMul_1574_quantized'
# for i, init in enumerate(all_initializer):
#     if init.name == target_initializer:
#         idx = i
#         break

layers = []
target_initializer = 'onnx::MatMul_1574_quantized'
for i, init in enumerate(all_initializer):
    if 'quantized' in init.name:
        layers.append(i)

# #更改参数的内容
# tensor_proto = onnx_model.graph.initializer[idx]
# # 获取数据类型和维度
# np_dtype = numpy_helper.tensor_dtype_to_np_dtype(tensor_proto.data_type)
# dims = tensor_proto.dims

# # 将raw_data转换为NumPy数组
# prev_data = np.frombuffer(tensor_proto.raw_data, dtype=np_dtype).reshape(dims)

print(layers)
for idx in layers:
    prev_data = onnx.numpy_helper.to_array(onnx_model.graph.initializer[idx])
    # print("prev data")
    # print(prev_data)
    int4_data = (prev_data >> 4) << 4
    # print("after data")
    # print(int4_data)
    onnx_model.graph.initializer[idx].raw_data = int4_data.tobytes()

# prev_data = onnx.numpy_helper.to_array(onnx_model.graph.initializer[idx])
# print("prev data")
# print(prev_data)
# int4_data = prev_data >> 4
# print("after data")
# print(int4_data)
# onnx_model.graph.initializer[idx].raw_data = int4_data.tobytes()

#保存修改后的模型
onnx.save(onnx_model, "../models_save/ptq_gau_int4_test.onnx")
