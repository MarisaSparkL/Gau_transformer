import torch
from torch import nn
import torchvision
from torchvision import models
import tensorrt as trt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# 加载预训练模型
model = models.vgg16(pretrained=True)

# 准备数据集
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=models.vgg16.default_transform())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

# 将模型转换为评估模式
model.eval()

# 将模型转换为ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "vgg16_cifar10.onnx", opset_version=11)

# 加载ONNX模型并进行PTQ量化
onnx_model = trt.OnnxParser().parse("vgg16_cifar10.onnx", trt.Logger(trt.Logger.WARNING))

# 创建TensorRT引擎构建器并设置PTQ参数
with trt.Builder(trt.Logger(trt.Logger.WARNING)) as builder, builder.create_network() as network:
    builder.max_workspace_size = 1 << 30
    builder.max_batch_size = 64
    builder.fp16_mode = False

    # 使用校准数据集进行PTQ量化
    with builder.create_builder_config() as config:
        config.flags = 1 << int(trt.BuilderFlag.QUANTIZE_WEIGHTS) | \
                       1 << int(trt.BuilderFlag.QUANTIZE_ACTIVATIONS)
        
        # 选择校准方法，这里使用EntropyCalibratorV2
        calibrator = trt.EntropyCalibratorV2(
            data_loader=train_loader,
            batch_size=64,
            cache_file='calibration_cache.txt',
            quantile=0.99
        )
        config.int8_calibrator = calibrator

    # 将ONNX模型解析到网络中
    network.add_onnx_parser(onnx_model)

    # 构建引擎
    engine = builder.build_engine(network, config)

# 保存构建的引擎
with open('vgg16_cifar10_trt.engine', 'wb') as f:
    f.write(engine.serialize())