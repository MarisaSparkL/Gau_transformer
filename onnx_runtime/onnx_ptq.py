import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset
import collections
from torchtext.vocab import vocab, GloVe

#加载数据

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

# 假设你已经有了一个 PyTorch DataLoader
# data_loader = ...

# 获取校准数据
# calib_data = calib_data_generator(data_loader)

class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_loader):
        """
        data_loader 是一个生成器或迭代器，返回用于校准的输入数据。
        """
        super().__init__()
        self.data_loader = data_loader

    def get_next(self):
        """
        返回下一个校准数据批次。
        """
        batch = next(self.data_loader, None)
        if batch is None:
            return None
        features, labels = batch
        # example only
        features = np.ascontiguousarray(features, dtype=np.int64)
        return {'modelInput': features}
        return np.ascontiguousarray(features, dtype=np.int32)
    
    def get_input_name(self, index):
        # 假设只有一个输入，直接返回名称
        return 'modelInput'
        
    
        # data = next(self.data_loader, None)
        # if data is None:
        #     return None
        # # 假设 data_loader 返回的是一个NumPy数组
        # # 并且模型只有一个输入
        # input_name = self.get_input_name(0)
        # return {input_name: data.astype(np.float32)}

config = Config()
train_iter,test_iter,vocabs_size = load_data(config)#加载数据
# test_data = ImdbDataset(folder_path="../aclImdb", is_train=False)
# test_set = TensorDataset(*preprocess_imdb(test_data, vocab,config))
# test_iter = DataLoader(test_set, config.batch_size)

calibration_data_reader = MyCalibrationDataReader(iter(test_iter))

# 加载 FP32 模型
model_fp32 = "../models_save/imdb_gau_best.onnx"
onnx_model = onnx.load(model_fp32)

quantized_nodes = ['/gaus.0/to_hidden/to_hidden.0/MatMul','/gaus.0/to_qk/to_qk.0/MatMul','/gaus.0/to_out/to_out.0/MatMul',
'/gaus.1/to_hidden/to_hidden.0/MatMul','/gaus.1/to_qk/to_qk.0/MatMul','/gaus.1/to_out/to_out.0/MatMul',
'/gaus.2/to_hidden/to_hidden.0/MatMul','/gaus.2/to_qk/to_qk.0/MatMul','/gaus.2/to_out/to_out.0/MatMul',
'/gaus.3/to_hidden/to_hidden.0/MatMul','/gaus.3/to_qk/to_qk.0/MatMul','/gaus.3/to_out/to_out.0/MatMul'
]

# 创建量化配置，指定量化模式为静态量化
quantization_config = {
    "activation_type": "None",  # 激活函数量化类型
    "weight_type": "int8",      # 权重量化类型
    "mode": "static",            # 静态量化
    "op_types_to_quantize": ['Gemm','MatMul'],
    "nodes_to_quantize": quantized_nodes,
    'quant_format': QuantFormat.QDQ, 
    #'quant_format': "QDQ",       # 量化格式
    'calibration_method': "entropy"  # 校准方法
}

# 执行量化
#quantized_model = quantize_static(model_fp32, quantization_config)
model_quant_path = "../models_save/ptq_imdb_gau_onnx.onnx"

# # 指定量化配置，例如使用 Entropy 方法
# quant_format = 'QDQ'
# calib_method = 'entropy'

# 执行静态量化
# model_quant = quantize_static(
#     onnx_model,
#     model_quant_path,
#     calibration_data_reader,
#     quantization_config
# )

# 保存量化后的模型
#onnx.save(model_quant, model_quant_path)

# model_quant = quantize_static(
#     model_fp32,
#     model_quant_path,
#     calibration_data_reader,
#     quantization_config
# )

model_quant = quantize_static(
    model_fp32,
    
    model_quant_path,
    calibration_data_reader,

    op_types_to_quantize = ['Gemm','MatMul'],
    nodes_to_quantize = quantized_nodes,
    weight_type = QuantType.QInt8
    # activation_type = QuantType.QInt16

)

onnx.save(model_quant, model_quant_path)



