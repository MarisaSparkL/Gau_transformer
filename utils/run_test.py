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

model_path = '../models_save/imdb_gau_lera.onnx'

# 验证模型合法性
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

config = Config()
train_data,test_data,vocabs_size = load_data(config)

sess = rt.InferenceSession(model_path)

val_acc = 0.0
val_loss = 0.0
batch_num = 0

for i, batch in enumerate(tqdm(test_data)):
    batch_num = max(batch_num,i)
    features, labels = batch
    features = features.cuda()
    
    labels = labels.cuda()
    features = features.detach().cpu().numpy()

    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: features})

    outputs = np.array(outputs)
    #outputs = torch.Tensor(outputs)
    outputs = torch.from_numpy(outputs)
    outputs = torch.squeeze(outputs).cuda()

    criterion = nn.CrossEntropyLoss()#多分类的任务
    loss = criterion(outputs, labels) 
    
    _, val_pred = torch.max(outputs, 1) 
    tmp_num = (val_pred.cpu() == labels.cpu()).sum().item()
    val_acc += tmp_num # get the index of the class with the highest probability
    #print(tmp_num)
    val_loss += loss.item()

print("batch_num")
print(batch_num)
print(f'Val Acc: {val_acc/25000:3.5f} loss: {val_loss/len(test_data):3.5f}')