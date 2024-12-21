import torch

# 创建一个二维矩阵 (假设为 A)
A = torch.tensor([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# 创建一个一维矩阵 (假设为 B)
B = torch.tensor([2, 3, 4])

# 执行逐列相乘
C = A / B  # B.view(-1, 1) 将 B 重塑为 (3, 1) 形状

print(C)