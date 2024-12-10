import numpy as np

# 使用np.loadtxt读取文件
gate = np.loadtxt('gate1.txt', dtype=float)

sub_matrix1 = gate[0:49, 0:49]

print(sub_matrix1)

print(np.min(gate))

print(np.max(gate))

print(np.mean(gate))

# threshold = 1

# bool_matrix = sub_matrix1 > threshold

# print(bool_matrix)
# count = np.sum(bool_matrix)
# print(count)

# np.savetxt('gate_bool.txt', bool_matrix, fmt='%d')

counts = np.zeros((100, 120))

for i in range(100):
    for j in range(120):
        sub_matrix = gate[5*i:(5*i + 4), 5*j:(5*j + 4)]
        #print(sub_matrix)
        bool_matrix = np.abs(sub_matrix) >= 1
        count = np.sum(bool_matrix)
        counts[i][j] = count

print(counts)
np.savetxt('counts.txt', counts, fmt='%d')

print(np.max(counts))
for i in range(25):
    c = np.sum(counts >= i)
    print("i ",i," ","c", c)

# for threshold in range(20):
#     bool_matrix = np.abs(gate) >= threshold
#     count = np.sum(bool_matrix)
#     print(threshold, " ", count, " ", count/300000)