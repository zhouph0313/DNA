import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import time

# 检查是否可以使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
# 配置
filenames = ['C:/Users/zhoupenghua/Desktop/ldpcdna.txt',
             'C:/Users/zhoupenghua/Desktop/ldpcdna2.txt',
             'C:/Users/zhoupenghua/Desktop/ldpcdna3.txt',
             'C:/Users/zhoupenghua/Desktop/ldpcdna4.txt',
             'C:/Users/zhoupenghua/Desktop/ldpcdna5.txt']
num_classes = len(filenames)  # 类别数，等于文件数
max_dataset_size_per_class = 100000  # 每类数据的最大样本数
min_length = 100  # 最小截取长度
max_length = 400  # 最大截取长度
padding_value = -1  # 填充值
error_rate = 0.35 # 设置错误率

# 定义自定义数据集
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float).to(device),  # 输入数据转移到GPU
            torch.tensor(self.labels[idx], dtype=torch.float).to(device)  # One-hot 标签转移到GPU
        )

def introduce_errors(sequence, error_rate):
    for i in range(len(sequence)):
        if sequence[i] != padding_value:  # 如果不是填充值才进行替换
            if random.random() < error_rate:  # 用设定的错误率来决定是否替换
                possible_replacements = [x for x in range(4) if x != sequence[i]]  # 假设值为 0, 1, 2, 3
                sequence[i] = random.choice(possible_replacements)  # 随机选择一个替代值
    return sequence

# 读取文件并构造数据集
sequences = []
labels = []

# 从每个文件均匀采样
for label in range(num_classes):
    with open(filenames[label], 'r') as file:
        content = file.read().strip()  # 读取文件内容为一整串序列
        sequence_length = len(content)

        # 从当前文件均匀选择样本
        for _ in range(max_dataset_size_per_class):
            if sequence_length < max_length:
                print(f"Warning: Sequence length {sequence_length} is shorter than max_length {max_length}.")
                continue  # 跳过该类

            start_index = np.random.randint(0, sequence_length - max_length + 1)
            sub_length = np.random.randint(min_length, max_length + 1)  # 随机选择截取长度
            end_index = start_index + sub_length

            # 确保截取不会超出范围
            if end_index <= sequence_length:
                encoded_sequence = [int(x) for x in content[start_index:end_index]]  # 将截取的部分转为整数
                if len(encoded_sequence) < max_length:
                    encoded_sequence += [padding_value] * (max_length - len(encoded_sequence))  # 使用填充值

                # 加入错误替换机制
                encoded_sequence = introduce_errors(encoded_sequence, error_rate)

                sequences.append(encoded_sequence)

                # 创建 one-hot标签
                one_hot_label = [0] * num_classes
                one_hot_label[label] = 1
                labels.append(one_hot_label)

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.1, random_state=42)

# 转换为张量
train_dataset = SequenceDataset(X_train, y_train)
test_dataset = SequenceDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义改进后的 CNN 模型
class AdvancedCNNModel(nn.Module):
    def __init__(self, input_channels, sequence_length, num_classes):
        super(AdvancedCNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # 池化层

        # 计算线性层的输入大小
        flatten_size = 64 * (sequence_length // 4)  # 每个卷积层后卷积核数量和池化减半
        self.fc1 = nn.Linear(flatten_size, 128)  # 全连接层 1
        self.fc2 = nn.Linear(128, num_classes)  # 分类层

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加通道维度
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # 池化层
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # 池化层

        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(self.relu(self.fc1(x)))  # 第一个全连接层和 dropout
        x = self.fc2(x)  # 分类层
        return x

def masked_loss(output, target, mask):
    criterion = nn.CrossEntropyLoss(reduction='none')  # 不进行自动平均
    loss = criterion(output, target)  # 计算损失
    loss = loss * mask  # 应用 mask
    return loss.sum() / mask.sum()  # 归一化

# 训练模型
def train_model(model, train_loader, test_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.0002)  # 优化器
    train_accuracy_list = []
    test_accuracy_list = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # 将 batch_X 和 batch_y 转移到GPU
            optimizer.zero_grad()
            outputs = model(batch_X)

            # 为每个序列生成掩码
            mask = (batch_X != padding_value).float().to(device)  # 将 mask 转移到GPU
            mask = mask.view(-1, 1)  # 扩展为二维

            loss = masked_loss(outputs, torch.argmax(batch_y, dim=1).to(device), mask)  # 使用 masked loss
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == torch.argmax(batch_y, dim=1).to(device)).sum().item()
        print(loss)

        train_accuracy = correct / total
        train_accuracy_list.append(train_accuracy)

        # 测试模型
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # 将 batch_X 和 batch_y 转移到GPU
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == torch.argmax(batch_y, dim=1).to(device)).sum().item()

                # 收集所有的预测和真实标签
                all_preds.extend(predicted.tolist())
                all_targets.extend(torch.argmax(batch_y, dim=1).tolist())

        test_accuracy = correct / total
        test_accuracy_list.append(test_accuracy)

        print(f'Epoch [{epoch + 1}/{epochs}], Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

    return train_accuracy_list, test_accuracy_list, all_preds, all_targets

start0 = time.time()
# 创建和训练模型
input_channels = 1  # 输入通道数
model = AdvancedCNNModel(input_channels=input_channels, sequence_length=max_length, num_classes=num_classes).to(device)  # 转移模型到GPU
train_accuracy_list, test_accuracy_list, all_preds, all_targets = train_model(model, train_loader, test_loader, epochs=30)
end0 = time.time()
print(end0 - start0)

# 保存模型
torch.save(model.state_dict(), 'parameteradvanced0.35_cnn_model100000.pth')

# 保存准确率数据
accuracy_data = pd.DataFrame({
    'Epoch': range(1, len(train_accuracy_list) + 1),
    'Train Accuracy': train_accuracy_list,
    'Test Accuracy': test_accuracy_list
})
accuracy_data.to_csv('parameteradvanced0.35_cnn_model_accuracy_data100000.csv', index=False)

# 计算各项指标和每种类别的指标
precision = precision_score(all_targets, all_preds, average=None, labels=list(range(num_classes)))
recall = recall_score(all_targets, all_preds, average=None, labels=list(range(num_classes)))
f1 = f1_score(all_targets, all_preds, average=None, labels=list(range(num_classes)))

# 保存每种类别的评价指标
class_metrics = pd.DataFrame({
    'Class': [f'Class {i}' for i in range(num_classes)],
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
})

class_metrics.to_csv('parameteradvanced0.35_cnn_model_class_metrics100000.csv', index=False)

# 生成混淆矩阵
conf_matrix = confusion_matrix(all_targets, all_preds)
name = ["LDPC1", "LDPC2", "LDPC3", "LDPC4",'LDPC5']
# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=name, yticklabels=name)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('parameteradvanced0.35_cnn_model_confusion_matrix100000.png')  # 保存混淆矩阵图像
plt.show()