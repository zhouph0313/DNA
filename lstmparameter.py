import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
from torch.nn.utils.rnn import pad_sequence

# 配置
filenames = ['C:/Users/zhoupenghua/Desktop/ldpcdna.txt',
             'C:/Users/zhoupenghua/Desktop/ldpcdna2.txt',
             'C:/Users/zhoupenghua/Desktop/ldpcdna3.txt',
             'C:/Users/zhoupenghua/Desktop/ldpcdna4.txt',
             'C:/Users/zhoupenghua/Desktop/ldpcdna5.txt']
num_classes = len(filenames)
max_length = 400
min_length = 100
max_samples_per_class = 100000
error_rate = 0.4


# 定义自定义数据集
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, lengths):
        self.sequences = sequences
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float),
            torch.tensor(self.labels[idx], dtype=torch.float),
            self.lengths[idx]
        )

    # 引入错误替换机制


def introduce_errors(sequence, error_rate):
    for i in range(len(sequence)):
        if random.random() < error_rate:
            possible_replacements = [x for x in range(num_classes)]
            sequence[i] = random.choice(possible_replacements)
    return sequence


# 读取文件并构造数据集
sequences = []
labels = []
lengths = []

for label in range(num_classes):
    with open(filenames[label], 'r') as file:
        content = file.read().strip()
        sequence_length = len(content)
        current_samples_count = 0

        while current_samples_count < max_samples_per_class:
            if sequence_length < min_length:
                print(f"Warning: Sequence length {sequence_length} is shorter than min_length {min_length}.")
                break

            start_index = np.random.randint(0, sequence_length - max_length + 1)
            sub_length = np.random.randint(min_length, max_length + 1)
            end_index = start_index + sub_length

            if end_index <= sequence_length:
                encoded_sequence = [int(x) for x in content[start_index:end_index]]
                encoded_sequence = introduce_errors(encoded_sequence, error_rate)

                sequences.append(encoded_sequence)
                one_hot_label = [0] * num_classes
                one_hot_label[label] = 1
                labels.append(one_hot_label)
                lengths.append(len(encoded_sequence))
                current_samples_count += 1

            # 划分训练集与测试集
X_train, X_test, y_train, y_test, lengths_train, lengths_test = train_test_split(
    sequences, labels, lengths, test_size=0.1, random_state=42
)

# 转换为张量并维持适当形状
train_dataset = SequenceDataset(X_train, y_train, lengths_train)
test_dataset = SequenceDataset(X_test, y_test, lengths_test)


def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    padded_sequences = pad_sequence([torch.tensor(seq, dtype=torch.float).unsqueeze(-1) for seq in sequences],
                                    batch_first=True, padding_value=-1)
    return padded_sequences, torch.stack(labels), torch.tensor(lengths)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout_rate=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        # dropout_rate 用于 LSTM 内部的 dropout
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # lengths 在使用 pack_padded_sequence 时需要移动到 CPU
        lengths = lengths.cpu()
        # 打包序列以适应 LSTM 输入
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        out, _ = self.lstm(x)  # LSTM 处理步骤
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        batch_size = out.size(0)

        # 提取最后时间步的输出
        last_outputs = out[range(batch_size), lengths - 1, :]

        out = self.fc(last_outputs)  # 最后通过全连接层
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


# 训练模型
def train_model(model, train_loader, test_loader, epochs=10):
    model.to(device)  # 将模型移动到 GPU
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_accuracy_list = []
    test_accuracy_list = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

        for batch_X, batch_y, lengths in train_loader:
            batch_X, batch_y, lengths = batch_X.to(device), batch_y.to(device), lengths.to(device)  # 移动数据到 GPU

            optimizer.zero_grad()
            outputs = model(batch_X, lengths)

            loss = nn.CrossEntropyLoss()(outputs, torch.argmax(batch_y, dim=1).to(outputs.device))
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == torch.argmax(batch_y, dim=1).to(outputs.device)).sum().item()

        train_accuracy = correct / total
        train_accuracy_list.append(train_accuracy)

        # 测试模型
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y, lengths in test_loader:
                batch_X, batch_y, lengths = batch_X.to(device), batch_y.to(device), lengths.to(device)  # 移动数据到 GPU
                outputs = model(batch_X, lengths)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == torch.argmax(batch_y, dim=1).to(outputs.device)).sum().item()

                all_preds.extend(predicted.tolist())
                all_targets.extend(torch.argmax(batch_y, dim=1).tolist())

        print(loss.item())
        test_accuracy = correct / total
        test_accuracy_list.append(test_accuracy)
        print(f'Epoch [{epoch + 1}/{epochs}], Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

    return train_accuracy_list, test_accuracy_list, all_preds, all_targets


start0 = time.time()
# 训练 LSTM 模型
input_size = 1
hidden_size = 32
num_layers = 2
model = LSTMModel(input_size, hidden_size, num_classes, num_layers)

train_accuracy_list, test_accuracy_list, all_preds, all_targets = train_model(model, train_loader, test_loader,
                                                                              epochs=50)
end0 = time.time()
print("Training Time:", end0 - start0)

# 保存模型
torch.save(model.state_dict(), 'parameterlstm0.4_model30_100000.pth')

# 保存准确率数据
accuracy_data = pd.DataFrame({
    'Epoch': range(1, len(train_accuracy_list) + 1),
    'Train Accuracy': train_accuracy_list,
    'Test Accuracy': test_accuracy_list
})
accuracy_data.to_csv('parameterlstm_model0.4_accuracy30_data100000.csv', index=False)

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
class_metrics.to_csv('parameterlstm0.4_model_class30_metrics100000.csv', index=False)

# 生成混淆矩阵
conf_matrix = confusion_matrix(all_targets, all_preds)
name = ["LDPC1", "LDPC2", "LDPC3", "LDPC4",'LDPC5']

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=name, yticklabels=name)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('parameterlstm0.4_model_confusion30_matrix100000.png')
plt.show()