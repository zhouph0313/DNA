import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time

# ----------------- 配置 -----------------
# 文件路径，每种类型的三个文件
file_indices = {
    0: ['C:/Users/zhoupenghua/Desktop/convedna.txt',
        'C:/Users/zhoupenghua/Desktop/convedna2.txt',
        'C:/Users/zhoupenghua/Desktop/convedna3.txt'],

    1: ['C:/Users/zhoupenghua/Desktop/ldpcdna.txt',
        'C:/Users/zhoupenghua/Desktop/ldpcdna2.txt',
        'C:/Users/zhoupenghua/Desktop/ldpcdna3.txt'],

    2: ['C:/Users/zhoupenghua/Desktop/polardna.txt',
        'C:/Users/zhoupenghua/Desktop/polardna2.txt',
        'C:/Users/zhoupenghua/Desktop/polardna3.txt'],

    3: ['C:/Users/zhoupenghua/Desktop/bchdna.txt',
        'C:/Users/zhoupenghua/Desktop/bchdna2.txt',
        'C:/Users/zhoupenghua/Desktop/bchdna3.txt']
}

num_classes = 4  # 分类数（类型）
min_length = 100  # 最小截取长度
max_length = 400  # 最大截取长度
error_rate = 0.15  # 注入错误率

max_samples_per_file = 100000  # 每个文件最大读取数量


# ----------------- 数据预处理及数据集 -----------------
def introduce_errors(sequence, error_rate):
    # 对序列中每个数字（代表碱基）以一定概率替换成 [0,1,2,3] 中的随机值
    for i in range(len(sequence)):
        if random.random() < error_rate:
            possible_replacements = [x for x in range(num_classes)]
            sequence[i] = random.choice(possible_replacements)
    return sequence


# 读取所有文件构造数据集（列表形式）
sequences = []
labels = []
lengths = []

# 用于记录每个类别的读取计数
total_read_counts = {label: 0 for label in range(num_classes)}

for label, files in file_indices.items():
    for file in files:
        with open(file, 'r') as f:
            content = f.read().strip()
            sequence_length = len(content)
            current_samples_count = 0

            while current_samples_count < max_samples_per_file:
                if sequence_length < min_length:
                    print(f"Warning: Sequence length {sequence_length} is shorter than min_length {min_length}.")
                    break

                start_index = np.random.randint(0, sequence_length - max_length + 1)
                sub_length = np.random.randint(min_length, max_length + 1)
                end_index = start_index + sub_length

                if end_index <= sequence_length:
                    # 文件中存储的为字符形式，转换为 int 列表
                    encoded_sequence = [int(x) for x in content[start_index:end_index]]
                    # 注入错误
                    encoded_sequence = introduce_errors(encoded_sequence, error_rate)

                    sequences.append(encoded_sequence)
                    one_hot_label = [0] * num_classes
                    one_hot_label[label] = 1
                    labels.append(one_hot_label)
                    lengths.append(len(encoded_sequence))
                    current_samples_count += 1
                    total_read_counts[label] += 1

for label in range(num_classes):
    print(f'Total sequences read for class {label}: {total_read_counts[label]}')

# 划分训练集与测试集
X_train, X_test, y_train, y_test, lengths_train, lengths_test = train_test_split(
    sequences, labels, lengths, test_size=0.1, random_state=42
)
print("训练集样本数：", len(y_train))


# 先定义一个辅助函数，将数字序列转换为 one-hot 编码
def seq_to_onehot(seq):
    """
    seq: list of ints (each in {0,1,2,3})
    返回: Tensor of shape (len(seq), 4)
    """
    seq_tensor = torch.tensor(seq, dtype=torch.long)
    onehot = F.one_hot(seq_tensor, num_classes=4).float()
    return onehot


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, lengths):
        self.sequences = sequences
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 每个序列先转换为 one-hot 编码
        onehot_seq = seq_to_onehot(self.sequences[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        length = self.lengths[idx]
        return onehot_seq, label, length


def collate_fn(batch):
    # batch中每个元素：(onehot_seq, label, length)
    sequences, labels, lengths = zip(*batch)
    # 按序列长度进行 padding，注意此处每个序列已经是 (seq_len, 4) 的 tensor，padding_value=0（对应 [0,0,0,0]）
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return padded_sequences, torch.stack(labels), torch.tensor(lengths)


train_dataset = SequenceDataset(X_train, y_train, lengths_train)
test_dataset = SequenceDataset(X_test, y_test, lengths_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


# ----------------- Transformer 模型 -----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # 若 d_model 为奇数，保证维度匹配
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, num_tokens=4, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=2, num_classes=4,
                 dropout=0.1):
        """
        num_tokens: 输入向量的最后一维大小，DNA序列 one-hot 表示为4维
        embed_dim: 投影到 embedding 空间的维度
        """
        super(TransformerClassifier, self).__init__()
        # 由于输入已经是 one-hot (4维)，用 linear 层映射到 embed_dim
        self.input_linear = nn.Linear(num_tokens, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_length + 10)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, lengths):
        """
        x: (batch, seq_len, 4)   one-hot 编码的序列
        lengths: 每个序列的有效长度（不含 pad）
        """
        # 映射至 embedding空间 -> (batch, seq_len, embed_dim)
        x = self.input_linear(x)
        # 加入位置编码
        x = self.pos_encoder(x)
        # Transformer 接受 (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)  # (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)  # (batch, seq_len, embed_dim)

        # 对每个样本利用有效 token 做平均池化
        pooled = []
        for i, l in enumerate(lengths):
            # 防止 l==0的异常
            if l > 0:
                valid_out = x[i, :l, :]
                pooled.append(valid_out.mean(dim=0))
            else:
                pooled.append(torch.zeros(x.size(2), device=x.device))
        pooled = torch.stack(pooled, dim=0)  # (batch, embed_dim)
        pooled = self.dropout(pooled)
        out = self.fc(pooled)
        return out

    # ----------------- 训练及评估 -----------------


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


def train_model(model, train_loader, test_loader, epochs=10):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_accuracy_list = []
    test_accuracy_list = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

        for batch_X, batch_y, lengths in train_loader:
            batch_X, batch_y, lengths = batch_X.to(device), batch_y.to(device), lengths.to(device)
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

        # 测试阶段
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y, lengths in test_loader:
                batch_X, batch_y, lengths = batch_X.to(device), batch_y.to(device), lengths.to(device)
                outputs = model(batch_X, lengths)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == torch.argmax(batch_y, dim=1).to(outputs.device)).sum().item()
                all_preds.extend(predicted.tolist())
                all_targets.extend(torch.argmax(batch_y, dim=1).tolist())

        test_accuracy = correct / total
        test_accuracy_list.append(test_accuracy)
        print(f'Epoch [{epoch + 1}/{epochs}], Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # 每 5 个 epoch 保存一次模型及指标
        if (epoch + 1) % 5 == 0:
            model_save_path = f'ztype_transformer_{error_rate}_epoch{epoch + 1}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')

            accuracy_data = pd.DataFrame({
                'Epoch': range(1, len(train_accuracy_list) + 1),
                'Train Accuracy': train_accuracy_list,
                'Test Accuracy': test_accuracy_list
            })
            accuracy_data.to_csv(f'ztype_transformer_{error_rate}_accuracy_epoch{epoch + 1}.csv', index=False)

            precision = precision_score(all_targets, all_preds, average=None, labels=list(range(num_classes)))
            recall = recall_score(all_targets, all_preds, average=None, labels=list(range(num_classes)))
            f1 = f1_score(all_targets, all_preds, average=None, labels=list(range(num_classes)))

            class_metrics = pd.DataFrame({
                'Class': [f'Class {i}' for i in range(num_classes)],
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })
            class_metrics.to_csv(f'ztype_transformer_{error_rate}_class_metrics_epoch{epoch + 1}.csv', index=False)

            conf_matrix = confusion_matrix(all_targets, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=[f'Class {i}' for i in range(num_classes)],
                        yticklabels=[f'Class {i}' for i in range(num_classes)])
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title(f'Confusion Matrix for Epoch {epoch + 1}')
            plt.savefig(f'ztype_transformer_{error_rate}_confusion_matrix_epoch{epoch + 1}.png')
            plt.close()

    return train_accuracy_list, test_accuracy_list, all_preds, all_targets


# ----------------- 模型训练 -----------------
embed_dim = 64
num_heads = 4
hidden_dim = 128
num_layers = 2

model = TransformerClassifier(num_tokens=4, embed_dim=embed_dim, num_heads=num_heads,
                              hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes, dropout=0.1)

start_time = time.time()
train_accuracy_list, test_accuracy_list, all_preds, all_targets = train_model(model, train_loader, test_loader,
                                                                              epochs=30)
end_time = time.time()
print("Training Time:", end_time - start_time)

# 保存最终模型
torch.save(model.state_dict(), f'ztype_transformer_{error_rate}_final_model.pth')

accuracy_data = pd.DataFrame({
    'Epoch': range(1, len(train_accuracy_list) + 1),
    'Train Accuracy': train_accuracy_list,
    'Test Accuracy': test_accuracy_list
})
accuracy_data.to_csv(f'ztype_transformer_{error_rate}_final_accuracy_data.csv', index=False)

precision = precision_score(all_targets, all_preds, average=None, labels=list(range(num_classes)))
recall = recall_score(all_targets, all_preds, average=None, labels=list(range(num_classes)))
f1 = f1_score(all_targets, all_preds, average=None, labels=list(range(num_classes)))

class_metrics = pd.DataFrame({
    'Class': [f'Class {i}' for i in range(num_classes)],
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
})
class_metrics.to_csv(f'ztype_transformer_{error_rate}_final_class_metrics.csv', index=False)

conf_matrix = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f'Class {i}' for i in range(num_classes)],
            yticklabels=[f'Class {i}' for i in range(num_classes)])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Final Confusion Matrix')
plt.savefig(f'ztype_transformer_{error_rate}_final_confusion_matrix.png')
plt.show()