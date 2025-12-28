import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import io
import os
from sklearn.metrics import classification_report

class CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def load_pickle_safe(path):
    path = path.strip('"').strip("'")
    if not os.path.exists(path):
        raise FileNotFoundError(f" 找不到文件: {path}")
    with open(path, 'rb') as f:
        try:
            return CpuUnpickler(f).load()
        except Exception:
            f.seek(0)
            return pickle.load(f)

def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data.float().cpu()
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], torch.Tensor):
            return torch.stack([t.cpu() for t in data]).float()
        return torch.FloatTensor(np.array(data))
    return torch.FloatTensor(data)

# ==========================================
# 加载数据
# ==========================================
base_dir = r"D:\desktop\备份\专业课\模式识别"
path_p = os.path.join(base_dir, "feature_p(3).pkl")
path_n = os.path.join(base_dir, "feature_n(3).pkl")
path_test_x = os.path.join(base_dir, "sentences_feature.pkl")
path_test_y = os.path.join(base_dir, "labels.pkl")

print("正在加载训练数据...")
# 加载并拼接数据
data_p = to_tensor(load_pickle_safe(path_p))
data_n = to_tensor(load_pickle_safe(path_n))
train_X = torch.cat((data_p, data_n), dim=0)

train_y = torch.cat((torch.ones(len(data_p), dtype=torch.long),
                     torch.zeros(len(data_n), dtype=torch.long)), dim=0)

print("正在加载测试数据...")
test_X = to_tensor(load_pickle_safe(path_test_x))
test_y = torch.LongTensor(np.array(load_pickle_safe(path_test_y)))

print(f"训练集形状: {train_X.shape}")
print(f"测试集形状: {test_X.shape}")

# DataLoader
batch_size = 64
train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=batch_size, shuffle=False)

# ==========================================
# 定义 MLP 模型
# ==========================================
class SentimentMLP(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_classes=2):
        super(SentimentMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

# ==========================================
# 训练配置
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentimentMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ==========================================
# 训练循环
# ==========================================
def evaluate_acc(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

print(f"\n开始训练 MLP 模型 (设备: {device})...")
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    # 每2轮打印一次
    if (epoch + 1) % 2 == 0:
        test_acc = evaluate_acc(test_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {total_loss / len(train_loader):.4f} | Test Acc: {test_acc:.2f}%")

print("正在进行预测...")

# ==========================================
# 【修改点3】最终报告
# ==========================================
def get_all_preds_and_labels(loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            # 将 Tensor 转为 CPU 的 Numpy 数组
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds

# 获取全量真实标签和预测值
y_true, y_pred = get_all_preds_and_labels(test_loader)

# 计算整体准确率
from sklearn.metrics import accuracy_score
final_acc = accuracy_score(y_true, y_pred)

print(f"\n测试集准确率 (Accuracy): {final_acc*100:.2f}%\n")

# 打印详细分类报告 (完全复刻截图格式)
print("详细分类报告:\n")
# digits=4 确保显示4位小数，target_names 设置显示的标签名
print(classification_report(y_true, y_pred, target_names=['Negative (0)', 'Positive (1)'], digits=4))