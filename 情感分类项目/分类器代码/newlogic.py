import numpy as np
import pickle
import io
import os
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def load_pickle_safe(path):
    # 去除可能存在的引号
    path = path.strip('"').strip("'")
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件未找到: {path}")
    with open(path, 'rb') as f:
        try:
            return CpuUnpickler(f).load()
        except Exception:
            f.seek(0)
            return pickle.load(f)

def to_numpy(data):
    """将 Tensor 或 list 转换为 numpy 数组"""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], torch.Tensor):
            return np.stack([t.cpu().numpy() for t in data])
        return np.array(data)
    return data

# ==========================================
# 数据加载
# ==========================================
# 设置文件夹路径
base_dir = r"D:\desktop\备份\专业课\模式识别"

path_p = os.path.join(base_dir, "feature_p(3).pkl")
path_n = os.path.join(base_dir, "feature_n(3).pkl")
path_test_x = os.path.join(base_dir, "sentences_feature.pkl")
path_test_y = os.path.join(base_dir, "labels.pkl")

print("正在加载训练数据...")
# 加载正负样本特征
X_p = to_numpy(load_pickle_safe(path_p))
X_n = to_numpy(load_pickle_safe(path_n))

# 创建标签 (1: Positive, 0: Negative)
y_p = np.ones(X_p.shape[0])
y_n = np.zeros(X_n.shape[0])

# 合并为训练集
X_train = np.vstack((X_p, X_n))
y_train = np.concatenate((y_p, y_n))

print("正在加载测试数据...")
X_test = to_numpy(load_pickle_safe(path_test_x))
y_test = to_numpy(load_pickle_safe(path_test_y))

print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")

# ==========================================
# 训练逻辑回归模型
# ==========================================
print("\n开始训练逻辑回归 (Logistic Regression)...")

# 初始化模型
# C=1.0: 正则化强度，越小越强（防止过拟合）
# max_iter=1000: 最大迭代次数，保证模型能收敛
# solver='lbfgs': 求解优化问题的算法，适合中等规模数据
lr_classifier = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)

# 拟合数据
lr_classifier.fit(X_train, y_train)

# ==========================================
# 4. 评估与结果
# ==========================================
print("正在进行预测...")
y_pred = lr_classifier.predict(X_test)

# 计算准确率
acc = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率 (Accuracy): {acc * 100:.2f}%")

# 打印详细分类报告
target_names = ['Negative (0)', 'Positive (1)']
print("\n详细分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))



