import numpy as np
import pickle
import io
import os
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

class CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def load_pickle_safe(path):
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
# 数据加载与预处理
# ==========================================

# 训练集文件路径
path_p = r"D:\desktop\备份\专业课\模式识别\feature_p(3).pkl"
path_n = r"D:\desktop\备份\专业课\模式识别\feature_n(3).pkl"

# 测试集文件路径
path_test_x = r"D:\desktop\备份\专业课\模式识别\sentences_feature.pkl"
path_test_y = r"D:\desktop\备份\专业课\模式识别\labels.pkl"
# -----------------------------------------------

print("正在加载训练数据...")
# 加载正负样本
features_p = load_pickle_safe(path_p)
features_n = load_pickle_safe(path_n)

# 转为 Numpy
X_p = to_numpy(features_p)
X_n = to_numpy(features_n)

# 创建标签 (1: Positive, 0: Negative)
y_p = np.ones(X_p.shape[0])
y_n = np.zeros(X_n.shape[0])

# 合并训练集
X_train = np.vstack((X_p, X_n))
y_train = np.concatenate((y_p, y_n))

print("正在加载测试数据...")
# 加载测试集
X_test_raw = load_pickle_safe(path_test_x)
y_test_raw = load_pickle_safe(path_test_y)

# 转为 Numpy
X_test = to_numpy(X_test_raw)
y_test = to_numpy(y_test_raw)

print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")

# ==========================================
# 训练决策树模型
# ==========================================
print("\n开始训练决策树 (Decision Tree)...")

# 初始化决策树分类器
dt_classifier = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)

# 拟合数据
dt_classifier.fit(X_train, y_train)

# ==========================================
# 4. 评估与结果
# ==========================================
print("正在进行预测...")
y_pred = dt_classifier.predict(X_test)

# 计算准确率
acc = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率 (Accuracy): {acc * 100:.2f}%")

# 打印详细分类报告
print("\n详细分类报告:")
target_names = ['Negative (0)', 'Positive (1)']
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))


# 寻找最佳参数
param_grid = {
    'max_depth': [10, 15, 20, 30, None],       # 限制深度，防止过拟合
    'min_samples_leaf': [1, 5, 10, 20],        # 叶子节点最小样本数
    'max_features': ['sqrt', 'log2', None],    # 针对256维特征的筛选方案
    'criterion': ['gini', 'entropy']           # 纯度衡量标准
}

print("\n正在启动网格搜索 (GridSearchCV)... ")

# 2. 初始化网格搜索对象
# cv=5 表示五折交叉验证，n_jobs=-1 表示使用所有CPU核心加速
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# 3. 在训练集上进行搜索
grid_search.fit(X_train, y_train)

# 4. 输出结果
print("\n--- 搜索完成 ---")
print(f"最佳参数组合: {grid_search.best_params_}")
print(f"交叉验证最高准确率: {grid_search.best_score_:.4f}")

# 5. 使用最佳模型在测试集上验证
best_dt = grid_search.best_estimator_
y_pred_best = best_dt.predict(X_test)

print(f"\n最佳模型测试集准确率: {accuracy_score(y_test, y_pred_best) * 100:.2f}%")
print("\n最佳模型详细报告:")
print(classification_report(y_test, y_pred_best, target_names=['Negative (0)', 'Positive (1)'], digits=4))