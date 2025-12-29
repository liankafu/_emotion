import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')


def load_data(path_p, path_n):
    """加载正负样本数据"""
    with open(path_p, 'rb') as f:
        features_p = pickle.load(f)
    with open(path_n, 'rb') as f:
        features_n = pickle.load(f)

    features_p = np.array(features_p)
    features_n = np.array(features_n)

    # 创建标签：正样本为1，负样本为0
    labels_p = np.ones(features_p.shape[0])
    labels_n = np.zeros(features_n.shape[0])

    # 合并数据和标签
    X = np.vstack((features_p, features_n))
    y = np.concatenate((labels_p, labels_n))

    return X, y


def evaluate_classifier(clf, X_train, X_test, y_train, y_test, classifier_name):
    """评估分类器性能"""
    print(f"\n{'=' * 50}")
    print(f"{classifier_name} 分类器性能评估")
    print(f"{'=' * 50}")

    # 训练模型
    clf.fit(X_train, y_train)

    # 预测
    y_pred = clf.predict(X_test)

    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['负样本', '正样本']))

    return clf, accuracy


def cross_validation_evaluation(clf, X, y, classifier_name, cv=5):
    """执行交叉验证评估"""
    print(f"\n{'=' * 50}")
    print(f"{classifier_name} 交叉验证评估 (k={cv})")
    print(f"{'=' * 50}")

    # 使用分层K折交叉验证（保持类别比例）
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # 计算交叉验证得分
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')

    print(f"交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"各折准确率: {[f'{score:.4f}' for score in cv_scores]}")

    return cv_scores.mean(), cv_scores.std()


def hyperparameter_tuning_svm(X_train, y_train):
    """SVM超参数调优"""
    print("\n" + "=" * 50)
    print("SVM超参数调优...")
    print("=" * 50)

    # 定义参数网格
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
        'degree': [2, 3, 4]  # 多项式核的阶数
    }

    # 创建基础模型
    svm = SVC(probability=True, random_state=42)

    # 使用网格搜索进行超参数调优
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=5,  # 5折交叉验证
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # 执行网格搜索
    grid_search.fit(X_train, y_train)

    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def main():
    # 加载数据
    print("加载数据...")

    # 加载训练数据
    path_n = "D:/jupyter/feature_n.pkl"
    path_p = "D:/jupyter/feature_p.pkl"
    X_train_full, y_train_full = load_data(path_p, path_n)

    # 加载测试数据
    with open('D:/jupyter/sentences_feature.pkl', 'rb') as f:
        sentences_feature = pickle.load(f)
    with open('D:/jupyter/labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    X_test = np.array(sentences_feature)
    y_test = np.array(labels)

    print(f"完整训练集形状: {X_train_full.shape}, 测试集形状: {X_test.shape}")
    print(f"完整训练集正负样本比例: {np.sum(y_train_full == 1)}:{np.sum(y_train_full == 0)}")
    print(f"测试集正负样本比例: {np.sum(y_test == 1)}:{np.sum(y_test == 0)}")

    # 将完整训练数据划分为训练集和验证集（用于超参数调优）
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    print(f"\n超参数调优训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}")

    # 数据标准化（对于SVM很重要）
    print("\n数据标准化处理...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_train_full_scaled = scaler.fit_transform(X_train_full)

    # 第一阶段：超参数调优
    print("\n" + "=" * 60)
    print("第一阶段：SVM超参数调优")
    print("=" * 60)

    # SVM分类器超参数调优
    best_svm = hyperparameter_tuning_svm(X_train_scaled, y_train)

    # 第二阶段：在验证集上评估调优后的模型
    print("\n" + "=" * 60)
    print("第二阶段：验证集性能评估")
    print("=" * 60)

    # 使用调优后的SVM模型在验证集上评估
    y_pred_val = best_svm.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_pred_val)

    print(f"SVM 验证集准确率: {val_accuracy:.4f}")

    # 第三阶段：交叉验证评估
    print("\n" + "=" * 60)
    print("第三阶段：交叉验证评估")
    print("=" * 60)

    # 在完整训练集上进行交叉验证
    cv_mean, cv_std = cross_validation_evaluation(best_svm, X_train_full_scaled, y_train_full, "SVM")

    # 第四阶段：在测试集上最终评估
    print("\n" + "=" * 60)
    print("第四阶段：测试集最终评估")
    print("=" * 60)

    # 使用完整训练集重新训练最佳模型
    print("使用完整训练集重新训练最佳SVM模型...")
    best_svm.fit(X_train_full_scaled, y_train_full)

    # 最终测试
    y_pred_final = best_svm.predict(X_test_scaled)
    final_accuracy = accuracy_score(y_test, y_pred_final)

    print(f"\nSVM模型在测试集上的性能:")
    print(f"测试集准确率: {final_accuracy:.4f}")
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred_final, target_names=['负样本', '正样本']))

    # 性能总结
    print("\n" + "=" * 60)
    print("模型性能总结")
    print("=" * 60)
    print(f"{'模型':<15} {'验证集准确率':<15} {'CV平均准确率':<15} {'CV标准差':<15} {'测试集准确率':<15}")
    print("-" * 75)
    print(f"{'SVM':<15} {val_accuracy:<15.4f} {cv_mean:<15.4f} {cv_std:<15.4f} {final_accuracy:<15.4f}")

    # 可选：保存最佳模型
    print("\n" + "=" * 50)
    save_model = input("是否保存最佳SVM模型？(y/n): ")
    if save_model.lower() == 'y':
        with open('best_svm_model_tuned.pkl', 'wb') as f:
            pickle.dump({
                'model': best_svm,
                'scaler': scaler,
                'validation_accuracy': val_accuracy,
                'cross_val_mean': cv_mean,
                'cross_val_std': cv_std,
                'test_accuracy': final_accuracy,
                'best_params': best_svm.get_params()
            }, f)
        print("调优后的SVM模型已保存为 'best_svm_model_tuned.pkl'")


if __name__ == "__main__":
    main()