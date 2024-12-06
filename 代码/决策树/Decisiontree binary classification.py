__author__ = '曹申'
__edit_time__ = '2024.12.06'
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

def process_file(file_path):
    """
    用决策树进行二分类，并评估模型性能。
    """
    # 读取数据
    data = pd.read_csv(file_path)

    # 提取标签和特征
    y = data['Labe1']  # 标签列
    X = data.drop(columns=['Labe1'])  # 特征列

    # 仅保留数值型特征
    X = X.select_dtypes(include=[np.number])

    # 标签二值化
    y = y.apply(lambda x: 1 if x == 1 else 0)

    # 初始化决策树模型
    model = DecisionTreeClassifier(random_state=42)

    # 使用 StratifiedKFold 保持标签分布
    skf = StratifiedKFold(n_splits=10)

    # 初始化指标存储
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    # 交叉验证
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 模型训练
        model.fit(X_train, y_train)

        # 模型预测
        y_pred = model.predict(X_test)

        # 计算指标
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=1))
        recalls.append(recall_score(y_test, y_pred, zero_division=1))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=1))

    # 打印结果
    print("Results for Binary Classification:")
    print(f"Accuracy: {np.mean(accuracies):.8f}")
    print(f"Precision: {np.mean(precisions):.8f}")
    print(f"Recall: {np.mean(recalls):.8f}")
    print(f"F1 Score: {np.mean(f1_scores):.8f}\n")


# 示例文件路径
file_paths = ['C:/Users/CaoShen/Downloads/CIC_IDS_2018_binary_preprocessed.csv']

# 处理文件
for file_path in file_paths:
    print(f"Processing {file_path}...\n")
    process_file(file_path)
