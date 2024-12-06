from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np


def process_file(file_path):
    print(f"Processing file: {file_path}")

    # 读取数据
    data = pd.read_csv(file_path)

    # 提取标签和特征
    y = data['Labe1']
    X = data.drop(columns=['Labe1'])

    # 仅保留数值特征
    X = X.select_dtypes(include=[np.number])

    # 转为稀疏矩阵，节省内存
    X = csr_matrix(X)

    # 检查标签分布
    print("Unique labels in y before processing:", y.unique())
    print("y data type before processing:", y.dtype)

    # 多分类：将连续值离散化为整数类别
    y = pd.cut(y, bins=5, labels=False)  # 将 y 分为 5 个区间，并生成离散类别

    # 确保标签是整数
    y = y.astype(int)

    # 检查标签分布
    print("Unique labels in y after processing:", y.unique())
    print("y data type after processing:", y.dtype)

    # 初始化决策树模型
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42
    )

    # 使用 StratifiedKFold 进行交叉验证
    skf = StratifiedKFold(n_splits=10)

    # 定义指标存储容器
    metrics = {
        "accuracy": [], "precision_macro": [], "recall_macro": [], "f1_macro": [],
        "precision_micro": [], "recall_micro": [], "f1_micro": [],
        "precision_weighted": [], "recall_weighted": [], "f1_weighted": []
    }

    # 交叉验证
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 模型训练
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 计算各类指标
        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision_macro"].append(precision_score(y_test, y_pred, average="macro", zero_division=1))
        metrics["recall_macro"].append(recall_score(y_test, y_pred, average="macro", zero_division=1))
        metrics["f1_macro"].append(f1_score(y_test, y_pred, average="macro", zero_division=1))

        metrics["precision_micro"].append(precision_score(y_test, y_pred, average="micro", zero_division=1))
        metrics["recall_micro"].append(recall_score(y_test, y_pred, average="micro", zero_division=1))
        metrics["f1_micro"].append(f1_score(y_test, y_pred, average="micro", zero_division=1))

        metrics["precision_weighted"].append(precision_score(y_test, y_pred, average="weighted", zero_division=1))
        metrics["recall_weighted"].append(recall_score(y_test, y_pred, average="weighted", zero_division=1))
        metrics["f1_weighted"].append(f1_score(y_test, y_pred, average="weighted", zero_division=1))

    # 输出指标平均值
    print("\nResults for Multi Classification:")
    for metric, values in metrics.items():
        print(f"{metric.capitalize()}: {np.mean(values):.8f}")


# 示例文件路径
file_paths = ['C:/Users/CaoShen/Downloads/CIC_IDS_2018_multi_preprocessed.csv']

# 处理文件
for file_path in file_paths:
    process_file(file_path)
