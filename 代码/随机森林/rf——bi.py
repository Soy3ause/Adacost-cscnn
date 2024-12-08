import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import numpy as np

# 读取CSV文件
df = pd.read_csv('15name.csv')

# 替换无穷值
df.replace([np.inf, -np.inf], np.nan, inplace=True)


# 转换数据类型并清理 NaN 值
def convert_to_numeric(df):
    numeric_df = df.copy()  # 复制数据框以避免修改原始数据
    for col in numeric_df.select_dtypes(include=['object']).columns:
        try:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        except Exception as e:
            print(f"Error converting column {col}: {e}")
            numeric_df.drop(columns=[col], inplace=True)

    numeric_df.fillna(0, inplace=True)
    return numeric_df


df_numeric = convert_to_numeric(df)

# 最后一列是目标变量（标签），其余列是特征变量
X = df_numeric.iloc[:, :-1]  # 特征变量
y = df_numeric.iloc[:, -1]  # 目标变量

# 初始化随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 初始化KFold交叉验证器
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 交叉验证的评分列表
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# 进行十折交叉验证
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 收集每一折的评分
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
    recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

# 输出每一折的平均评分
print("平均准确率:", np.mean(accuracy_scores))
print("平均精确率:", np.mean(precision_scores))
print("平均f1分数:", np.mean(f1_scores))
print("平均召回率:", np.mean(recall_scores))