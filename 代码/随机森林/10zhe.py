__author__ = '田宇航'
__edit_time__ = '2024.12.06'
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve
)
import numpy as np
from sklearn.preprocessing import label_binarize

# 读取CSV文件
df = pd.read_csv('CIC_IDS_2018_multi_preprocessed.csv')
df = df.sample(frac=0.1)  # 随机抽样10%的数据

# 替换无穷值
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 定义一个函数来转换数据类型
def convert_to_numeric(df):
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"Error converting column {col}: {e}")
            # 可以选择删除或保留这些列
            # df.drop(columns=[col], inplace=True)  # 如果选择删除
    return df

# 转换数据类型并清理 NaN 值
df_numeric = convert_to_numeric(df)
df_numeric.dropna(subset=df_numeric.columns[:-1], inplace=True)  # 删除特征列中的 NaN 值
df_numeric = df_numeric[~df_numeric.iloc[:, -1].isna()]  # 删除目标变量列中的 NaN 值（假设目标变量是最后一列）

# 重新分配 X 和 y
X = df_numeric.iloc[:, :-1]
y = df_numeric.iloc[:, -1]

# 输出转换后的数据
print(df_numeric.head())

# 将标签二值化
y_bin = label_binarize(y, classes=np.unique(y))
n_classes = y_bin.shape[1]

# 初始化随机森林分类器
clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))

# 使用KFold进行十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 存储每折的预测结果
y_pred_all = []
y_true_all = []

# 存储每类的平均精确率（AP）
ap_scores_all = np.zeros(n_classes)

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)
    
    y_pred_all.extend(y_pred)
    y_true_all.extend(y_test)
    
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    for i in range(n_classes):
        precision, recall, thresholds = precision_recall_curve(y_test_bin[:, i], y_pred_prob[:, i])
        ap_scores_all[i] += average_precision_score(y_test_bin[:, i], y_pred_prob[:, i])

# 计算平均精确率均值（mAP）
map_score = np.mean(ap_scores_all / 10)

# 计算其他评估指标
acc_score = accuracy_score(y_true_all, y_pred_all)
pr_macro = precision_score(y_true_all, y_pred_all, average='macro')
pr_micro = precision_score(y_true_all, y_pred_all, average='micro')
pr_weighted = precision_score(y_true_all, y_pred_all, average='weighted')
rc_macro = recall_score(y_true_all, y_pred_all, average='macro')
rc_micro = recall_score(y_true_all, y_pred_all, average='micro')
rc_weighted = recall_score(y_true_all, y_pred_all, average='weighted')
f1_macro = f1_score(y_true_all, y_pred_all, average='macro')
f1_micro = f1_score(y_true_all, y_pred_all, average='micro')
f1_weighted = f1_score(y_true_all, y_pred_all, average='weighted')

# 输出结果
print(f'平均精确率均值（mAP）: {map_score:.4f}')
print(f'准确率（ACC）: {acc_score:.4f}')
print(f'精确率（PR）宏平均: {pr_macro:.4f}')
print(f'精确率（PR）微平均: {pr_micro:.4f}')
print(f'精确率（PR）加权平均: {pr_weighted:.4f}')
print(f'召回率（RC）宏平均: {rc_macro:.4f}')
print(f'召回率（RC）微平均: {rc_micro:.4f}')
print(f'召回率（RC）加权平均: {rc_weighted:.4f}')
print(f'F1分数 宏平均: {f1_macro:.4f}')
print(f'F1分数 微平均: {f1_micro:.4f}')
print(f'F1分数 加权平均: {f1_weighted:.4f}')