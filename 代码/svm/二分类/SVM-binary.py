__author__ = '黄心怡'
__edit_time__ = '2024.12.06'
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight, compute_class_weight
from sklearn.metrics import average_precision_score
from datetime import datetime


# 划分训练集和测试集评价指标
def pjzb_1(y_test, y_pred):
    print("20%测试集结果")
    accuracy = accuracy_score(y_test, y_pred)
    precision_micro = precision_score(y_test, y_pred)
    recall_micro = recall_score(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred)  # ,pos_label='0'

    print("准确率", accuracy)
    print(f"精确率 : {precision_micro}")
    print(f"召回率 : {recall_micro}")
    print(f"F1 Score : {f1_micro}")

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # 计算并打印分类报告，包含准确率、精确率、召回率和F1分数等
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
# K折交叉验证评价指标
def pjzb_k(y, y_pred_k):
    print("K=10折交叉验证结果")
    accuracy = accuracy_score(y, y_pred_k)
    precision_micro = precision_score(y, y_pred_k)
    recall_micro = recall_score(y, y_pred_k)
    f1_micro = f1_score(y, y_pred_k)  # ,pos_label='0'

    print("准确率", accuracy)
    print(f"精确率 : {precision_micro}")
    print(f"召回率 : {recall_micro}")
    print(f"F1 Score : {f1_micro}")

    # 计算混淆矩阵
    # cm = confusion_matrix(y_test, y_pred)
    cm = confusion_matrix(y, y_pred_k)
    print("Confusion Matrix:")
    print(cm)

    # 计算并打印分类报告，包含准确率、精确率、召回率和F1分数等
    # report = classification_report(y_test, y_pred)
    report = classification_report(y, y_pred_k)
    print("Classification Report:")
    print(report)


# 指定CSV文件的路径
file_path = ('D:/python_worksapce/lunwen/CIC_IDS_2018_binary_preprocessed.csv')
data = pd.read_csv(file_path)
# data = pd.read_csv(file_path,header=None)
#filtered_corrected_Binary   kddcup.data_10_percent_corrected_Binary kddcup.data.corrected_Binary[41]
#Chrome_all  dns2tcp_all   dnscat2_all  Firefox_all  iodine_all[34]  CIC-Darknet2020[83]
#CIC_IDS_2017_binary CIC_IDS_2018_binary_preprocessed

print(data.head())  # 显示前几行数据（默认前五行），检查是否读取成功
print(data.info())  # 输出有关DataFrame的简要摘要，包括：数据的类型（dtype） 内存使用情况 非空值的数量

# 取样测试（数据集太大可以先做取样测试看有没有报错）
# 2017取样测试
"""
# 假设'label'列包含标签
labels = data[' Label'].unique()  # 获取所有唯一的标签
# 初始化一个空的列表来存储抽样后的数据
sampled_data = []
# 对每个标签进行抽样
for label in labels:
    # 获取该标签的所有数据
    label_data = data[data[' Label'] == label]
    # 检查数据量是否足够
    if len(label_data) >= 100:
        # 如果数据量足够，随机抽取 1000 个样本
        sample = label_data.sample(n=100, replace=False)
    else:
        # 如果数据量不足，抽取所有可用的数据
        sample = label_data
    # 将抽样后的数据添加到列表中
    sampled_data.append(sample)
# 将列表中的DataFrame合并为一个新的DataFrame
data = pd.concat(sampled_data, ignore_index=True)
print(data.head())
print(data.info())
"""
# 2018取样测试
"""
# 假设'label'列包含标签
labels = data['Labe1'].unique()  # 获取所有唯一的标签
# 初始化一个空的列表来存储抽样后的数据
sampled_data = []
# 对每个标签进行抽样
for label in labels:
    # 获取该标签的所有数据
    label_data = data[data['Labe1'] == label]
    # 检查数据量是否足够
    if len(label_data) >= 100:
        # 如果数据量足够，随机抽取 1000 个样本
        sample = label_data.sample(n=100, replace=False)
    else:
        # 如果数据量不足，抽取所有可用的数据
        sample = label_data
    # 将抽样后的数据添加到列表中
    sampled_data.append(sample)
# 将列表中的DataFrame合并为一个新的DataFrame
data = pd.concat(sampled_data, ignore_index=True)
print(data.head())
print(data.info())
"""



# 数据处理
# Chrome_all  dns2tcp_all   dnscat2_all  Firefox_all  iodine_all
"""
#ip
data['SourceIP'] = pd.to_numeric(data['SourceIP'],errors='coerce')
data['DestinationIP'] = pd.to_numeric(data['DestinationIP'],errors='coerce')
#时间戳
format_string = '%Y-%m-%d %H:%M:%S'
# 使用 pd.to_datetime 解析时间戳，并指定格式
data['TimeStamp'] = pd.to_datetime(data['TimeStamp'], format=format_string)
data['TimeStamp'] = data['TimeStamp'].apply(lambda x: x.timestamp())
#标签
data['DoH'] = data['DoH'].astype(str)
value_mapping = {
    'True':'0',
    'False':'1',
}
data['DoH'] = data['DoH'].replace(value_mapping)
data.replace([np.inf, -np.inf, np.nan], 0.0, inplace=True)
"""


X = data.iloc[:, 0:52]  # 特征(不包含52列）
y = data.iloc[:, 52]   # 标签

# 缺失值替换为0
#X = X.fillna(0)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化(对特征做处理）
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
X = ss.fit_transform(X)


# 模型训练及预测
model = SVC()
# model = SVC(C=1.0, class_weight=class_weights)

# 1
""""""
model.fit(X_train,y_train)
y_pred = model.predict(X_test)



# 设置k折交叉验证
"""
k = 10
scores = cross_val_score(model, X, y, cv=k)
# 输出k折交叉验证的准确率
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# 执行k折交叉验证并获取预测结果
y_pred_k = cross_val_predict(model, X, y, cv=k)
"""


print("模型是SVM")
# 模型评价
print(file_path)
# acc准确率，pr精确率，rc召回率，f1
pjzb_1(y_test, y_pred)  # 划分训练集和测试集评价指标
# pjzb_k(y,y_pred_k) # K折交叉验证评价指标






