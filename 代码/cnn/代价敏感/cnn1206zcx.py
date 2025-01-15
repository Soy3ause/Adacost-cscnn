__author__ = '张楚昕'
__edit_time__ = '2024.12.06'
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import average_precision_score,classification_report
import csv
from sklearn.model_selection import KFold
'''
含代价敏感，十折交叉验证，确实值补零
'''
def FOLD2(y_true, y_pred,fold_no):
    print("1:")
    accuracy1[fold_no] = accuracy_score(y_true, y_pred)
    print("准确率", accuracy1[fold_no])
    precision1[fold_no] = precision_score(y_true, y_pred)
    print("精确率", precision1[fold_no])
    f11[fold_no] = f1_score(y_true, y_pred)
    print("f1", f11[fold_no])
    recall1[fold_no] = recall_score(y_true, y_pred)
    print("召回率", recall1[fold_no])
    print("\n")
    print("0:")
    accuracy2[fold_no] = accuracy_score(y_true, y_pred)
    print("准确率", accuracy2[fold_no])
    precision2[fold_no] = precision_score(y_true, y_pred, pos_label=0)
    print("精确率", precision2[fold_no])
    f12[fold_no] = f1_score(y_true, y_pred, pos_label=0)
    print("f1", f12[fold_no])
    recall2[fold_no] = recall_score(y_true, y_pred, pos_label=0)
    print("召回率", recall2[fold_no])

def FOLDduo(y_true, y_pred,fold_no):
    mAP[fold_no] =average_precision_score(y_true, y_pred)
    accuracy1[fold_no] = accuracy_score(y_true, y_pred)
    precision_macro[fold_no] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_micro[fold_no] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    precision_weighted[fold_no] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro[fold_no] = f1_score(y_true, y_pred, average='macro')
    f1_micro[fold_no] = f1_score(y_true, y_pred, average='micro')
    f1_weighted[fold_no] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_macro[fold_no] = recall_score(y_true, y_pred, average='macro')
    recall_micro[fold_no] = recall_score(y_true, y_pred, average='micro')
    recall_weighted[fold_no] = recall_score(y_true, y_pred, average='weighted')

    # 打印每个折的结果（可选）
    print(f"折 {fold_no} 的 mAP", mAP[fold_no])
    print(f"折 {fold_no} 的 准确率", accuracy1[fold_no])
    print(f"折 {fold_no} 的 精确率宏平均", precision_macro[fold_no])
    print(f"折 {fold_no} 的 精确率微平均", precision_micro[fold_no])
    print(f"折 {fold_no} 的 精确率加权平均", precision_weighted[fold_no])
    print(f"折 {fold_no} 的 f1宏平均", f1_macro[fold_no])
    print(f"折 {fold_no} 的 f1微平均", f1_micro[fold_no])
    print(f"折 {fold_no} 的 f1加权平均", f1_weighted[fold_no])
    print(f"折 {fold_no} 的 召回率宏平均", recall_macro[fold_no])
    print(f"折 {fold_no} 的 召回率微平均", recall_micro[fold_no])
    print(f"折 {fold_no} 的 召回率加权平均", recall_weighted[fold_no])


#####deep CNN-binary-CS
from tensorflow.keras import backend as K

def custom_weighted_binary_crossentropy(y_true, y_pred, fn_weight=1.0, tp_weight=1.0, fp_weight=1.0, tn_weight=1.0):
    # 确保y_pred在epsilon和1-epsilon之间，以避免log(0)或log(1)的情况
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)

    # 计算标准的二元交叉熵部分
    bce = - (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))

    fn_loss = fn_weight * (1 - y_pred) ** 2 * y_true  # 当y_true=1且y_pred较小时，损失较大
    tp_loss = tp_weight * y_pred ** 2 * y_true  # 当y_true=1且y_pred较大时，损失较小（但这里我们仍然保留y_true以确保只在y_true=1时计算）
    fp_loss = fp_weight * y_pred ** 2 * (1 - y_true)  # 当y_true=0且y_pred较大时，损失较大
    tn_loss = tn_weight * (1 - y_pred) ** 2 * (1 - y_true)  # 当y_true=0且y_pred较小时，损失较小（但同样地，我们保留1-y_true）

    weighted_loss = bce + fn_loss + fp_loss  # 我们在这里忽略了tp_loss和tn_loss，因为它们通常会在正确分类时很小

    return K.mean(weighted_loss)

file_path=("E:/develop/Adacost-cscnn/代码/cnn/代价敏感/vpnall.csv")
df = pd.read_csv(file_path, low_memory=False)
df=df.fillna(0)#缺失值补零

with open(file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)
    num_columns = len(headers)
print(f"CSV 文件中的列数: {num_columns}")
funumber=num_columns-1#funumber是特征数

X = df.iloc[:, :-1].values.reshape(-1,funumber, 1)#2017
y = df.iloc[:, -1].values
# 对y进行独热编码
unique_categories = np.unique(y)
y_onehot = np.zeros((len(y), len(unique_categories)))
for idx, category in enumerate(y):
    y_onehot[idx, np.where(unique_categories == category)[0][0]] = 1
num_classes = y_onehot.shape[1]
# num_class是类别个数
# X = X.astype(np.float32)#这个是对数据整形，有时会需要
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, kernel_size=4, strides=3, activation='relu', input_shape=(funumber, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(32, kernel_size=5, strides=1, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
])

model.compile(loss=lambda y_true, y_pred: custom_weighted_binary_crossentropy(y_true, y_pred,
                                                                              fn_weight=1, fp_weight=900), optimizer='adagrad', metrics=['accuracy'])

print(num_classes)
if num_classes <=2:
    accuracy1, precision1, f11, recall1 = np.zeros(11), np.zeros(11), np.zeros(11), np.zeros(11)
    accuracy2, precision2, f12, recall2 = np.zeros(11), np.zeros(11), np.zeros(11), np.zeros(11)
else:
    mAP=np.zeros(11)
    accuracy1 = np.zeros(11)
    precision_macro, precision_micro, precision_weighted,  = np.zeros(11), np.zeros(11), np.zeros(11)
    f1_macro,f1_micro,f1_weighted = np.zeros(11), np.zeros(11), np.zeros(11)
    recall_macro,recall_micro, recall_weighted = np.zeros(11), np.zeros(11), np.zeros(11)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_no = 1
for train_index, val_index in kf.split(X_train):
    print(f'Training for fold {fold_no} ...')
    X_tr, X_val = X_train[train_index], X_train[val_index]
    y_tr, y_val = y_train[train_index], y_train[val_index]
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')
    y_pred_proba = model.predict(X_test)  # 返回概率分布
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    y_pred_classes_2d = y_pred_classes.reshape(-1, 1)
    y_test_classes_2d = y_test_classes.reshape(-1, 1)
    y_true = y_test_classes_2d
    y_pred = y_pred_classes_2d
    if num_classes ==2:
        FOLD2(y_true=y_test_classes_2d, y_pred=y_pred_classes_2d,fold_no=fold_no)
    else:
        FOLDduo(y_true=y_test_classes_2d, y_pred=y_pred_classes_2d,fold_no=fold_no)
    fold_no += 1

# 输出十折的平均值
if num_classes ==2:
    # 其中二分类对两类的分类情况都进行了计算
    mean_accuracy1 = np.mean(accuracy1[1:11])
    mean_precision1 = np.mean(precision1[1:11])
    mean_f11 = np.mean(f11[1:11])
    mean_recall1 = np.mean(recall1[1:11])
    print("accuracy1 的平均值是:", mean_accuracy1)
    print("precision1 的平均值是:", mean_precision1)
    print("f11 的平均值是:", mean_f11)
    print("recall1 的平均值是:", mean_recall1)

    mean_accuracy2 = np.mean(accuracy2[1:11])
    mean_precision2 = np.mean(precision2[1:11])
    mean_f12 = np.mean(f12[1:11])
    mean_recall2 = np.mean(recall2[1:11])
    print("accuracy1 的平均值是:", mean_accuracy2)
    print("precision1 的平均值是:", mean_precision2)
    print("f11 的平均值是:", mean_f12)
    print("recall1 的平均值是:", mean_recall2)
else:
    mean_accuracy1 = np.mean(accuracy1[1:11])
    mean_precision_macro = np.mean(precision_macro[1:11])
    mean_precision_micro = np.mean(precision_micro[1:11])
    mean_precision_weighted = np.mean(precision_weighted[1:11])
    mean_f1_macro = np.mean(f1_macro[1:11])
    mean_f1_micro = np.mean(f1_micro[1:11])
    mean_f1_weighted = np.mean(f1_weighted[1:11])
    mean_recall_macro = np.mean(recall_macro[1:11])
    mean_recall_micro = np.mean(recall_micro[1:11])
    mean_recall_weighted = np.mean(recall_weighted[1:11])

    print("accuracy1 的平均值是:", mean_accuracy1)
    print("precision_macro 的平均值是:", mean_precision_macro)
    print("precision_micro 的平均值是:", mean_precision_micro)
    print("precision_weighted 的平均值是:", mean_precision_weighted)
    print("f1_macro 的平均值是:", mean_f1_macro)
    print("f1_micro 的平均值是:", mean_f1_micro)
    print("f1_weighted 的平均值是:", mean_f1_weighted)
    print("recall_macro 的平均值是:", mean_recall_macro)
    print("recall_micro 的平均值是:", mean_recall_micro)
    print("recall_weighted 的平均值是:", mean_recall_weighted)



