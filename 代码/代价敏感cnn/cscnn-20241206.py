import numpy as np
import pandas as pd
import os, re, time, math, tqdm, itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
import plotly.offline as pyo
import seaborn as sns
import tensorflow as tf

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# !pip install interpret
# from interpret.blackbox import LimeTabular
# from interpret import show

import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import graphviz
import shap

import pickle

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('data/CIC_IDS_2017_binary_preprocessed.csv')

X=df.drop([" Labe1"], axis=1)
y=df[" Labe1"]
kf = KFold(n_splits=3, shuffle=True, random_state=42)

#cnn
import numpy
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
def custom_weighted_binary_crossentropy(y_true, y_pred, fn_weight=1.0, tp_weight=1.0, fp_weight=1.0, tn_weight=1.0):
    # 确保y_pred在epsilon和1-epsilon之间，以避免log(0)或log(1)的情况
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
    
    # 计算标准的二元交叉熵部分
    bce = - (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))

    fn_loss = fn_weight * (1 - y_pred) ** 2 * y_true  # 当y_true=1且y_pred较小时，损失较大
    tp_loss = tp_weight * y_pred ** 2 * y_true       # 当y_true=1且y_pred较大时，损失较小（但这里我们仍然保留y_true以确保只在y_true=1时计算）
    fp_loss = fp_weight * y_pred ** 2 * (1 - y_true) # 当y_true=0且y_pred较大时，损失较大
    tn_loss = tn_weight * (1 - y_pred) ** 2 * (1 - y_true) # 当y_true=0且y_pred较小时，损失较小（但同样地，我们保留1-y_true）
    
    weighted_loss = bce + fn_loss + fp_loss  # 我们在这里忽略了tp_loss和tn_loss，因为它们通常会在正确分类时很小
    
    return K.mean(weighted_loss)

n_features = 23  # 特征数量
n_classes = 2  # 类别数量

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 重塑数据以适应CNN
X_train=numpy.array(X_train)
X_test=numpy.array(X_test)
y_train=numpy.array(y_train)
y_test=numpy.array(y_test)
X_train_r = X_train.reshape((X_train.shape[0], n_features, 1))
X_test_r = X_test.reshape((X_test.shape[0], n_features, 1))

# 将标签转换为one-hot编码
y_train_c = to_categorical(y_train, num_classes=n_classes)
y_test_c = to_categorical(y_test, num_classes=n_classes)

# 定义CNN模型
model = Sequential()
model.add(Conv1D(32, 3, padding='same', input_shape=(n_features, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_classes, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

# 训练模型
model.fit(X_train_r, y_train_c, epochs=10, batch_size=32, validation_split=0.2)

# 评估模型
loss, accuracy = model.evaluate(X_test_r, y_test_c)
print(f'Test accuracy: {accuracy}')

from sklearn.metrics import precision_score, recall_score, f1_score

# 获取预测结果
y_pred_probs = model.predict(X_test_r)
y_pred_classes = np.argmax(y_pred_probs, axis=1)  # 将预测概率转换为类别标签

# 计算精确率、召回率和F1分数
precision = precision_score(np.argmax(y_test_c, axis=1), y_pred_classes)
recall = recall_score(np.argmax(y_test_c, axis=1), y_pred_classes)
f1 = f1_score(np.argmax(y_test_c, axis=1), y_pred_classes)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')