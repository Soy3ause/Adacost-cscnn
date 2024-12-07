__author__ = '杨敬坤'
__edit_time__ = '2024.12.06'
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

df = pd.read_csv('glass.data.csv') #这里改数据集

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
kf = KFold(n_splits=3, shuffle=True, random_state=42)

#adaboostcnn多分类
import numpy
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Activation
from keras.layers import MaxPooling1D,Conv1D,UpSampling1D
from keras.models import Model
from keras import initializers


#############randome seed:
#seed = 100
seed = 50
numpy.random.seed(seed)
#TensorFlow has its own random number generator
# from tensorflow import set_random_seed
# set_random_seed(seed)
tensorflow.random.set_seed(seed)
####################
from sklearn.ensemble import AdaBoostClassifier
#from multi_AdaBoost import AdaBoostClassifier
from multi_AdaBoost import AdaBoostClassifier as Ada

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, make_scorer, confusion_matrix, roc_curve, auc, average_precision_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import OneHotEncoder# LabelBinarizer

#####deep CNN-binary-CS
from keras import backend as K
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

#theano doesn't need any seed because it uses numpy.random.seed
#######function def:
def train_CNN(X_train=None, y_train=None, epochs=None, batch_size=None, X_test=None, y_test=None, n_features =10, seed =100):
    ######ranome seed
    numpy.random.seed(seed)
    # set_random_seed(seed)
    tensorflow.random.set_seed(seed)

    model = baseline_model(n_features, seed)
    #reshape imput matrig to be compatibel to CNN
    newshape=X_train.shape
    newshape = list(newshape)
    newshape.append(1)
    newshape = tuple(newshape)
    X_train_r = numpy.reshape(X_train, newshape)#reshat the trainig data to (2300, 10, 1) for CNN
    #binarize labes:
    # lb=LabelBinarizer()
    # y_train_b = lb.fit_transform(y_train)

    lb=OneHotEncoder(sparse=False)
    y_train_b =y_train.reshape(len(y_train), 1)
    y_train_b = lb.fit_transform(y_train_b)
    #train CNN
    numpy.random.seed(seed)
    tensorflow.random.set_seed(seed)
    # set_random_seed(seed)
    model.fit(X_train_r, y_train_b, epochs=epochs, batch_size=batch_size)

    #####################reshap test data and evaluate:
    newshape = X_test.shape
    newshape = list(newshape)
    newshape.append(1)
    newshape = tuple(newshape)
    X_test_r = numpy.reshape(X_test, newshape)
    #bibarize lables:
    # lb=LabelBinarizer()
    # y_test_b = lb.fit_transform(y_test)
    lb=OneHotEncoder(sparse=False)
    y_test_b = y_test.reshape(len(y_test),1)
    y_test_b = lb.fit_transform(y_test_b)

    yp=model.evaluate(X_train_r, y_train_b)
    print('\nSingle CNN evaluation on training data, [loss, test_accuracy]:')
    print(yp)


    yp=model.evaluate(X_test_r, y_test_b)
    print('\nSingle CNN evaluation on testing data, [loss, test_accuracy]:')
    print(yp)
    return y_test_b
    ########################
#####deep CNN
def baseline_model(n_features=10, seed=100):
    numpy.random.seed(seed)
    # set_random_seed(seed)
    tensorflow.random.set_seed(seed)
	# create model
    model = Sequential()
    model.add(Conv1D(32, 3, padding = "same", input_shape=(n_features, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

#    model.add(Conv1D(32, 3, border_mode='valid',  activation='relu'))
#    model.add(MaxPooling1D(pool_size=(2, 2)))
#    model.add(Dropout(0.2))#

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.2))#
    model.add(Dense(64, activation='relu'))#

    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
	# Compile model
    numpy.random.seed(seed)
    # set_random_seed(seed)
    tensorflow.random.set_seed(seed)
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    #model.compile(loss=lambda y_true, y_pred: custom_weighted_binary_crossentropy(y_true, y_pred, fn_weight=1, fp_weight=1), optimizer='adagrad', metrics=['accuracy'])


    print (model.summary())
    return model
#from keras import backend as K
#K.set_image_dim_ordering('th')


#X, y = make_gaussian_quantiles(n_samples=13000, n_features=10,
#                               n_classes=3, random_state=1)


def reshape_for_CNN(X):
       ###########reshape input mak it to be compatibel to CNN
       newshape=X.shape
       newshape = list(newshape)
       newshape.append(1)
       newshape = tuple(newshape)
       X_r = numpy.reshape(X, newshape)#reshat the trainig data to (2300, 10, 1) for CNN

       return X_r


acc_scores = []
pr_macro_scores = []
pr_micro_scores = []
pr_weighted_scores = []
rc_macro_scores = []
rc_micro_scores = []
rc_weighted_scores = []
f1_macro_scores = []
f1_micro_scores = []
f1_weighted_scores = []

n_features = 10
n_classes = 6
X_contiguous = np.ascontiguousarray(X)
#y = medi['RES']
for train_index, test_index in kf.split(X_contiguous):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X_train = numpy.array(X_train)
    X_test = numpy.array(X_test)
    y_train = numpy.array(y_train)
    y_test = numpy.array(y_test)
    batch_size = 10
    X_train_r = reshape_for_CNN(X_train)
    X_test_r = reshape_for_CNN(X_test)

    ###########################################Adaboost+CNN:

    from multi_adaboost_CNN import AdaBoostClassifier as Ada_CNN

    # from test1 import AdaBoostClassifier as Ada_CNN
    n_estimators = 50
    epochs = 1
    bdt_real_test_CNN = Ada_CNN(
        base_estimator=baseline_model(n_features=n_features),
        n_estimators=n_estimators,
        learning_rate=1,
        epochs=epochs)
    bdt_real_test_CNN.fit(X_train_r, y_train, batch_size)
    # test_real_errors_CNN=bdt_real_test_CNN.estimator_errors_[:]


    y_pred_CNN = bdt_real_test_CNN.predict(X_test_r)
    acc_scores.append(accuracy_score(y_pred_CNN, y_test))
    pr_macro_scores.append(precision_score(y_pred_CNN, y_test, average='macro'))
    pr_micro_scores.append(precision_score(y_pred_CNN, y_test, average='micro'))
    pr_weighted_scores.append(precision_score(y_pred_CNN, y_test, average='weighted'))
    rc_macro_scores.append(recall_score(y_pred_CNN, y_test, average='macro'))
    rc_micro_scores.append(recall_score(y_pred_CNN, y_test, average='micro'))
    rc_weighted_scores.append(recall_score(y_pred_CNN, y_test, average='weighted'))
    f1_macro_scores.append(f1_score(y_pred_CNN, y_test, average='macro'))
    f1_micro_scores.append(f1_score(y_pred_CNN, y_test, average='micro'))
    f1_weighted_scores.append(f1_score(y_pred_CNN, y_test, average='weighted'))

acc_mean = np.mean(acc_scores)
pr_macro_mean = np.mean(pr_macro_scores)
pr_micro_mean = np.mean(pr_micro_scores)
pr_weighted_mean = np.mean(pr_weighted_scores)
rc_macro_mean = np.mean(rc_macro_scores)
rc_micro_mean = np.mean(rc_micro_scores)
rc_weighted_mean = np.mean(rc_weighted_scores)
f1_macro_mean = np.mean(f1_macro_scores)
f1_micro_mean = np.mean(f1_micro_scores)
f1_weighted_mean = np.mean(f1_weighted_scores)

print("准确率", acc_mean)
print("精确率宏平均", pr_macro_mean)
print("f1宏平均", f1_macro_mean)
print("召回率宏平均", rc_macro_mean)
print("精确率微平均", pr_micro_mean)
print("f1微平均", f1_micro_mean)
print("召回率微平均", rc_micro_mean)
print("精确率加权平均", pr_weighted_mean)
print("f1加权平均", f1_weighted_mean)
print("召回率加权平均", rc_weighted_mean)