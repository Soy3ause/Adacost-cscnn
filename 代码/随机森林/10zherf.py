
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight, compute_class_weight
from sklearn.metrics import average_precision_score
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier


# K折交叉验证评价指标
def pjzb_k(y, y_pred_k):
    print("K=10折交叉验证结果")
    y_pred_k = y_pred_k.reshape(-1, 1)
    print("mAP", average_precision_score(y, y_pred_k))
    print("准确率", accuracy_score(y, y_pred_k))

    print("精确率宏平均", precision_score(y, y_pred_k, average='macro'))
    print("f1宏平均", f1_score(y, y_pred_k, average='macro'))
    print("召回率宏平均", recall_score(y, y_pred_k, average='macro'))

    print("精确率微平均", precision_score(y, y_pred_k, average='micro'))
    print("f1微平均", f1_score(y, y_pred_k, average='micro'))
    print("召回率微平均", recall_score(y, y_pred_k, average='micro'))

    print("精确率加权平均", precision_score(y, y_pred_k, average='weighted'))
    print("f1加权平均", f1_score(y, y_pred_k, average='weighted'))
    print("召回率加权平均", recall_score(y, y_pred_k, average='weighted'))

   

   




file_path = ('glass.data.csv')
data = pd.read_csv(file_path)







X = data.iloc[:, :-1]  
y = data.iloc[:, -1] 



# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化(对特征做处理）
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
X = ss.fit_transform(X)

# 模型训练及预测
model = RandomForestClassifier()


# 1
""""""
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


#设置k折交叉验证

k = 10
scores = cross_val_score(model, X, y, cv=k)
# 输出k折交叉验证的准确率
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# 执行k折交叉验证并获取预测结果
y_pred_k = cross_val_predict(model, X, y, cv=k)



# 多分类
# 模型评价
# acc准确率，pr精确率，rc召回率，f1

print(file_path)
pjzb_k(y,y_pred_k) # K折交叉验证评价指标










