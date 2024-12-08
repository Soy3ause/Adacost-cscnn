import pandas as pd
from sklearn.model_selection import train_test_split
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
y = df_numeric.iloc[:, -1]   # 目标变量



# 划分测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 预测测试集
y_pred = clf.predict(X_test)

# 输出评估指标
print("准确率", accuracy_score(y_test, y_pred))
print("精确率", precision_score(y_test, y_pred, average='weighted'))  
print("f1", f1_score(y_test, y_pred, average='weighted'))  
print("召回率", recall_score(y_test, y_pred, average='weighted'))  