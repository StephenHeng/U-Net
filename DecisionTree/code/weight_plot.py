from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

from IPython.display import Image
from sklearn import tree
import pydotplus
import os
import data_process
os.environ["PATH"] += os.pathsep + 'D:/release/bin'

# 使用自带数据
dataset = data_process.datasets
A = np.array(dataset)
X = []
y = []
for item in dataset:
    if len(item) != 0 :
        y.append(item[-1])
        for i in range(len(item)-1):
            X.append(item[i])
X = np.reshape(X, (1599, 5))
y = np.array(y)
labels = ['1', '2', '3', '4', '5']
target = np.array(['0', '1'])
# 训练模型，限制树的最大深度4
clf = DecisionTreeClassifier(max_depth=4)
#拟合模型
clf.fit(X, y)

y_importances = clf.feature_importances_
x_importances = labels
y_pos = np.arange(len(x_importances))
# 横向柱状图
plt.barh(y_pos, y_importances, align='center')
plt.yticks(y_pos, x_importances)
plt.xlabel('Importances')
plt.xlim(0,1)
plt.title('Features Importances')
plt.show()

# 竖向柱状图
plt.bar(y_pos, y_importances, width=0.4, align='center', alpha=0.4)
plt.xticks(y_pos, x_importances)
plt.ylabel('Importances')
plt.ylim(0,1)
plt.title('Features Importances')
plt.show()
