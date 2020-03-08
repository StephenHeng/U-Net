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
import numpy as np
os.environ["PATH"] += os.pathsep + 'D:/release/bin'
#本代码主要是利用乳腺数据集的特征来对乳腺超声图像进行良恶性的判断
#仍然使用自带的iris数据
# iris = datasets.load_iris()
# X = iris.data
# print(X)
# y = iris.target
# print(type(y))
#使用自己的数据集
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

dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=labels,
                         class_names=target,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
# 使用ipython的终端jupyter notebook显示。
Image(graph.create_png())
# 如果没有ipython的jupyter notebook，可以把此图写到pdf文件里，在pdf文件里查看。
graph.write_pdf("iris.pdf")
