import os
import sys
import numpy as np
from numpy import random as rand

# 均匀分布，[0,1)
a = rand.rand(3)  # array([ 0.3482032 ,  0.8529326 ,  0.29806286])

"""array([[ 0.16264963,  0.47026467,  0.63152363],
        [ 0.16287412,  0.24696563,  0.71448236]]) """
a = rand.rand(2, 3)

a = rand.randn(2, 3)  # 2*3纬度的正态分布

# randint(low[, high, size])
a = rand.randint(3, 6)  # 3到5之间的一个数，6不包含
"""
array([[6, 6],
       [3, 2],
       [5, 3]])
"""
a = rand.randint(1, 10, size= [3,2])
"""
每项都在0,1之间, 连续的均匀分布，看文档也不知道和rand()有啥区别
array([[ 0.66246962,  0.03469166,  0.9865776 ],
       [ 0.47714682,  0.87002386,  0.98806874]])
"""
a = rand.random_sample(size= (2,3)) 


# sklearn的测试数据
import sklearn as sklearn
import matplotlib.pyplot as plt

# # 线性回归
# """
# 这里我们使用make_regression生成回归模型数据。几个关键参数有n_samples（生成样本数）
# ， n_features（样本特征数），noise（样本随机噪音）和coef（是否返回回归系数）。例子代码如下：
# """
# from sklearn.datasets.samples_generator import make_regression
# X, y, coef =make_regression(n_samples=1000, n_features=1,noise=10, coef=True) # noise是标准差
# # print(X, y ,coef)
# plt.scatter(X,y)
# plt.plot(X, X * coef, color ="black")
# # plt.scatter(X, y,  color='black')
# # plt.plot(X, X*coef, color='blue',
# #          linewidth=3)

# # plt.xticks(())
# # plt.yticks(())

# plt.show()

# # 分类模型
# """
# 　这里我们用 make_classification 生成三元分类模型数据。几个关键参数有n_samples（生成样本数）， 
# n_features（样本特征数）， n_redundant（冗余特征数）和n_classes（输出的类别数），例子代码如下：
# n_clusters_per_class： 每类有几个簇
# """
# from sklearn.datasets.samples_generator import make_classification
# """
# # X1为样本特征，Y1为样本类别输出， 共400个样本，每个样本2个特征，输出有3个类别，没有冗余特征，每个类别一个簇
# X1, Y1 = make_classification(n_samples=400, n_features=2, n_redundant=0,
#                              n_clusters_per_class=1, n_classes=3)
# """
# X, y = make_classification(n_samples = 1000, n_features= 2, n_redundant= 0, n_classes=3, n_informative = 2, n_clusters_per_class= 1)
# x_class0 = X[y==0]
# x_class1 = X[y==1]
# x_class2 = X[y==2]

# print(x_class0.dtype)
# print(x_class0.shape)
# print(x_class0[:,0])
# print(x_class0[:,1])

# # plt.scatter([1,2], [2,4])
# # plt.scatter([-1.17385393, -0.32160158, -0.44750658, -0.26539759], [-0.90838756, -1.29470974, -1.35971911, -1.37039875])
# plt.scatter(x_class0[:,0], x_class0[:,1], color = "red", label="class0")
# plt.scatter(x_class1.T[0],x_class1.T[1], color = "blue", label="class 1")
# plt.scatter(x_class2.T[0],x_class2.T[1], color = "black", label="class 2")

# plt.legend()
# plt.show()

# exit()

# # 聚类模型
# """
# 这里我们用make_blobs生成聚类模型数据。几个关键参数有n_samples（生成样本数）， n_features（样本特征数），centers(簇中心的个数或者自定义的簇中心)和cluster_std（簇数据方差，代表簇的聚合程度）。例子如下：
# """
# from sklearn.datasets.samples_generator import make_blobs
# # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共3个簇，簇中心在[-1,-1], [1,1], [2,2]， 簇方差分别为[0.4, 0.5, 0.2]
# X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,2], [1,1], [2,2]], cluster_std=[0.4, 0.5, 0.2])
# # plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
# plt.show()

"""
我们用make_gaussian_quantiles生成分组多维正态分布的数据。几个关键参数有n_samples（生成样本数）， 
n_features（正态分布的维数），mean（特征均值）， cov（样本协方差的系数）， n_classes（数据在正态分布中按分位数分配的组数）。 例子如下：
"""
from sklearn.datasets import make_gaussian_quantiles
#生成2维正态分布，生成的数据按分位数分成3组，1000个样本,2个样本特征均值为1和2，协方差系数为2
X1, Y1 = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=3, mean=[1,2],cov=2)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
plt.show()