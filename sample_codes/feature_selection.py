import os
import sys
import numpy as np
import sklearn
from sklearn.datasets import load_iris

# from sys import argv
abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(abs_path)
sys.path.append(abs_path)
from utils import *

# # 方差选择
# from sklearn.feature_selection import VarianceThreshold
# X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]

# sel = VarianceThreshold(threshold=(.8 * (1 - .8))) # 小于threshod的都被去除掉
# X2 = sel.fit_transform(X)
# print(sel.variances_) # 可以打印出每个特征的方差，可以由此判断被剔除的特征
# print(sorted(enumerate(sel.variances_), key = lambda x : x[1], reverse= True))
# print(X2)

# 选k个最好的
from sklearn.feature_selection import SelectKBest

# 皮尔逊相关系数
from scipy.stats import pearsonr

# np.random.seed(0)
# size = 300
# x = np.random.normal(0, 1, size)
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# 使用卡方检验
from sklearn.feature_selection import chi2
selector = SelectKBest(chi2, k=2)
X_new = selector.fit_transform(X, y)
print(X_new)
print(X[-1])
print(selector.scores_ ) # 每个特征的得分
print(selector.pvalues_ ) # 每个特征的置信度 p value

# 用utils里包装过的皮尔逊相关系数计算(特征和结果的相似度)
# selector = SelectKBest(other_util.mul_pearson, k=2)
# X_new = selector.fit_transform(X, y)
# print(X_new)
# print(selector.scores_ ) # 每个特征的得分
# print(selector.pvalues_ ) # 每个特征的置信度 p value

# 用自己包的互信息计算
# selector = SelectKBest(other_util.mul_mutula_info, k = 2)
# X_new = selector.fit_transform(X, y)
# print(X_new)
# print(selector.scores_ ) # 每个特征的得分
# # print(selector.pvalues_ ) # 互信息没有置信度
