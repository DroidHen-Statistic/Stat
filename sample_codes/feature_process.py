import os
import sys
import numpy as np
import sklearn
from sklearn.datasets import load_iris

# from sys import argv
abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)
from utils import *

iris = load_iris()

a = [1, 2, 3]
np.var(a)  # 算方差
np.cov(a)  # 算相关性



# 特征的预处理

# 改造成标准差1，均值0, 如果是二维的就按列处理，比如150*4，就求出一共4列，每列单独处理（列当做特征）
# 方差 = 标准差的平方，所以方差也是1
# 对非高斯分布表现较差
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(iris.data)  # 喂数据
data = scaler.transform(iris.data)  # 转换数据

scaler = StandardScaler()
data = scaler.fit_transform(iris.data)  # 喂+转换数据

scaler.mean_  # 这是喂过以后的均值
scaler.scale_  # 这是喂过以后的方差

# 其他变量的去查文档
scaler.set_params(with_mean=False)  # 设置参数

# 三个参数
# class sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)[source]
# with_mean 是否将均值变成0，False不动均值
# with_std 是否将方差置为1，False不动方差，而且也不会计算方差
# copy : False: 不copy数值，原地返回
scaler = StandardScaler(copy=False)
data = iris.data
scaler.fit(data)
# 原地变换，不需要返回赋值，如果copy = True, 需要这么写 data = scaler.transform(data)
scaler.transform(data)

# 改造成0到1之间的数，线性缩放 (x - min) / max ，最大值是1，最小值是0
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# 和前面的用法一样

# 两个参数class sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)[source]
# copy 和前面一样
scaler.set_params(feature_range=(0, 10))
scaler.fit(data)
scaler.transform(data)

# 标准化，按行处理，把每行的长度变为1
# Normalization主要思想是对每个样本计算其p-范数，然后对该样本中每个元素除以该范数，这样处理的结果是使得每个处理后样本的p-范数（l1-norm,l2-norm）等于1。
# p-范数的计算公式：||X||p=(|x1|^p+|x2|^p+...+|xn|^p)^1/p
# 默认就是把每个样本的模长变成1
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
scaler.fit(data)
scaler.transform(data)

# 参数 class sklearn.preprocessing.Normalizer(norm='l2', copy=True)[source]¶
# l2表示2次方求和，欧氏距离。l1就是直接相加，


# 二值化
from sklearn.preprocessing import Binarizer
scaler = Binarizer()
scaler.fit(data)
scaler.transform(data)


# 参数class sklearn.preprocessing.Binarizer(threshold=0.0, copy=True)[source]
b = np.array([0, 2, 4])
b = b.reshape(-1, 1)
scaler = Binarizer(threshold=0)  # 超过0的才保留
# ....

# 哑值编码
from sklearn.preprocessing import OneHotEncoder

a = np.array([[0, 2, 4], [3, 2, 5], [1, 2, 4]])
coder = OneHotEncoder()
coder.fit(a)
coder.transform(a)

# 参数 class sklearn.preprocessing.OneHotEncoder(n_values='auto',
# categorical_features='all', dtype=<type 'numpy.float64'>, sparse=True,
# handle_unknown='error')

# sparse,是否采用稀疏矩阵形式
coder = OneHotEncoder(sparse=False)

# categorical_features 处理那些特征，按列算
coder.set_params(categorical_features=[2])  # 只处理第3个特征。 注意，处理过后特征的顺序会变化
coder.set_params(categorical_features=[0, 1])  # 处理1,2个特征
coder.fit_transform(a)

# n_values 处理以后向量的长度，注意这个长度是和实际值挂钩的
a = np.array([[0,  2,  4],
              [3,  2,  5],
              [10,  2,  4]])
coder.set_params(categorical_features=[0])

coder.set_params(n_values=10)  # 报错，因为第一列最大值是10(从0开始算)
coder.set_params(n_values=11)  # 成功
coder.fit_transform(a)

# 其他的不常用
# handle_unknown : str, ‘error’ or ‘ignore’
# Whether to raise an error or ignore if a unknown categorical feature is
# present during transform.

# dtype : number type, default=np.float
# Desired dtype of output.


# 补齐缺失的参数
from sklearn.preprocessing import Imputer
a = np.array([[0,  2,  4],
              [3,  2,  5],
              [10,  2,  4]])
np.vstack((a, [None] * 3))

im = Imputer()
im.fit_transform(a)

# 参数 class sklearn.preprocessing.Imputer(missing_values='NaN',
# strategy='mean', axis=0, verbose=0, copy=True)

im.set_params(strategy="mean")  # 均值
im.set_params(strategy="median")  # 中位数
im.set_params(strategy="most_frequent")  # 最频繁
im.fit_transform(a)

# missing_values : integer or “NaN”, optional (default=”NaN”)
# The placeholder for the missing values. All occurrences of
# missing_values will be imputed. For missing values encoded as np.nan,
# use the string value “NaN”.

# 行还是列推断
im.set_params(axis=1)
a = a.T
im.fit_transform(a)

# verbose，控制详细程度


# 增加特征之多项式处理
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures()
a = np.array([[0, 1],
              [1, 2],
              [2, 3],
              [3, 4]])
poly.fit_transform(a)
# 参数PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)

# degree : integer
#   The degree of the polynomial features. Default = 2.
poly.set_params(degree=1)  # 一次多项式 1， x1， x2
poly.fit_transform(a)
# interaction_only : boolean, default = False
#   If true, only interaction features are produced: features that are
#   products of at most degree distinct input features (so not x[1] ** 2,
#   x[0] * x[2] ** 3, etc.). # 只有交叉项，每个特征最多出现一次
poly.set_params(interaction_only=True)
poly.fit_transform(a)  # 没有x1平方这种项了
# include_bias : boolean
#   If True (default), then include a bias column, the feature in which all
#   polynomial powers are zero (i.e. a column of ones - acts as an intercept
#   term in a linear model).
poly.set_params(include_bias=False)
poly.fit_transform(a)  # 没有常数项

# TODO 其他的选择，看 sklearn.preprocessing的文档
Here is a veteran of porn Maki Hojo. She has began doing porn in 2006 and she is still active. During her more than 10 years career in this business she did fucked so many dicks that it is impossible to count them. Gangbangs, oral bangs toy fucking sessions and bukakke... Maki is doing all these things and she enjoys this more than anything else in her life. That is why Hojo san is not going to finish her career in the near future.