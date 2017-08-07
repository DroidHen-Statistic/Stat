import numpy as np

from sklearn.datasets import load_iris
iris = load_iris()
iris_X = iris.data
iris_Y = iris.target


# 过滤方差太小的特征
from sklearn.feature_selection import VarianceThreshold
# X = np.array(range(30)).reshape(10,3)
# vt = VarianceThreshold()
# X [:,1] = 1
# X[0,1]=2
# #print(X)
# Xt = vt.fit_transform(X)
# #print(Xt)
# # 参数lass sklearn.feature_selection.VarianceThreshold(threshold=0.0)[source]

# # threshold 方差低于这个值被过滤掉
# vt.set_params(threshold=0.2)
# Xt = vt.fit_transform(X)
# print(Xt)
# print(vt.variances_) # 方差

# 根据相关性选择
# SelectKBest搭配的函数要有固定格式，下面是一个例子
from sklearn.feature_selection import SelectKBest
# 默认使用方差分析，留下方差最大的


# 皮尔逊相关系数,线性相关表现比较好
from scipy.stats import pearsonr  # 皮尔逊相关系数
# 和select搭配的相关性计算函数


def mul_score_pv(X, Y, my_func):
    """
    X: 训练数据, 列是特征，行是样本
    Y  训练标签
    my_func 相似度计算函数，格式和scipy.stats.pearsonr一样
        输入两个等长向量，输出一个元组(score, pv)
    样本数m，特征数f
    返回一个(score, p_value)的元祖
    score长度为特征，p_value长度一样
    """
    feature_len = len(X.T)
    scores = np.zeros((feature_len), dtype=float)
    p_value = np.zeros((feature_len), dtype=float)
    pos = 0
    for feature in X.T:
        # print (feature)
        # print(pos)
        tmp = my_func(feature, Y)
        scores[pos] = tmp[0]
        p_value[pos] = tmp[1]
        pos += 1
    return (scores, p_value)


def mul_score(X, Y, my_func):
    """
    同mul_score_pv很像，只是my_func只输出打分，最后的返回也只有打分的数组
    """
    feature_len = len(X.T)
    scores = np.zeros((feature_len), dtype=float)
    pos = 0
    for feature in X.T:
        # print (feature)
        # print(pos)
        scores[pos] = my_func(feature, Y)
        pos += 1
    return scores


def mul_pearson(X, Y):
    return mul_score_pv(X, Y, pearsonr)
    # """
    # X: 训练数据, 列是特征，行是样本
    # Y  训练标签
    # 样本数m，特征数f
    # 返回一个(score, p_value)的元祖
    # score长度为特征，p_value长度一样
    # """
    # feature_len = len(X.T)
    # scores = np.zeros((feature_len), dtype=float)
    # p_value = np.zeros((feature_len), dtype=float)
    # pos = 0
    # for feature in X.T:
    #     # print (feature)
    #     # print(pos)
    #     tmp = pearsonr(feature, Y)
    #     scores[pos] = tmp[0]
    #     p_value[pos] = tmp[1]
    #     pos += 1
    # return (scores, p_value)

from sklearn import metrics as mr

# SelectKBest(lambda X, Y: np.array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(X, Y)
X = np.array(range(30)).reshape(10, 3)
Y = (np.array(range(10)) > 5).astype(int)
# k = mul_pearson(X, Y)
# print(k)

selector = SelectKBest(mul_pearson, k=2)
selector.fit_transform(X, Y)

# 参数：
# print(selector.scores_) # 分数
# print(selector.pvalues_) # p值


# 卡方检验, 直接就可以用
from sklearn.feature_selection import chi2
# k = chi2(X, Y)
# print(k)

selector = SelectKBest(chi2, k=2)
selector.fit_transform(X, Y)


# 互信息，速度比卡方检验慢

def mul_mutula_info(X, Y):
    return mul_score(X, Y, mr.mutual_info_score)
from sklearn import metrics as mr
selector = SelectKBest(mul_mutula_info, k=2)
selector.fit_transform(X, Y)


# 参数：
# print(selector.scores_) # 分数

# 递归特征选择, 可以搭配各种回归模型，用这种模型反复迭代看哪个特征表现最差
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# 参数：class sklearn.feature_selection.RFE(estimator, n_features_to_select=None, step=1, verbose=0)[source]
# n_features_to_select： 选几个特征
# step表示每次迭代去掉的特征数，>=1表示每次减少这么多，小数表示按这个比例删减（0.3就表示每次删除30%）
RFE(estimator=LogisticRegression(),
    n_features_to_select=2).fit_transform(iris_X, iris_Y)


# 这是K折打分，过滤出几个特征自动帮你算了
from sklearn.feature_selection import RFECV
# class sklearn.feature_selection.RFECV(estimator, step=1, cv=None, scoring=None, verbose=0, n_jobs=1)
# 参数： cv，n折
#       scoring：打分标准
#       n_jobs:用几个cpu
selector = RFECV(estimator=LogisticRegression(),
                 n_jobs=1, cv=3,  scoring="accuracy")
selector.fit_transform(iris_X, iris_Y)
print(selector.n_features_) # 帮你选出来的特征
print(selector.support_ ) # 是哪几个特征


# 逻辑回归正则化为L1正则，参数绝对值求和
# l2正则会倾向于平均化每个参数，L1不会，所以L1适合拿来做特征选取
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
SelectFromModel(LogisticRegression(penalty="l1", C=0.1)
                ).fit_transform(iris.data, iris.target)
