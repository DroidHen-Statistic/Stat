import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import config


from functools import reduce
import numpy as np


def union_dict(*objs, f=lambda x, y: x + y, initial=0):
    """
    合并多个字典，相同的键，值相加

    union_dict({'a':1, 'b':2, 'c':3}, {'a':2, 'b':3}) ----> {'a':3, 'b':5, 'c':3}

    Arguments:
            *objs {dict} -- 要合并的字典

    Returns:
            [dict] -- 合并后的字典
    """
    keys = set(sum([list(obj.keys()) for obj in objs], []))
    total = {}
    for key in keys:
        total[key] = reduce(f, [obj.get(key, initial) for obj in objs])
    return total


def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):

        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


"""
以下为各种计算相关性的多维版本
请搭配sklearn.feature_selection.SelectKBest或类似的特征选择器使用
范例：
from sklearn.feature_selection import SelectKBest
X = np.array(range(30)).reshape(10, 3) # 10个样本，3个特征
Y = (np.array(range(10)) > 5).astype(int) # 10个样本的分类结果
selector = SelectKBest(mul_pearson, k =2)
selector.fit_transform(X, Y)
"""
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

# 多维的皮尔逊相关


def mul_pearson(X, Y):
    return mul_score_pv(X, Y, pearsonr)

# 多维的互信息
from sklearn import metrics as mr


def mul_mutula_info(X, Y):
    return mul_score(X, Y, mr.mutual_info_score)


# 卡方检验，本来就是多维的
from sklearn.feature_selection import chi2


from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
class ScoreEstimator(BaseEstimator):
    """ScoreEstimator
        可以打分的分类器
        传入想打分的类型，预测器和折叠次数
        可供流水线使用

    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression()
    reg.set_params(C = 0.1, class_weight ='balanced') # class_weight={0:0.9, 1:0.1}
    my_est = ScoreEstimator(reg, "recall")
    # my_est.set_params(C = 0.2)
    pipe = Pipeline([('reg', my_est)])
    from numpy.random import rand
    X = rand(10,2)
    print(X)
    y = rand(10) > 0.5 
    pipe.fit(X, y)
    print(my_est.predict(X))
    print(pipe.score(X, y))
    """
    def __init__(self, est_, scoring, cv=2):
        super(ScoreEstimator, self).__init__()
        self.est_ = est_
        self.scoring = scoring
        self.cv = cv

    def fit(self, X, y):
        self.est_.fit(X, y)

    # def set_params(self, **params):
    #     self.est_.set_params(params)

    # def get_params(self, deep=True):
    #     self.est_.set_params(deep)

    def predict(self, X):
        return self.est_.predict(X)

    # def fit_transform(self, X, y):
    #     self.est_fit_transform(X, y)

    def score(self, X, y):
        score = cross_val_score(self.est_, X,
                                        y, scoring=self.scoring, cv=self.cv)
        return score