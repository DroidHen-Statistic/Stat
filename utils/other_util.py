import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import config
import numpy as np

from functools import reduce
import numpy as np
import copy
import re
import gc

"""
反转字典的k和v
"""


def flip_dict(d):
    return {pair[1]: pair[0] for pair in d.items()}


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

        返回list，每项是每折的打分

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


from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from collections import defaultdict
# import exceptions


class KFoldsClassifier():
    """KFoldEstimator
        K折打分，会保留训练模型，并保留已经算过的分数
    """

    def __init__(self, cls_, score_name, **argv_for_cv):
        self.cls_ = cls_
        self.skf_ = StratifiedKFold(**argv_for_cv)
        self.score_mean = {}
        self.score_std = {}
        for name in score_name:
            self.score_mean[name] = -1
            self.score_std[name] = -1

    def score(self, X, y, pay_weight = 1):
        split_data = self.skf_.split(X, y)
        all_scores = defaultdict(list)
        for train, dev in split_data:
            pay_index = [i for i, y in enumerate(y[train]) if y == 1]
            nopay_index = [i for i, y in enumerate(y[train]) if y != 1]
            X_pay = X[train][pay_index]
            X_nopay = X[train][nopay_index]
            X_train = copy.copy(X_pay)
            for i in range(1, pay_weight):
                X_train = np.vstack((X_train, X_pay))
            X_train = np.vstack((X_train, X_nopay))
            y_train = np.array([1] * len(X_pay) * pay_weight + [0] * len(X_nopay))
            # X_train = X[train]
            # y_train = y[train]
            self.cls_.fit(X_train, y_train)
            # cls.predict_proba(X[dev])
            y_pre = self.cls_.predict(X[dev])
            # right = sum(y_pre == y)
            # err = len(y) - right
            cr_score = 0
            for name in self.score_mean.keys():
                if(name == "recall"):
                    cr_score = recall_score(y[dev], y_pre)
                    # print("recall : ", cr_score)
                elif (name == "accuracy"):
                    cr_score = accuracy_score(y[dev], y_pre)
                    # print("accuracy : ", cr_score)
                elif (name == "precision"):
                    cr_score = average_precision_score(y[dev], y_pre)
                else:
                    raise (Exception("wrong score name:%s, use recall, accuracy or precision" % name))
                all_scores[name].append(cr_score)
        for name in self.score_mean.keys():
            self.score_mean[name] = np.mean(all_scores[name])
            self.score_std[name] = np.array(all_scores[name]).std()

        return {"mean":self.score_mean, "std":self.score_std}


import socket
import struct


def ip2long(ipstr):
    return struct.unpack("!I", socket.inet_aton(ipstr))[0]


def long2ip(ip):
    return socket.inet_ntoa(struct.pack("!I", ip))

import csv


class IPDB(object):

    def __init__(self, filename):
        self.ipslist = []
        self.ipelist = []
        self.countries = []
        self.length = 0
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                ips = row[0]
                ipe = row[1]
                country = row[2]
                self.ipslist.append(ip2long(ips))
                self.ipelist.append(ip2long(ipe))
                self.countries.append(country)
        self.length = len(self.ipslist)

    def findcc(self, s, e, ip):
        if (s > e):
            return false
        n = int((s + e) / 2)
        r = self.compare(ip, n)
        if (r == -1):
            return self.findcc(s, n - 1, ip)
        elif (r == 1):
            return self.findcc(n + 1, e, ip)
        else:
            return self.countries[n]

    def ip2cc(self, ip):
        s = 0
        ip = ip2long(ip)
        e = self.length
        ret = self.findcc(s, e - 1, ip)
        return ret

    def compare(self, ip, index):
        if ip < self.ipslist[index]:
            return -1
        elif ip > self.ipelist[index]:
            return 1
        return 0


def cdf(data):
    data_dis = dict(zip(*np.unique(data, return_counts=True)))
    x = list(data_dis.keys())
    _y = list(data_dis.values())
    y = [0]
    l = len(data)
    for i,a in enumerate(_y):
        y.append(y[i] + a / l)
    return [x, y[1:]]



def add(x,y):
    return x + y

def substract(x,y):
    return x - y

def times(x,y):
    return x * y

def divide(x,y):
    return (x + 0.001)/(y + 0.001)

CrossMethod = {'+':add,
               '-':substract,
               '*':times,
               '/':divide,}

def processCombine(df,col):
    pattern = re.compile('(.*?)([+-/*//])(.+)')  
    ops = ['+','-','*','/']
    for k in col:
        res = pattern.match(k)
        if(res):
            print("find combine", k)
            k1 = res.group(1)
            op = res.group(2)
            k2 = res.group(3)
            # print(res.group(0), k1, op, k2)
        else:
            # print('cant find  ' , k)
            continue
        df[k] = CrossMethod[op](df[k1], df[k2])
    return df

def dataCombine(df, col):
    
    """ edit col for your combined features """
#     col =['user_age_level', 'context_page_id', 'predict_category_property2', 'user_cnt_hour_1', 'item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'day', 'user_item_cumcount', 'user_shop_nunique', 'user_cumcount', 'len_item_property', 'hour', 'len_predict_category_property', 'len_item_category', 'gender0', 'shop_score_delivery0', 'shop_cumcount', 'gender_star', 'shop_star_level', '(user_age_level-item_collected_level)', '(context_page_id+shop_cumcount)', '(context_page_id/user_item_cumcount)', '(predict_category_property2-item_pv_level)', '(predict_category_property2*item_price_level)', '(predict_category_property2-item_category_list)', '(item_category_list-len_predict_category_property)', '(item_price_level*user_cumcount)', '(item_sales_level/shop_score_delivery0)', '(item_sales_level*len_item_category)']
    # df = df[~pd.isnull(df.is_trade)]
    df = processCombine(df,col)
    
    # col.append('is_trade')
    # df = df[col]
    gc.collect()
    return df