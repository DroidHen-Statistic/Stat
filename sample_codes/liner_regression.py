import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# 数据
csv_file = os.path.join( os.path.dirname(os.path.abspath(__file__)), "data"+ os.path.sep + "CCPP" + os.path.sep + "Folds5x2_pp.csv")
with open(csv_file, "r") as f:

    reader = csv.reader(f)
    # data_len = len(reader)
    pos_to_name = np.array(5)
    # train_X = np.array((data_len, 4), dtype = float)
    # train_Y = np.array(data_len, dtype = float)

    X = []
    y = []
    y_class = [] # 分类
    for row, line in enumerate(reader):
        # print (line)
        # break
        if(row < 1):
            pos_to_name = line
            print(line)
            continue
        if(len(line) != 5):
            continue
        data_pos = row - 1
        X.append(list(map(float, line[:4]) ))
        y.append(float(line[-1]))
        y_class.append( int(float(line[-1]) > 454.3 ) )
        # print(X[data_pos], y[data_pos])
        # break
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
class My_est(BaseEstimator):
    """docstring for ClassName"""
    def __init__(self, est_, scoring):
        super(My_est, self).__init__()
        self.est_ = est_
        self.scoring = scoring
    def fit(self, X, y):
        self.est_.fit(X, y)
    def fit_transform(self, X, y):
        self.est_fit_transform(X, y)
    def fit_transform(self, X, y):
        self.est_fit_transform(X, y)
    def score(self, X, y):
        score = np.mean(cross_val_score(self.est_, X,
                                y, scoring=self.scoring, cv=2))
        return score


    

# 逻辑回归
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
alphas_num = 10
alphas = np.logspace(-1, 2, 4, base = 3)
reg = LogisticRegression('l1')
scores = []
for alpha in alphas:
    # print(alpha)
    reg.set_params(C = alpha, class_weight ='balanced') # class_weight={0:0.9, 1:0.1} 第一类权重0.9，第二类0.1
    # pipe = Pipeline([('reg', reg)])
    my_est = My_est(reg, "recall")
    pipe = Pipeline([('reg', my_est)])
    pipe.fit(X, y_class)
    scores.append(pipe.score(X, y_class))

import matplotlib.pyplot as plt

g1 = plt.gca()
# for pos, score in enumerate(scores):
#     g1.plot(alpha, score, label = "")
# g1.plot(alpha, )
# g1.plot(odds_pay_var, '-o', label = 'pay')
# plt.plot(odds_no_pay_var,'-o', label = 'no_pay')

plt.plot(alphas, scores,  label = "test")
plt.plot(np.array(alphas) + 2, scores,  label = "kk")
plt.legend(loc = "upper right")
plt.xlabel("alpha")
plt.ylabel("score")
plt.show()

# ax.set_xscale('log')
# #翻转x轴的大小方向，让alpha从大到小显示
# ax.set_xlim(ax.get_xlim()[::-1]) 
# plt.xlabel('alpha')
# plt.ylabel('weights')
# plt.title('Ridge coefficients as a function of the regularization')
# plt.axis('tight')
# plt.show()


   # plt.plot([x_[1] for x_ in x[-pay_count:]],'b-o')
    # plt.show()
    # plt.cla()

# import matplotlib.pyplot as plt



# 逻辑回归
# print(pos_to_name)
from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge
estimater = Ridge(alpha = 1)
# estimater.fit(X_train, y_train)
# print(estimater.coef_, estimater.intercept_, estimater.n_iter_) # 输出的分别是系数，常数和迭代次数，迭代次数只对某些求解算法有效

from sklearn.linear_model import RidgeCV
ridgecv = RidgeCV(alphas = list(np.arange(10) / 100 )[1:-1] , scoring= "neg_mean_absolute_error", store_cv_values =False, cv = 3 )
# ridgecv = RidgeCV(alphas = list(np.arange(10, dtype = float)[1:-1]) )
# ridgecv = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100])
# 常见错误：ValueError: cv!=None and store_cv_values=True  are incompatible 
#       alpha不能有0

# ridgecv = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100])

# ridgecv.fit(X, y)
# print(ridgecv.score(X, y))

# print(ridgecv.coef_, ridgecv.intercept_, ridgecv.alpha_) # 输出的分别是系数，常数和最后选择的alpha值
# print("------------------")
# print(ridgecv.cv_values_ ) # 交叉验证时候的alpha，只有 cv=None and store_cv_values=True的时候有用，具体含义没看！

X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)
n_alphas = 5
alphas = np.logspace(-10, -2, n_alphas) # 创建等比数列

# clf = Ridge()
# coefs = []
# for a in alphas:
#     clf.set_params(alpha = a)
#     clf.fit(X, y)
#     coefs.append(clf.coef_)
# clf = linear_model.Ridge(fit_intercept=False) # 是否有截距
clf = linear_model.Ridge()

from sklearn.pipeline import Pipeline
pip = Pipeline([('clf', clf)])
# pip.set_params(steps = [('clf2', clf)]) # 这个可以改变流程
# pip.named_steps['clf2'] = clf # 这个没用，不知道为啥
pip.fit(X_test, y_test)
pip.score(X_test, y_test)

exit()
coefs = []
# 循环200次
for a in alphas:
    #设置本次循环的超参数
    clf.set_params(alpha=a)
    #针对每个alpha做ridge回归
    clf.fit(X, y)
    # 把每一个超参数alpha对应的theta存下来
    coefs.append(clf.coef_)
    print(clf.coef_)
    # exit()





ax = plt.gca()

ax.plot(alphas, coefs)
#将alpha的值取对数便于画图
# print(ax.get_xlim())
# print(ax.get_xlim()[::-1])
# exit()
ax.set_xscale('log')
#翻转x轴的大小方向，让alpha从大到小显示
ax.set_xlim(ax.get_xlim()[::-1]) 
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()




# clf = linear_model.Ridge(fit_intercept=False)
# coefs = []
# # 循环200次
# for a in alphas:
#     #设置本次循环的超参数
#     clf.set_params(alpha=a)
#     #针对每个alpha做ridge回归
#     clf.fit(X, y)
#     # 把每一个超参数alpha对应的theta存下来
#     coefs.append(clf.coef_)