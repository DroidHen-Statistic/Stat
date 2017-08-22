import numpy as np
import matplotlib.pyplot as plt
import csv
import os

aplha = 1
max_iter = 1000 # 最大迭代数

# # 最简单的线性回归
# x1 = np.arange(0, 20)
# x2 = np.arange(20, 0, -1)
# y = 3 + 2 * x1 + 4 * x2
# y = y.astype(float)
# # 加噪声
# from numpy import random as rand
# y += rand.randn(len(y))

# from sklearn import datasets, linear_model
# from sklearn.linear_model import Ridge

# ridge = Ridge(alpha = aplha, max_iter= max_iter)
# x = np.vstack((x1, x2)).T
# ridge.fit(x, y)
# y_pre = ridge.predict(x)
# print(ridge)

# import matplotlib.pyplot as plt

# plt.plot(x[:,0], y_pre, label="y_pre")
# plt.plot(x[:,0], y, label="y_ture")
# plt.xlabel("x values")
# plt.ylabel("y values")
# from sklearn.metrics import mean_squared_error as mse
# err = mse(y, y_pre)
# print(err)
# plt.legend()
# plt.show()
# exit()


# 对回归进行分析
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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# # 交叉选择cv
# from sklearn.linear_model import RidgeCV
# alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100]
# scoring = "neg_mean_squared_error" # 平方最小
# scoring = "neg_mean_absolute_error" # 绝对值最小
# scoring = "neg_median_absolute_error" # 误差绝对值的中位数最小
# ridgecv = RidgeCV(alphas= alphas, scoring=scoring, cv =5)
# ridgecv.fit(X_train, y_train)
# print(ridgecv.alpha_)

# 照着别人博客写的
from sklearn.linear_model import Ridge
import numpy.random as rand
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
# y = np.ones(10)
y = rand.randn(10) * 10
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)
coefs = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha) 
    # ridge = Ridge(alpha=alpha, fit_intercept=False) # 如果有bias，按博客的样例，y都是1，那让所有系数都是0就好了
    ridge.fit(X, y)
    coefs.append(ridge.coef_)


# # 使用前面的数据，自己写的
# # 把alpha的图画出来，随着alpha的增大，theda迅速变成0
# alphas = np.logspace(1, 2, 3, base= 10) * 0.01
# # alphas = np.logspace(0.1, 1, 2)
# from sklearn.linear_model import Ridge
# coefs = []
# # ridge = Ridge()
# for alpha in alphas:
#     ridge = Ridge(alpha=alpha)
#     # ridge.set_params(alpha=alpha)
#     ridge.fit(X_train, y_train)
#     coefs.append(ridge.coef_)

coefs = np.array(coefs)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("alpha")
ax.set_ylabel("theta")

# x = [1,2]
# y = [[1,2], [3,4]]
# ax.plot(x,y)
# print(alphas, coefs.T[0])
# ax.plot(alphas, coefs, "-o")
for pos, cr_theta in enumerate(coefs.T):
    ax.set_xscale("log",basex=10) 
    ax.plot(alphas, cr_theta, "-", label = "theta%s" % pos)

plt.legend(loc="upper right")
plt.show()

# from sklearn import metrics as mt
# y = [0,0,0,0,0]
# y_pre = [0,1,7,9,10]
# print(mt.neg_median_absolute_error(y, y_pre))


# print(alpha)
exit()





# 打分
from sklearn.base import BaseEstimator
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


#交叉验证
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
exit()

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
# plt.title('Ridge coefs as a function of the regularization')
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
plt.title('Ridge coefs as a function of the regularization')
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