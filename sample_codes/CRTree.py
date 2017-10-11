import numpy as np
import os
import sys
# head_path = os.path.join(os.path.dirname(
#     (os.path.abspath(__file__))), "slot")
# print(head_path)
# sys.path.append(head_path)

head_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
print(head_path)
sys.path.append(head_path)

import config
from utils import *

from MysqlConnection import MysqlConnection
from collections import defaultdict
from time import time

from numpy import random as rand
import random
from scipy.stats import randint as sp_randint # 生成随机数
import pydotplus
import copy

# Utility function to report best scores
def report(results, n_top=10):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate])) # n折打分的标准差，表示波动多大
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

class UserData:
    features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time",
                "average_login_interval", "average_spin_interval", "average_bonus_win"]
    locales = ["US", "MY", "HU", "MM", "RU", "IT", "BR", "DE", "GR", "EG", "ES",
               "FR", "PT", "PL", "AU", "CA", "ID", "RO", "GB", "UA", "CZ", "NL", "SG"]
    locale_dict = {}

    @staticmethod
    def locales_2_num(locale):
        if len(UserData.locale_dict) != len(UserData.locales):
            i = 0
            for _locale in UserData.locales:
                UserData.locale_dict[_locale] = i
                i += 1
        try:
            return UserData.locale_dict[locale]
        except Exception as err:
            return -1

    def __init__(self, pay_weight = 10):
        self.pay_weight = pay_weight

    @staticmethod
    def transSqlValue(v, v_name):
        if v_name == "locale":
            v = UserData.locales_2_num(v)
        return v

    #从玩家数据里采样反例
    def get_X_y(self):
         # result_no_pay = random.sample(result_no_pay, pay_weight * len(result_pay))
        len_no_pay =  len(self.no_pay_vectors)
        len_pay   =  len(self.pay_vectors)
        pay_weight = self.pay_weight
        cr_no_pay = random.sample(list(self.no_pay_vectors), min(pay_weight * len_pay, len_no_pay))
        cr_all_sample = copy.copy(self.pay_vectors)
        for i in range(1, pay_weight):
            cr_all_sample = np.vstack((cr_all_sample, self.pay_vectors))
        cr_all_sample = np.vstack((cr_all_sample, cr_no_pay))

        # rand.shuffle(cr_all_feature)
        self.X = cr_all_sample[:,1:]
        self.y = cr_all_sample[:,0].astype(int)
        print("total len:%s" % len(self.y))


        # self.pay_flag = np.zeros()
        # self.vectors = np.zeros(
        #     ((total_len), len(argv) + 1))
        # self.vectors[:len_no_pay, 0] = 0
        # self.vectors[len_no_pay:total_len, 0] = 1
        # row = 0
        # for value in result_no_pay:
        #     i = 1
        #     for v_name in argv:
        #         self.vectors[row][i] = UserData.transSqlValue(value[v_name], v_name)
        #         i += 1
        #     row += 1
        # for value in result_pay:
        #     i = 1
        #     for v_name in argv:
        #         self.vectors[row:row + pay_weight, i] = UserData.transSqlValue(value[v_name], v_name)
        #         i += 1
        #     row += pay_weight
        # rand.shuffle(self.vectors)
        # self.X = self.vectors[:,1:]
        # self.y = self.vectors[:,0].astype(int)

    # 把玩家数据生成vecotr，这里不过滤,保存所有的
    def get_uid_2_vector(self, argv = []):
        # 取得原始数据
        conn = MysqlConnection(config.dbhost, config.dbuser,
                               config.dbpassword, config.dbname)
        sql = "select * from slot_user_profile where purchase_times = 0"
        result_no_pay = conn.query(sql)
        sql = "select * from slot_user_profile where purchase_times > 0"
        result_pay = conn.query(sql)
        # pay_weight = self.pay_weight
        # result_no_pay = random.sample(result_no_pay, pay_weight * len(result_pay))
        len_no_pay =  len(result_no_pay)
        len_pay = len(result_pay)
        # len_pay   =  len(result_pay * pay_weight)
        total_len = len_pay + len_no_pay
        # print("total len:%s" % total_len)
        if len(argv) == 0:
            argv = UserData.features
        # self.pay_flag = np.zeros()
        # self.pay_vectors = np.zeros(len_pay)
        self.pay_vectors = np.zeros(
             ((len_pay), len(argv) + 1))
        self.pay_vectors[:, 0 ]= 1
        self.no_pay_vectors = np.zeros(
             ((len_no_pay), len(argv) + 1))
        self.no_pay_vectors[:, 0 ]= 0
        # self.vectors = np.zeros(
        #     ((total_len), len(argv) + 1))
        # self.vectors[:len_no_pay, 0] = 0
        # self.vectors[len_no_pay:total_len, 0] = 1
        row = 0
        for value in result_no_pay:
            i = 1
            for v_name in argv:
                self.no_pay_vectors[row][i] = UserData.transSqlValue(value[v_name], v_name)
                i += 1
            row += 1
        row = 0
        for value in result_pay:
            i = 1
            for v_name in argv:
                self.pay_vectors[row][i] = UserData.transSqlValue(value[v_name], v_name)
                i += 1
            row += 1
        # rand.shuffle(self.vectors)
        # self.X = self.vectors[:,1:]
        # self.y = self.vectors[:,0].astype(int)

if __name__ == "__main__":
    user_data = UserData(pay_weight=5)
    # user_data.get_uid_2_vector(["bonus_times", "active_days", "locale"])
    user_data.get_uid_2_vector()

    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import StratifiedKFold


    from sklearn import tree # 树形结构
    from sklearn.metrics import recall_score
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import accuracy_score


    # 要选的超参数
    min_samples_per_node = list(range(16,64))
    max_deep = list(range(4,8))


    # path = file_util.get_figure_path("slot", "predict_purchase", "DT")
    path = file_util.get_figure_path("slot", "predict_purchase", "TestParams")
    acc_threshold = 0.75
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    fig.set_size_inches(15,12)
    axMean = fig.add_subplot(211)  # 增加一个子图，注意，fig要用add
    axStd = fig.add_subplot(212)  # 增加一个子图，注意，fig要用add
    # figMean = plt.figure(1)
    # figStd = plt.figure(2)
    # figMean.set_size_inches(12,6)
    # figStd.set_size_inches(12,6)
    # axMean =figMean.gca()
    # axStd =figStd.gca()

    ##################### 换x坐标，这几个一定要搭配起来用！#############################
    xlim = [0, len(min_samples_per_node)] # 把显示范围固定
    # ylim = [0, 1]
    axMean.set_xlim(xlim)
    # axMean.set_ylim(ylim)
    axMean.yaxis.set_major_locator(plt.LogLocator(2.1, [2.0])) # 按对数画刻度,第一个参数底数，第二个参数是倍数。比如这个例子在1,3,9,27...上标注刻度，也在2,6,12...上标注
    # axMean.set_yscale("log", basey=2)  # 对数坐标轴
    axMean.xaxis.set_major_locator(plt.MultipleLocator(1)) # 按1步长标示刻度
    axMean.set_xticklabels([-1] + min_samples_per_node) # 补一个-1，因为坐标原点强制不显示，不补就错位
    # # ax.set_xticklabels([-1] + max_deep) # 补一个-1，因为坐标原点强制不显示，不补就错位

    axStd.set_xlim(xlim)
    axStd.xaxis.set_major_locator(plt.MultipleLocator(1)) # 按1步长标示刻度
    axStd.set_xticklabels([-1] + min_samples_per_node) # 补一个-1，因为坐标原点强制不显示，不补就错位
    # # ax.set_xticklabels([-1] + max_deep) # 补一个-1，因为坐标原点强制不显示，不补就错位
    ##################### 换x坐标，这几个一定要搭配起来用！#############################
    axMean.set_title('means')
    axStd.set_title('stds')
    axMean.set_xlabel('min_sample')
    axStd.set_xlabel('min_sample')
    axMean.set_ylabel('recall_mean')
    axStd.set_ylabel('recall_std')
    # skf = StratifiedKFold(n_splits =3, shuffle =True)
    for test_time in range(0, 1):
        print("test time:%s" % test_time)
        user_data.get_X_y()
        X= user_data.X
        y = user_data.y
        # min_samples_per_node = [16,32,64,128]
        scores = [] # 得分
        stds = [] # 均值
        scoring = "recall"
        for min_sample in min_samples_per_node:
        # for deep in max_deep:
            cls = tree.DecisionTreeClassifier(max_depth=7, min_samples_split= min_sample, min_samples_leaf= min_sample, class_weight= {0: 1, 1: 5})
            cvClassfier = other_util.KFoldsClassifier(cls, ["recall", "accuracy"], n_splits =3, shuffle =True)
            cr_scores = cvClassfier.score(X, y)
            if cr_scores["mean"]["accuracy"] < acc_threshold:
                scores.append(0)
                stds.append(0)
            else:
                scores.append(cr_scores["mean"]["recall"] ** 2 *100)    
                stds.append(cr_scores["std"]["recall"])

            # dot_data = tree.export_graphviz(cvClassfier.cls_ , out_file=None, 
            #                                                             feature_names=UserData.features, 
            #                                                             class_names=["no_pay", "pay"],
            #                                                             filled=True, rounded=True,
            #                                                             impurity=False)
            # graph = pydotplus.graph_from_dot_data(dot_data)
            # graph.write_pdf(os.path.join(path, "tree%s.pdf" % test_time))


            # scores.append(np.mean(score_ret))
            # cr_var = np.array(score_ret).var()
            # variance.append(cr_var)
        # print(scores)
        # print(variance)

        # import matplotlib.pyplot as plt
        # plt.gcf().set_size_inches(12,6)
        scores = np.logspace(0,9,10, base =2)
        axMean.plot(scores)
        axStd.plot(stds)
        # plt.show()

        # # ax.plot(variance, label = "var")
        # ax.legend(loc="upper right")
    figure_path = file_util.get_figure_path("slot", "predict_purchase", "TestParams")
    file_name = os.path.join(figure_path, "tree_"  + "node" + ".png")
    # file_name_std = os.path.join(figure_path, "tree_"  + "std" + ".png")
    # gcf.savefig(file_name, dpi= 160)
    # if payed >= 1:
    # gcf.savefig(file_name)
    plt.show()
    fig.savefig(file_name)
    # figMean.savefig(file_name)
    # figStd.savefig(file_name_std)
    plt.cla()
    # figStd.cla()
    # plt.show()

    exit()


    param_dist = {"criterion": ["gini", "entropy"]
                    ,"max_depth" : [4,5,6,7]
                    # ,"min_samples_split" : sp_randint(9,81)
                    ,"min_samples_split" : [27,81]
                    ,"min_samples_leaf" : [27,81]
                    # ,"max_features" : [0.5, 1.0, "sqrt"]
                    , "class_weight" : [None]
    }
    n_iter_search = 3
    random_search = RandomizedSearchCV(estimator = tree.DecisionTreeClassifier(), param_distributions  = param_dist, scoring ="recall", n_jobs = 1)
    start = time()
    # exit()
    # __spec__ = "1"
    random_search.fit(user_data.X, user_data.y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    path = file_util.get_figure_path("slot", "predict_purchase", "DT")
    dot_data = tree.export_graphviz(random_search.best_estimator_ , out_file=None, 
                                                                feature_names=UserData.features, 
                                                                class_names=["no_pay", "pay"],
                                                                filled=True, rounded=True,
                                                                impurity=False)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(os.path.join(path, "tree5.pdf"))

    grid_search = GridSearchCV(estimator = tree.DecisionTreeClassifier, param_grid = {})

    # param_dist = {"max_depth": [5, None],
    #               "max_features": sp_randint(1, 11),
    #               "min_samples_split": sp_randint(2, 11),
    #               "min_samples_leaf": sp_randint(1, 11),
    #               "bootstrap": [True, False],
    #               "criterion": ["gini", "entropy"]}

    # 决策树
    myTree = tree.DecisionTreeClassifier()


    # k = UserData.locales_2_num("US")
    # kk = UserData.locales_2_num("BB")





 # 画ROC曲线
probas_ = pipe_lr.predict_proba(x_test) 
# print(pipe_lr.classes_ )

# 通过roc_curve()函数，求出fpr和tpr，以及阈值  
# 这里传进去的第二个参数是预测为正例的概率
# The predicted class probability is the fraction of samples of the same class in a leaf.
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)
#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来  
plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (1, roc_auc))

#画对角线
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')


plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()