'''
这里是利用用户信息进行付费预测
使用的数据是数据库slot_user_profile_tmp 或者 slot_churn_profile里的数据

'''

import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')  # 服务器上跑
import os
import sys
from ready_for_train import Vector_Reader as v_reader
head_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
# print(head_path)
sys.path.append(head_path)
import config
from utils import *

from MysqlConnection import MysqlConnection
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from scipy import interp  

import pydotplus

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB


def plot_tree():
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data) 
    graph.write_pdf(os.path.join(path,"tree.pdf"))

def train(x, y):
    scaler = MinMaxScaler()
    # x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    pay_count = np.sum(y)
    print("pay_count: %d" %pay_count)
    print(len(x))   

    # x_train = x
    # y_train = y

    clfs = {"DT":tree.DecisionTreeClassifier(max_depth = 9, class_weight = {0:1,1:5},min_samples_split = 21, min_samples_leaf = 21),
        "RF":RandomForestClassifier(n_estimators = 20, max_depth = 8, class_weight = {0:1,1:5}, min_samples_split = 20, min_samples_leaf = 20, n_jobs = 2),"ExtraTrees":ExtraTreesClassifier(),"AdaBoost":AdaBoostClassifier(),"GBDT":GradientBoostingClassifier()}
    # clfs = {"RF":RandomForestClassifier(max_depth = 10)}
    # clfs = {"AdaBoost":AdaBoostClassifier(),"GBDT":GradientBoostingClassifier()}
    # clfs = {"DT":tree.DecisionTreeClassifier(max_depth = 9, class_weight = {0:1,1:5},min_samples_split = 21, min_samples_leaf = 21)} 
    for name, clf in clfs.items():
        print("------%s-------" % name)
        # pipe_lr = Pipeline([('feature_selection', SelectKBest(lambda X, Y: np.array(list(map(lambda x:pearsonr(x, Y)[0], X.T))).T, k=7)),('clf', clf)])
        # pipe_lr = Pipeline([('feature_selection',SelectKBest(other_util.mul_pearson, k=7)),('clf', clf)])
        # pipe_lr = Pipeline([('scaler', scaler),('feature_selection',VarianceThreshold(threshold = 0.01)),('clf', clf)])
        pipe_lr = Pipeline([('clf', clf)])
        # pipe_lr = Pipeline([('feature_selection', RFECV(estimator=clf, step=1, cv=3, scoring='recall')),('clf', clf)])
        cross_accuracy = np.mean(cross_val_score(pipe_lr, x_train,
                                y_train, scoring="accuracy", cv=10))
        cross_recall = np.mean(cross_val_score(pipe_lr, x_train,
                                y_train, scoring="recall", cv=10))
        cross_precision = np.mean(cross_val_score(pipe_lr, x_train,
                                y_train, scoring="precision", cv=10))
        cross_auc = np.mean(cross_val_score(pipe_lr, x_train,
                                y_train, scoring="roc_auc", cv=10))
        # roc_auc = np.mean(cross_val_score(clf, x_train,
        #                         y_train, scoring="roc_auc", cv=5))
        f1 = np.mean(cross_val_score(pipe_lr, x_train,
                                y_train, scoring="f1", cv=10))
        print("cross_validation accuracy:%f recall: %f, precision: %f, f1: %f, roc_auc : %f" %(cross_accuracy, cross_recall, cross_precision, f1, cross_auc))
       
        # test
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
        # print(y_test)
        pay_count = np.sum(y_test)
        pipe_lr.fit(x_train, y_train)
        # print(pipe_lr.named_steps['feature_selection'].get_support(indices=True))
        y_predict = pipe_lr.predict(x_test)
        # print(y_predict)
        # print(pipe_lr.named_steps['clf'].theta_)
        # print(pipe_lr.named_steps['clf'].sigma_)
        # print(pay_count)
        positive_true = 0
        negative_true = 0
        for i in range(len(y_test)):
            if y_test[i] == 1 and y_predict[i] == 1:
                positive_true += 1
            if y_test[i] == 0 and y_predict[i] == 0:
                negative_true += 1
        recall = float(positive_true/pay_count)
        precision = positive_true/np.sum(y_predict)
        accuracy = (positive_true + negative_true)/len(y_predict)
        f1 = 2 * (precision * recall)/(precision + recall)
        
        print('Test accuracy: %.3f pay_count: %d recall: %f precision : %f f1 : %f' % (pipe_lr.score(x_test, y_test), pay_count, recall, precision, f1))

        # 画PRC曲线
        y_score = pipe_lr.predict_proba(x_test)
        from sklearn.metrics import average_precision_score
        average_precision = average_precision_score(y_test, y_score[:,1])

        print('Average precision-recall score: {0:0.2f}'.format(
              average_precision))
        
        precision, recall, _ = precision_recall_curve(y_test, y_score[:,1])
        plt.plot(recall, precision, color='b')
        # plt.step(recall, precision, color='b', alpha=0.2,
                 # where='post')
        # plt.fill_between(recall, precision, step='post', alpha=0.2,
                         # color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
          average_precision))
        plt.show()
        

        # 画ROC曲线
        probas_ = pipe_lr.predict_proba(x_test) 
        # print(pipe_lr.classes_ )

        # 通过roc_curve()函数，求出fpr和tpr，以及阈值  
        # 这里传进去的第二个参数是预测为正例的概率
        # The predicted class probability is the fraction of samples of the same class in a leaf.
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来  
        plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))
      
        #画对角线
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating curve')
        plt.legend(loc="lower right")
        path = file_util.get_figure_path("slot", "predict_purchase", name)
        plt.savefig(os.path.join(path, "ROC"))
        plt.show()

        



        # features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time", "average_login_interval", "average_spin_interval", "average_bonus_win", "average_bet", "bonus_ratio", "spin_per_active_day", "bonus_per_active_day", "locale"]
        # features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time", "average_login_interval", "average_spin_interval", "average_bonus_win"]
        features = ["average_day_active_time","average_login_interval", "average_spin_interval", "average_bonus_win", "spin_per_active_day", "bonus_per_active_day","average_bet", "bonus_ratio", "free_spin_ratio", "coin"]
        class_names = ["non-purchase", "purchase"]
        # 决策树
        if name == "DT":
            path = file_util.get_figure_path("slot", "predict_purchase", "DT")
            dot_data = tree.export_graphviz(pipe_lr.named_steps['clf'], out_file=None,
                                                                        feature_names=features,
                                                                        class_names=class_names,
                                                                        filled=True,
                                                                        rounded=True,
                                                                        impurity=False)
            graph = pydotplus.graph_from_dot_data(dot_data)
            graph.write_pdf(os.path.join(path, "tree_per_day_without_feature_selection.pdf"))

        # 随机森林
        if name == "RF":
            path = file_util.get_figure_path("slot", "predict_purchase", "RF", "tree_per_day_without_feature_selection_activeday_7")
            DTs = pipe_lr.named_steps['clf'].estimators_
            for i, dt in enumerate(DTs):
                dot_data = tree.export_graphviz(dt, out_file=None,
                                                    feature_names=features,
                                                    filled=True,
                                                    rounded=True,
                                                    class_names=class_names,
                                                    impurity=False)
                graph = pydotplus.graph_from_dot_data(dot_data) 
                graph.write_pdf(os.path.join(path, str(i) + ".pdf"))

        # adaboost
        if name == "AdaBoost":
            path = file_util.get_figure_path("slot", "predict_purchase", "AdaBoost", "tree_per_day_without_feature_selection_activeday_7")
            DTs = pipe_lr.named_steps['clf'].estimators_
            for i, dt in enumerate(DTs):
                dot_data = tree.export_graphviz(dt, out_file=None,
                                                    feature_names=features,
                                                    filled=True,
                                                    rounded=True,
                                                    class_names=class_names,
                                                    impurity=False)
                graph = pydotplus.graph_from_dot_data(dot_data)
                graph.write_pdf(os.path.join(path, str(i) + ".pdf"))

if __name__ == "__main__":
    conn = conn = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
    # features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time", "average_login_interval", "average_spin_interval", "average_bonus_win", "average_bet", "bonus_ratio", "spin_per_active_day", "bonus_per_active_day"]
    # features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time", "average_login_interval", "average_spin_interval", "average_bonus_win"]
    features = ["average_day_active_time","average_login_interval", "average_spin_interval", "average_bonus_win", "spin_per_active_day", "bonus_per_active_day","average_bet", "bonus_ratio", "free_spin_ratio", "coin"]
    locales = ["US","MY","HU","MM","RU","IT","BR","DE","GR","EG","ES","FR","PT","PL","AU","CA","ID","RO","GB","UA","CZ","NL","SG"]
    x = []
    y = []
    # sql = "select uid, level, coin, purchase_times, active_days, average_day_active_time, average_login_interval, average_spin_interval from slot_user_profile where purchase_times > 0"
    sql = "select * from slot_purchase_profile where purchase_times > 0 and active_days > 1"
    result_pay = conn.query(sql)
    pay_num = len(result_pay)
    for record in result_pay:
        d = []
        for feature in features:
            d.append(record[feature])
        
        # # locale one-hot的编码
        # locale_encode = [0] * (len(locales) + 1)
        # loc = record["locale"]
        # if loc in locales:
        #     locale_encode[locales.index(loc)] = 1
        # else:
        #     locale_encode[-1] = 1
        # d += locale_encode

        # locale 数字编码
        # loc = record["locale"]
        # if loc in locales:
        #     d.append(locales.index(loc))
        # else:
        #     d.append(-1)

        x.append(d)
        y.append(1)

    # sql = "select uid, level, coin, purchase_times, active_days, average_day_active_time, average_login_interval, average_spin_interval from slot_user_profile where purchase_times = 0"
    sql = "select * from slot_purchase_profile where purchase_times = 0 and active_days > 1"
    result_no_pay = conn.query(sql)
    result_no_pay = random.sample(result_no_pay, 5 * pay_num)
    no_pay_num = len(result_no_pay)
    for record in result_no_pay:
        d = []
        for feature in features:
            d.append(record[feature])

        # # locale的编码
        # locale_encode = [0] * (len(locales) + 1)
        # loc = record["locale"]
        # if loc in locales:
        #     locale_encode[locales.index(loc)] = 1
        # else:
        #     locale_encode[-1] = 1
        # d += locale_encode


        # locale 数字编码
        # loc = record["locale"]
        # if loc in locales:
        #     d.append(locales.index(loc))
        # else:
        #     d.append(-1)

        x.append(d)
        y.append(0)
    # scaler = preprocessing.StandardScaler()
    # x = scaler.fit_transform(np.array(x))

    train(x, y)