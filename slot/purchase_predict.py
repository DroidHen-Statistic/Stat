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
import random
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve

import pydotplus

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB

def train(x, y):
    # x = scaler.fit_transform(x)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    # print(y_test)

    pay_count = np.sum(y)
    print("pay_count: %d" %pay_count)
    print(len(x))

    x_train = x
    y_train = y

    clfs = {"DT":tree.DecisionTreeClassifier(max_depth = 5, class_weight = "balanced"),
        "RF":RandomForestClassifier(max_depth = 5),"ExtraTrees":ExtraTreesClassifier(),"AdaBoost":AdaBoostClassifier(),"GBDT":GradientBoostingClassifier(),"Bayes":GaussianNB()}
    # clfs = {"AdaBoost":AdaBoostClassifier(),"GBDT":GradientBoostingClassifier()}
    # clfs = {"DT":tree.DecisionTreeClassifier(max_depth = 5)} 
    for name, clf in clfs.items():
        print("------%s-------" % name)
        pipe_lr = Pipeline([('clf', clf)])

        cross_accuracy = np.mean(cross_val_score(clf, x_train,
                                y_train, scoring="accuracy", cv=5))
        cross_recall = np.mean(cross_val_score(clf, x_train,
                                y_train, scoring="recall", cv=5))
        cross_precision = np.mean(cross_val_score(clf, x_train,
                                y_train, scoring="precision", cv=5))
        # roc_auc = np.mean(cross_val_score(clf, x_train,
        #                         y_train, scoring="roc_auc", cv=5))
        f1 = np.mean(cross_val_score(clf, x_train,
                                y_train, scoring="f1", cv=5))
        print("cross_validation accuracy:%f recall: %f, precision: %f, f1: %f" %(cross_accuracy, cross_recall, cross_precision, f1))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
        # print(y_test)
        pay_count = np.sum(y_test)
        pipe_lr.fit(x_train, y_train)
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
        
        print('Test accuracy: %.3f pay_count: %d recall: %f precision : %f' % (pipe_lr.score(x_test, y_test), pay_count, recall, precision))

        # y_score = pipe_lr.decision_function(x_test)
        # from sklearn.metrics import average_precision_score
        # average_precision = average_precision_score(y_test, y_score)

        # print('Average precision-recall score: {0:0.2f}'.format(
        #       average_precision))
        
        # precision, recall, _ = precision_recall_curve(y_test, y_score)

        # plt.step(recall, precision, color='b', alpha=0.2,
        #          where='post')
        # plt.fill_between(recall, precision, step='post', alpha=0.2,
        #                  color='b')

        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        # plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
        #   average_precision))
        # plt.show()


if __name__ == "__main__":
    conn = conn = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
    features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time", "average_login_interval", "average_spin_interval", "average_bonus_win"]
    x = []
    y = []
    # sql = "select uid, level, coin, purchase_times, active_days, average_day_active_time, average_login_interval, average_spin_interval from slot_user_profile where purchase_times > 0"
    sql = "select * from slot_user_profile where purchase_times > 0"
    result_pay = conn.query(sql)
    pay_num = len(result_pay)
    for record in result_pay:
        d = []
        for feature in features:
            d.append(record[feature])
        x.append(d)
        y.append(1)
    # sql = "select uid, level, coin, purchase_times, active_days, average_day_active_time, average_login_interval, average_spin_interval from slot_user_profile where purchase_times = 0"
    sql = "select * from slot_user_profile where purchase_times = 0"
    result_no_pay = conn.query(sql)
    result_no_pay = random.sample(result_no_pay, 5*pay_num)
    no_pay_num = len(result_no_pay)
    for record in result_no_pay:
        d = []
        for feature in features:
            d.append(record[feature])
        x.append(d)
        y.append(0)
    scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(np.array(x))

    train(x, y)