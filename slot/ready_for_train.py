import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import defaultdict
from enum import Enum, unique
from utils import *
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn import tree

import pydotplus
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

head_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
# print(head_path)
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import config
# PreVectorFormat = Enum('PreVectorFormat', 'last_coin bonus_coin win_free ')
# a = PreVectorFormat.last_coin.value
# print(a)
# @unique
# class VectorFormat(Enum):
#   Win_Bonus,
# 格式： last_coin bonus_coin total_time odds(seq_len)
# 读玩家数据

def read_user_data(file_dir, seq_len, max_len):
    """
    max_len : 原始文件的最大序列长度，目前是10
    """
    # 只读有充值记录的
    pay_file = os.path.join(file_dir, "pay_odds.txt")
    if (not os.path.exists(pay_file)):
        return []


    data = []
    label = []
    for file_type in range(2):
        pre_fix = ""
        if file_type == 1:
            pre_fix = 'pay_'
        file_odds = os.path.join(file_dir, pre_fix + "odds.txt")
        f_odds = open(file_odds, 'r')

        file_coin = os.path.join(file_dir, pre_fix + "coin.txt")
        f_coin = open(file_coin, 'r')

        file_bonus = os.path.join(file_dir, pre_fix + "win_bonus.txt")
        f_bonus = open(file_bonus, 'r')

        file_time = os.path.join(file_dir, pre_fix + "time_delta.txt")
        f_time = open(file_time, 'r')
        while True:
            line = f_coin.readline()
            if not line:
                break
            # line = line.relace("\n",'')
            line = line.strip()
            if len(line) < seq_len:
                break
            # cr_data = np.zeros(3 + seq_len)
            cr_data = [0] * (5 + seq_len)

            # 前后的金币差
            line = line.split(" ")
            cr_data[0] = float(line[1]) - float(line[-1])
            # if file_type == 0 and float(line[-1]) > 100000:
            #     continue
            # cr_data[1] = float(line[-1])

            line = f_bonus.readline().strip()
            line = list(map(float, line.split(" ")[max_len - seq_len::]))
            cr_data[4] = float(np.sum(line[::-1]))

            # line = f_time.readline().strip()
            # line = list(map(float, line.split(" ")[max_len - seq_len::]))
            # cr_data[2] = float(np.sum(line))

            line = f_odds.readline().strip()
            line = list(map(float, line.split(" ")))

            odds_std = np.std(line)
            cr_data[2] = odds_std
            cr_data[3] = np.average(line)
            for i in range(seq_len):
                cr_data[5 + i] = float(line[max_len - seq_len + i])
                

            cr_data = np.hstack((cr_data[0],cr_data[2:]))
            print(cr_data)
            data.append(cr_data)
            label.append(file_type)
        
        # print(data[0:2])
        f_bonus.close()
        f_coin.close()
        f_odds.close()
        f_time.close()


    pay_count = np.sum(label)
    if pay_count < 10:
        return []
    if(pay_count / len(label) < 0.09):
        data = data[-pay_count * 11:]
        label = label[-pay_count * 11:]
    return [data, label]

def gen_uid_vector(seq_len, max_len):
    base_dir = os.path.join(config.log_base_dir, "result")
    # base_dir = r"E:\codes\GitHubs\slot\result"

    uid_2_vectors = {}

    dir_list = os.listdir(base_dir)
    for cr_uid in dir_list:
        user_dir = os.path.join(base_dir, cr_uid)
        if not os.path.isdir(user_dir):
            continue
        ret = read_user_data(user_dir, seq_len, max_len)
        if len(ret) > 0:
            uid_2_vectors[cr_uid] = ret
    return uid_2_vectors
# exit()

def train_plot(x, y, seq_len, uid):
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)
    
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    # print(y_test)
    pay_count = np.sum(y)
    x_train = x_test = x
    y_train = y_test = y
    # print(len(x))
    # print([x_[1] for x_ in x[-pay_count:]])
    # print(np.mean([x_[1] for x_ in x[-pay_count:]]))
    # plt.plot([x_[1] for x_ in x[-pay_count:]],'b-o')
    # plt.show()
    # plt.cla()

    path = file_util.get_figure_path("slot",str(uid))
    plt.figure(1)
    for i in range(seq_len):
        plt.plot([x_[-seq_len + i] for x_ in x[-pay_count:]],'b-o')
        plt.savefig(os.path.join(path, "odds"+str(i)))
        plt.cla()

    for i in range(5):
        for j in range(5):
            plt.plot(x[-31 + i*5 +j][-seq_len:], '-o')
        plt.savefig(os.path.join(path, "odds_seq_" + str(i)))
        plt.cla()

    count = 0
    i = 0
    for seq in x[:-pay_count]:
        count += 1
        if count > 5:
            count = 1
            i += 1
            plt.savefig(os.path.join(path, "odds_seq_no_pay_"+str(i)))
            plt.cla()
        plt.plot(seq[-seq_len:],'-o')
    plt.savefig(os.path.join(path, "odds_seq_no_pay_"+str(i+1)))

    # for seq in x[-31:]:
    #     plt.plot(seq[-seq_len:], '-o')
    # plt.savefig(os.path.join(path, "odds_seq"))
    # plt.show()

    # tmp = [x_[0] for x_ in x[-31:]]
    # print(np.mean(tmp))
    
    odds_pay = np.mean(np.array(x[-31:])[:,-seq_len:], axis = 0)
    print(odds_pay)
    odds_no_pay = np.mean(np.array(x[:-31])[:,-seq_len:], axis = 0)
    print(odds_no_pay)

    odds_pay_var = np.var(np.array(x[-31:])[:,-seq_len:], axis = 0)
    print(odds_pay_var)
    odds_no_pay_var = np.var(np.array(x[:-31])[:,-seq_len:], axis = 0)
    print(odds_no_pay_var)

    plt.figure(2)
    plt.plot(odds_pay, '-o',label = 'pay')
    plt.plot(odds_no_pay,'-o',label = 'no_pay')
    plt.legend(loc = "upper right")
    plt.savefig(os.path.join(path, "odds_mean"))

    plt.figure(3)
    plt.plot(odds_pay_var, '-o', label = 'pay')
    plt.plot(odds_no_pay_var,'-o', label = 'no_pay')
    plt.legend(loc = "upper right")
    plt.savefig(os.path.join(path, "odds_var"))

    clfs = {"DT":tree.DecisionTreeClassifier(max_depth = 5),
        "RF":RandomForestClassifier(max_depth = 5),"ExtraTrees":ExtraTreesClassifier(),"AdaBoost":AdaBoostClassifier(),"GBDT":GradientBoostingClassifier(),"Bayes":GaussianNB()}
    for name, clf in clfs.items():
        print("------%s-------" % name)
        pipe_lr = Pipeline([('clf', clf)])

        # recall = np.mean(cross_val_score(clf, x_train,
        #                         y_train, scoring="recall", cv=5))
        # precision = np.mean(cross_val_score(clf, x_train,
        #                         y_train, scoring="precision", cv=5))
        # roc_auc = np.mean(cross_val_score(clf, x_train,
        #                         y_train, scoring="roc_auc", cv=5))
        # f1 = np.mean(cross_val_score(clf, x_train,
        #                         y_train, scoring="f1", cv=5))
        # print(recall, precision)
        pipe_lr.fit(x_train, y_train)
        y_predict = pipe_lr.predict(x_test)
        pay_count = np.sum(y_test)
        # print(pipe_lr.named_steps['clf'].theta_)
        # print(pipe_lr.named_steps['clf'].sigma_)
        # print(pay_count)
        recall = 0
        for i in range(len(y_test)):
            if y_test[i] == 1 and y_predict[i] == 1:
                recall += 1
        print('Test accuracy: %.3f pay_count: %d recall: %d' % (pipe_lr.score(x_test, y_test), pay_count, recall))

   
    # clf = RandomForestClassifier(max_depth = 5)
    # clf = clf.fit(x_train, y_train)
    # y_predict = clf.predict(x_test)
    # recall = 0
    # for i in range(len(y_test)):
    #     if y_test[i] == 1 and y_predict[i] == 1:
    #         recall += 1

    # print(y_predict)
    # print("recall: %f" %(recall/np.sum(y_test)))
    # print("positive ratio: ", np.sum(y_train)/len(y_train))
    # print("percision: ", clf.score(x_test, y_test))
    # # print(clf.predict(x_test))
    # # print(y_test)
    # # 
    # path = file_util.get_figure_path("slot")
    # with open(os.path.join(path,"test.dot"), 'w') as f:
    #     f = tree.export_graphviz(clf, out_file=f)
    # dot_data = tree.export_graphviz(clf, out_file=None)
    # graph = pydotplus.graph_from_dot_data(dot_data) 
    # graph.write_pdf(os.path.join(path,"test.pdf"))

def train(x, y):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    # print(y_test)
    x_train = x_test = x
    y_train = y_test = y
    # tmp = [x_[0] for x_ in x[-31:]]
    # print(np.mean(tmp))
    

    # clfs = [tree.DecisionTreeClassifier(max_depth = 5),
        # RandomForestClassifier(max_depth = 5),ExtraTreesClassifier(),AdaBoostClassifier(),GradientBoostingClassifier(),GaussianNB()]
    clfs = {"DT":tree.DecisionTreeClassifier(max_depth = 5),
        "RF":RandomForestClassifier(max_depth = 5),"ExtraTrees":ExtraTreesClassifier(),"AdaBoost":AdaBoostClassifier(),"GBDT":GradientBoostingClassifier(),"Bayes":GaussianNB()}
    for name, clf in clfs.items():
        print("------%s-------" % name)
        pipe_lr = Pipeline([('clf', clf)])

        recall = np.mean(cross_val_score(clf, x_train,
                                y_train, scoring="recall", cv=5))
        # precision = np.mean(cross_val_score(clf, x_train,
        #                         y_train, scoring="precision", cv=5))
        # roc_auc = np.mean(cross_val_score(clf, x_train,
        #                         y_train, scoring="roc_auc", cv=5))
        # f1 = np.mean(cross_val_score(clf, x_train,
        #                         y_train, scoring="f1", cv=5))
        # print(recall, precision, roc_auc)
        pipe_lr.fit(x_train, y_train)
        y_predict = pipe_lr.predict(x_test)
        pay_count = np.sum(y_test)
        # print(pay_count)
        recall = 0
        for i in range(len(y_test)):
            if y_test[i] == 1 and y_predict[i] == 1:
                recall += 1
        print('Test accuracy: %.3f pay_count: %d recall: %d' % (pipe_lr.score(x_test, y_test), pay_count, recall))

if __name__ == '__main__':
    file_names = ['coin', 'is_free', 'level', 'odds',
              'time_delta', 'win_bonus', 'win_free']
    seq_len = 10
    max_len = 10
    uid_2_vectors = gen_uid_vector(seq_len, max_len)
    uids = [1560678,1650303,1662611,1673914, 1674926,1675766]
    # for uid, vectors in uid_2_vectors.items():
    #     print("--------",uid,"--------------")
    # # X = np.array(sum([x[0] for x in uid_2_vectors.values()],[]))
    # # Y = np.array(sum([x[1] for x in uid_2_vectors.values()],[]))
    #     train(vectors[0], vectors[1])
    for uid in uids:
        train_plot(uid_2_vectors[str(uid)][0], uid_2_vectors[str(uid)][1], seq_len, uid)





# def dirlist(path, allfile):
#     filelist = os.listdir(path)
#     for filename in filelist:
#         filepath = os.path.join(path, filename)
#         if os.path.isdir(filepath):
#             cr_uid = filename
#             dirlist(filepath, allfile)
#         else:
#             allfile.append(filepath)
#     return allfile