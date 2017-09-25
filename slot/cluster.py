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

from collections import defaultdict


from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from MysqlConnection import MysqlConnection
from sklearn import preprocessing
import random

from sklearn.model_selection import train_test_split
import pydotplus
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline


from dtw import dtw
import shutil

from scipy.stats import spearmanr

class Cluster_Data_Reader(v_reader):

    def __init__(self, input_dir='null', out_put_dir='null'):
        v_reader.__init__(self, input_dir, out_put_dir)

    def _read_user_data(self, file_dir, seq_len, max_len):
        pay_file = os.path.join(file_dir, "pay_odds.txt")
        if (not os.path.exists(pay_file)):
            return []
        data = [[], []]  # 0没充值，1充值

        pay_count = 0
        no_pay_count = 0
        for file_type in range(2):
            pre_fix = ""
            payed = 0
            if file_type == 0:
                pre_fix = 'pay_'
                payed = 1
            # if file_type == 1:
            #     pay_coins = [x[0] for x in data[1]]
            #     coin_quartile = np.percentile(pay_coins, 75)
            #     for i,x in enumerate(data[1]):
            #         if(x[0] > coin_quartile):
            #             data[1].pop(i)
            #             pay_count -= 1
            file_odds = os.path.join(file_dir, pre_fix + "odds.txt")
            f_odds = open(file_odds, 'r')

            # file_coin = os.path.join(file_dir, pre_fix + "coin.txt")
            # f_coin = open(file_coin, 'r')

            # file_bonus = os.path.join(file_dir, pre_fix + "win_bonus.txt")
            # f_bonus = open(file_bonus, 'r')

            # file_time = os.path.join(file_dir, pre_fix + "time_delta.txt")
            # f_time = open(file_time, 'r')

            while True:
                line = f_odds.readline()
                if not line:
                    break
                line = line.strip()
                # cr_data = np.zeros(3 + seq_len)
                line = line.split(" ")
                if len(line) < seq_len or line[-1] == "-1": # 被抛掉的数据
                    continue

                odds = [float(x) for x in line[-seq_len:]]
                # odds_stat = process_data(odds)

                # line = f_coin.readline()
                # if not line:
                    # break
                # line = line.relace("\n",'')
                # line = line.strip().split(" ")
                # if len(line) < seq_len or line[-1] == "-1": # 被抛掉的数据
                #     continue
                # coin = float(line[-1])
                # if file_type == 1 and coin > coin_quartile:
                #     continue
                # origin_coin = float(line[0])
                # earn = float(line[-1]) - float(line[0])
                # ratio = earn / origin_coin if origin_coin else earn

                # cr_data = np.hstack((coin, earn, ratio, odds_stat))
                cr_data = np.array(odds)

                # cr_data = [0] * (3 + seq_len)
                # line = line.split(" ")
                # cr_data[0] = float(line[-1])

                # line = f_bonus.readline().strip()
                # line = list(map(float, line.split(" ")[max_len - seq_len::]))
                # cr_data[1] = float(np.sum(line[::-1]))b

                # line = f_time.readline().strip()
                # line = list(map(float, line.split(" ")[max_len - seq_len::]))
                # cr_data[2] = float(np.sum(line))
                # data[].append(cr_data)
                # lable.append(file_type)
                # data[payed].append(cr_data)
                data[payed].append(cr_data)

                if file_type == 0:
                    pay_count += 1
                else:
                    if pay_count < 10:
                        return []
                    no_pay_count += 1
                    if no_pay_count > (pay_count * 10):
                        break
            f_odds.close()
            # f_bonus.close()
            # f_coin.close()
            # f_time.close()
        # return [data, lable]
        if pay_count < 10:
            return []
        return data


def odds_cluster_DBSCAN(X):
    dbscan = DBSCAN(eps = 0.5, min_samples = 8, metric = cal_spearman)
    y_pred = dbscan.fit_predict(X)
    print(len(dbscan.core_sample_indices_))

    return y_pred

def odds_cluster_Birch(X):
    birch = Birch(n_clusters = None) #threshold branching_factor n_clusters 
    y_pred = birch.fit_predict(X)
    return y_pred

# 这里要注意的是Kmeans不能自定义相似度度量函数,而默认使用的应该是欧式距离
def odds_cluster_KMeans(X, k = 5):
    kmeans = KMeans(n_clusters = k, random_state = 9)
    y_pred = kmeans.fit_predict(X)
    return y_pred

def user_cluster_KMeans(X, k = 5):
    kmeans = KMeans(n_clusters = k, random_state = 9)
    y_pred = kmeans.fit_predict(X)
    return y_pred

def cal_dtw(x,y):
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    dist, cost, acc, path = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord = 1))

    # plt.figure(1)
    # plt.plot(x, 'bo-')
    # plt.plot(y + 10, 'o-')
    # for i in range(len(path[0])):
    #     x_tmp = [path[0][i], path[1][i]]
    #     tmp = [x[path[0][i]], y[path[1]][i] + 10]
    #     plt.plot(x_tmp, tmp, '.--')
    print(dist)

    # plt.figure(2)
    # plt.imshow(acc.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
    # plt.plot(path[0], path[1], 'w')
    # plt.xlim((-0.5, acc.shape[0]-0.5))
    # plt.ylim((-0.5, acc.shape[1]-0.5))
    # print(path[0], path[1])
    
    # plt.show()
    
    return dist

def cal_spearman(x,y):
    ret = spearmanr(x,y)
    if ret[0]:
        return ret[0]
    else:
        return -10

# # 这里是对赔率序列的聚类
# if __name__ == '__main__':
#     cluster_reader = Cluster_Data_Reader()
#     from scipy import stats
#     # from scipy.stat import skew
#     file_names = ['coin', 'is_free', 'level', 'odds',
#                   'time_delta', 'win_bonus', 'win_free']
#     seq_len = 10
#     max_len = 50

#     # mean_time = calc_len_times(seq_len, max_len)
#     # exit()
#     x = []
#     y = []
#     uid_2_vectors = cluster_reader.gen_uid_vector(seq_len, max_len)
#     coins = {}
#     # scaler = Normalizer()
#     # x = scaler.fit_transform(x)
#     for uid, vectors in uid_2_vectors.items():
#         print("--------",uid,"--------------")
#         data_pay = vectors[1]
#         data_not_pay = vectors[0]
#         print("data_pay:", len(data_pay))
#         print("data_not_pay", len(data_not_pay))

#         X = np.array((data_pay + data_not_pay))
#         Y = np.array(([1] * len(data_pay) + [0] * len(data_not_pay)))
#         # if uid == str(1560678):
#         #     x1 = X[3]
#         #     x2 = X[4]
#         #     cal_dtw(x1,x2)

#         # X = scaler.fit_transform(X)
#         # 
#         print(X)
#         y_pred = odds_cluster_DBSCAN(X)
#         print(y_pred)
#         labels = set(y_pred)
#         path = file_util.get_figure_path("slot",str(uid),"cluster")
#         if os.path.exists(path):
#             shutil.rmtree(path)
#         for label in labels:
#             path = file_util.get_figure_path("slot",str(uid),"cluster",str(label))
#             print("label: %s" %label)
#             x_tmp = X[y_pred == label, :]
#             for i,x in enumerate(x_tmp):
#                 plt.plot(x)
#                 plt.savefig(os.path.join(path, str(i)))
#                 plt.cla()


# 这里是对玩家的聚类
if __name__ == "__main__":
    conn = conn = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
    features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time", "average_login_interval", "average_spin_interval", "average_bonus_win"]
    data = []
    # sql = "select uid, level, coin, purchase_times, active_days, average_day_active_time, average_login_interval, average_spin_interval from slot_user_profile where purchase_times > 0"
    sql = "select * from slot_user_profile where purchase_times > 0"
    result_pay = conn.query(sql)
    pay_num = len(result_pay)
    for record in result_pay:
        d = []
        for feature in features:
            d.append(record[feature])
        data.append(d)
    # sql = "select uid, level, coin, purchase_times, active_days, average_day_active_time, average_login_interval, average_spin_interval from slot_user_profile where purchase_times = 0"
    sql = "select * from slot_user_profile where purchase_times = 0"
    result_no_pay = conn.query(sql)
    # result_no_pay = random.sample(result_no_pay, 10*pay_num)
    no_pay_num = len(result_no_pay)
    for record in result_no_pay:
        d = []
        for feature in features:
            d.append(record[feature])
        data.append(d)
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(np.array(data))
    k = 5
    y = user_cluster_KMeans(data, k)
    data = np.array(data)
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,5], data[:,6], data[:,7], c = y)
    ax.set_xlabel(features[5])
    ax.set_ylabel(features[6])
    ax.set_zlabel(features[7])
    plt.show()
    for i in range(k):
        pay_k = list(y[:pay_num]).count(i)
        all_k = list(y).count(i)
        print("The %d cluster: pay : %d all : %d ratio : %f" %(i + 1, pay_k, all_k, pay_k/all_k))