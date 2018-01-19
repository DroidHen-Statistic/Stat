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


from MysqlConnection import MysqlConnection
import random

# from sklearn.model_selection import cross_validate

class Odds_Sequence_Reader(v_reader):

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


def cdf(data):
    data_dis = dict(zip(*np.unique(data, return_counts=True)))
    x = list(data_dis.keys())
    _y = list(data_dis.values())
    y = [0]
    l = len(data)
    for i,a in enumerate(_y):
        y.append(y[i] + a / l)
    return [x, y[1:]]

if __name__ == '__main__':
    cluster_reader = Odds_Sequence_Reader()
    from scipy import stats
    # from scipy.stat import skew
    file_names = ['coin', 'is_free', 'level', 'odds',
                  'time_delta', 'win_bonus', 'win_free']
    seq_len = 20
    max_len = 50

    # mean_time = calc_len_times(seq_len, max_len)
    # exit()
    x = []
    y = []
    uid_2_vectors = cluster_reader.gen_uid_vector(seq_len, max_len)
    coins = {}
    # scaler = Normalizer()
    # x = scaler.fit_transform(x)
    path = file_util.get_figure_path("slot", "odds_seq")
    data_pay = []
    data_not_pay = []
    for uid, vectors in uid_2_vectors.items():
        print("--------",uid,"--------------")
        data_pay += vectors[1]
        data_not_pay += vectors[0]

    # last 20 spin odds
    # from matplotlib.ticker import MultipleLocator
    # xmajorLocator = MultipleLocator(1) #将x轴刻度标签设置为1的倍数
    # plt.gca().xaxis.set_major_locator(xmajorLocator)
    # for i in range(15):
    #     plt.plot(range(1,21), data_pay[i])
    # plt.title("last 20 odds of pay")
    # plt.gca().set_ylabel("odds")
    # plt.savefig(os.path.join(path, "pay_odds_15"))
    # plt.show()

    # for i in range(15):
    #     plt.plot(range(1,21), data_not_pay[i])
    # xmajorLocator = MultipleLocator(1) #将x轴刻度标签设置为1的倍数
    # plt.gca().xaxis.set_major_locator(xmajorLocator)
    # plt.title("last 20 odds of non-pay")
    # plt.gca().set_ylabel("odds")
    # plt.savefig(os.path.join(path, "not_pay_odds_15"))
    # plt.show()

    # exit()

    # mean_pay = np.mean(data_pay, axis = 0)
    # mean_not_pay = np.mean(data_not_pay, axis = 0)
    # std_pay = np.std(data_pay, axis = 0)
    # std_not_pay = np.std(data_not_pay, axis = 0)
    # # print(mean_pay)
    # from matplotlib.ticker import MultipleLocator
    # xmajorLocator = MultipleLocator(1) #将x轴刻度标签设置为0.5的倍数
    # plt.gca().xaxis.set_major_locator(xmajorLocator)
    # plt.plot(range(1, seq_len + 1), mean_pay, label = "pay")
    # plt.plot(range(1, seq_len + 1), mean_not_pay, label = "not_pay")
    # plt.legend(loc = "upper right")
    # plt.title("mean odds of pay and non-pay user")
    # plt.gca().set_ylabel("odds")

    # plt.show()
    # plt.cla()
    # plt.gca().xaxis.set_major_locator(xmajorLocator)
    # plt.plot(range(1, seq_len + 1), std_pay, label = "pay")
    # plt.plot(range(1, seq_len + 1), std_not_pay, label = "not_pay")
    # plt.legend(loc = "upper right")
    # plt.title("std odds of pay and non-pay user")
    # plt.gca().set_ylabel("odds")
    # plt.show()

    path = file_util.get_figure_path("slot", "odds_seq")

    # mean CDF
    mean_pay = np.mean(data_pay, axis = 1)
    mean_not_pay = np.mean(data_not_pay[1: 2 * len(mean_pay)], axis = 1)
    cdf_pay = cdf(mean_pay)
    cdf_not_pay = cdf(mean_not_pay)
    plt.plot(cdf_pay[0] + [cdf_not_pay[0][-1]], cdf_pay[1] + [1], label = "pay")
    plt.plot(cdf_not_pay[0], cdf_not_pay[1], label = "not_pay")
    plt.legend(loc = "lower right")
    plt.title("CDF of the mean of last 20 odds(sample 1: 2)")
    plt.gca().set_xlabel("mean")
    plt.gca().set_ylabel("CDF")
    plt.savefig(os.path.join(path, "mean CDF sample_1_2"))
    plt.show()
    plt.cla()

    
    # std CDF
    std_pay = np.std(data_pay, axis = 1)
    std_not_pay = np.std(data_not_pay[1: 2 * len(std_pay)], axis = 1)
    cdf_pay = cdf(std_pay)
    cdf_not_pay = cdf(std_not_pay)
    plt.plot(cdf_pay[0] + [cdf_not_pay[0][-1]], cdf_pay[1] + [1], label = "pay")
    plt.plot(cdf_not_pay[0], cdf_not_pay[1], label = "not_pay")
    plt.legend(loc = "lower right")
    plt.title("CDF of the std of last 20 odds(sample 1: 2)")
    plt.gca().set_xlabel("std")
    plt.gca().set_ylabel("CDF")
    plt.savefig(os.path.join(path, "std CDF sample_1_2"))
    plt.show()
    plt.cla()

    # kurtosis CDF
    from scipy.stats import kurtosis
    kurtosis_pay = kurtosis(data_pay, axis = 1)
    kurtosis_not_pay = kurtosis(data_not_pay[1: 2 * len(kurtosis_pay)], axis = 1)
    cdf_pay = cdf(kurtosis_pay)
    cdf_not_pay = cdf(kurtosis_not_pay)
    plt.plot(cdf_pay[0] + [cdf_not_pay[0][-1]], cdf_pay[1] + [1], label = "pay")
    plt.plot(cdf_not_pay[0], cdf_not_pay[1], label = "not_pay")
    plt.legend(loc = "lower right")
    plt.title("CDF of the kurtosis of last 20 odds(sample 1: 2)")
    plt.gca().set_xlabel("kurtosis")
    plt.gca().set_ylabel("CDF")
    plt.savefig(os.path.join(path, "kurtosis CDF sample_1_2"))
    plt.show()
    plt.cla()

    #  skew CDF
    from scipy.stats import skew
    skew_pay = skew(data_pay, axis = 1)
    skew_not_pay = skew(data_not_pay[1: 2 * len(skew_pay)], axis = 1)
    cdf_pay = cdf(skew_pay)
    cdf_not_pay = cdf(skew_not_pay)
    plt.plot(cdf_pay[0] + [cdf_not_pay[0][-1]], cdf_pay[1] + [1], label = "pay")
    plt.plot(cdf_not_pay[0], cdf_not_pay[1], label = "not_pay")
    plt.legend(loc = "lower right")
    plt.title("CDF of the skew of last 20 odds(sample 1: 2)")
    plt.gca().set_xlabel("skew")
    plt.gca().set_ylabel("CDF")
    plt.savefig(os.path.join(path, "skew CDF sample_1_2"))
    plt.show()
    plt.cla()


