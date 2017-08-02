import os
import sys
import numpy as np
from collections import defaultdict
from enum import Enum, unique
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


def calc_len_times(seq_len, max_len):
    """
    计算转盘平均时长，区分pay和nopay
    """
    base_dir = os.path.join(config.log_base_dir, "result")
    # base_dir = r"E:\codes\GitHubs\slot\result"
    dir_list = os.listdir(base_dir)
    uid_count = [0, 0]
    total_time = [0, 0]
    for cr_uid in dir_list:
        user_dir = os.path.join(base_dir, cr_uid)
        if not os.path.isdir(user_dir):
            continue

        cr_time = _do_calc_len_times(user_dir, seq_len, max_len)
        if len(cr_time) < 2:
            continue
        for payed in range(2):
            if cr_time[payed] > 0:
                total_time[payed] += cr_time[payed]  # 没付费
                uid_count[payed] += 1  # 付费

    mean_time = [-1, -1]
    for payed in range(2):
        if total_time[payed] > 0:
            mean_time[payed] = total_time[payed] / uid_count[payed]
    return mean_time


def _do_calc_len_times(file_dir, seq_len, max_len):
    """
    一个玩家转盘的平均时长，分pay和no pay
    max_len : 原始文件的最大序列长度，目前是10
    """
    # 充值记录
    pay_file = os.path.join(file_dir, "pay_odds.txt")
    # if (not os.path.exists(pay_file)):
    #     return []
    mean_time = [0, 0]
    count = [0, 0]
    for payed in range(2):
        pre_fix = ""
        if payed:
            pre_fix = 'pay_'

        file_time = os.path.join(file_dir, pre_fix + "time_delta.txt")
        if (not os.path.exists(file_time)):
            mean_time[payed] = -1
            continue

        f_time = open(file_time, 'r')
        while True:
            line = f_time.readline().strip()
            line = line.split(" ")
            if len(line) < seq_len:
                break
            line = list(map(float, line[1::]))
            mean_time[payed] += np.mean(line)
            count[payed] += 1
        f_time.close()

    for payed in range(2):
        if mean_time[payed] > 0:
            mean_time[payed] = mean_time[payed] / count[payed]

    # return [data, lable]
    return mean_time

from scipy import stats


def process_data(data):
    desc_no_pay = stats.describe(data)

    stat_info = np.zeros(6, dtype=float)

    stat_info[0] = desc_no_pay.mean
    stat_info[1] = np.sqrt(desc_no_pay.variance)  # 量纲平方
    # stat_info[2] = desc_no_pay.skewness * desc_no_pay.mean  # 无量纲
    # stat_info[3] = desc_no_pay.kurtosis * desc_no_pay.mean # 无量纲
    stat_info[2] = desc_no_pay.skewness  # 无量纲
    stat_info[3] = desc_no_pay.kurtosis # 无量纲
    stat_info[4] = desc_no_pay.minmax[0]
    stat_info[5] = desc_no_pay.minmax[1]

    # stat_info.append(desc_no_pay.mean )
    # stat_info.append(desc_no_pay.variance ) # 量纲平方
    # stat_info.append(desc_no_pay.skewness )  # 无量纲
    # stat_info.append(desc_no_pay.kurtosis ) # 无量纲
    # stat_info.append(desc_no_pay.minmax[0] )
    # stat_info.append(desc_no_pay.minmax[1] )

    return stat_info

def read_user_data_custom(file_dir, seq_len, max_len):
    """
    自定义向量读取，每次要改改这里
    """
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
        if file_type == 1:
            pay_coins = [x[0] for x in data[1]]
            coin_quartile = np.percentile(pay_coins, 75)
            for i,x in enumerate(data[1]):
                if(x[0] > coin_quartile):
                    data[1].pop(i)
                    pay_count -= 1
        file_odds = os.path.join(file_dir, pre_fix + "odds.txt")
        f_odds = open(file_odds, 'r')

        file_coin = os.path.join(file_dir, pre_fix + "coin.txt")
        f_coin = open(file_coin, 'r')

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
            odds_stat = process_data(odds)

            line = f_coin.readline()
            if not line:
                break
            # line = line.relace("\n",'')
            line = line.strip().split(" ")
            if len(line) < seq_len or line[-1] == "-1": # 被抛掉的数据
                continue
            coin = float(line[-1])
            if file_type == 1 and coin > coin_quartile:
                continue
            origin_coin = float(line[0])
            earn = float(line[-1]) - float(line[0])
            ratio = earn / origin_coin if origin_coin else earn

            cr_data = np.hstack((coin, earn, ratio, odds_stat))

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

# 读玩家数据
def read_user_data(file_dir, seq_len, max_len):
    """
    max_len : 原始文件的最大序列长度，目前是10
    """
    # 只读有充值记录的
    pay_file = os.path.join(file_dir, "pay_odds.txt")
    if (not os.path.exists(pay_file)):
        return []
    data = [[], []]  # 0没充值，1充值
    lable = [[], []]

    pay_count = 0
    no_pay_count = 0
    for file_type in range(2):
        pre_fix = ""
        payed = 0
        if file_type == 0:
            pre_fix = 'pay_'
            payed = 1
        file_odds = os.path.join(file_dir, pre_fix + "odds.txt")
        f_odds = open(file_odds, 'r')

        # file_coin = os.path.join(file_dir, pre_fix + "coin.txt")
        # f_coin = open(file_coin, 'r')

        # file_bonus = os.path.join(file_dir, pre_fix + "win_bonus.txt")
        # f_bonus = open(file_bonus, 'r')

        # file_time = os.path.join(file_dir, pre_fix + "time_delta.txt")
        # f_time = open(file_time, 'r')

        while True:
            line = f_odds.readline().strip()
            # cr_data = np.zeros(3 + seq_len)
            line = line.split(" ")
            if len(line) < seq_len or line[-1] == "-1": # 被抛掉的数据
                break
            cr_data = np.zeros(seq_len)
            for i in range(seq_len):
                # cr_data[3 + i] = float(line[max_len - seq_len + i])
                cr_data[i] = float(line[max_len - seq_len + i])

            # line = f_coin.readline()
            # if not line:
            #     break
            # # line = line.relace("\n",'')
            # line = line.strip()

            # cr_data = [0] * (3 + seq_len)
            # line = line.split(" ")
            # cr_data[0] = float(line[-1])

            # line = f_bonus.readline().strip()
            # line = list(map(float, line.split(" ")[max_len - seq_len::]))
            # cr_data[1] = float(np.sum(line[::-1]))

            # line = f_time.readline().strip()
            # line = list(map(float, line.split(" ")[max_len - seq_len::]))
            # cr_data[2] = float(np.sum(line))
            # data[].append(cr_data)
            # lable.append(file_type)
            # data[payed].append(cr_data)
            data[payed].append(process_data(cr_data))
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
    return data


def gen_uid_vector(seq_len, max_len):
    base_dir = os.path.join(config.log_base_dir, "result")
    # base_dir = r"E:\codes\GitHubs\slot\result"

    uid_2_vectors = {}

    dir_list = os.listdir(base_dir)
    for cr_uid in dir_list:
        user_dir = os.path.join(base_dir, cr_uid)
        if not os.path.isdir(user_dir):
            continue
        ret = read_user_data_custom(user_dir, seq_len, max_len)
        if len(ret) > 0:
            uid_2_vectors[cr_uid] = ret
    return uid_2_vectors
# exit()

if __name__ == '__main__':

    from scipy import stats
    # from scipy.stat import skew
    file_names = ['coin', 'is_free', 'level', 'odds',
                  'time_delta', 'win_bonus', 'win_free']
    seq_len = 10
    max_len = 50

    # mean_time = calc_len_times(seq_len, max_len)
    # exit()
    uid_2_vectors = gen_uid_vector(seq_len, max_len)
    # print(len(uid_2_vectors))
    # print(uid_2_vectors)
    stat_uid = {}
    for uid, vectors in uid_2_vectors.items():
        no_pay = np.array(vectors[0])
        pay = np.array(vectors[1])

        types = ["均值", "方差", "偏度", "峰度", "最小", "最大"]

        no_pay_stat = []
        pay_stat = []
        for i in range(len(types)):
            no_pay_stat.append(np.mean(no_pay[:, i]))
            pay_stat.append(np.mean(pay[:, i]))

        from matplotlib import pyplot as plt
        plt.xlabel = "stat type"
        plt.ylabel = "stat value"
        types = ["均值", "方差", "偏度", "峰度", "最小", "最大"]
        plt.plot(no_pay_stat, label="no pay")
        plt.plot(pay_stat, label="pay")

        plt.legend()
        plt.show()

        print(pay)

    from matplotlib import pyplot as plt
    for uid, cr_data in stat_uid.items():
        no_pay = cr_data[0]
        pay = cr_data[1]

        plt.xlabel = "stat type"
        plt.ylabel = "stat value"
        types = ["均值", "方差", "偏度", "峰度", "最小", "最大"]
        plt.plot(no_pay, types, label="no pay")
        plt.plot(pay, types, label="pay")

        plt.legend()
        plt.show()

        # stat_pay.append(np.mean(pay))
        # stat_pay.append(np.var(pay))
        # stat_pay.append(skew(pay))


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
