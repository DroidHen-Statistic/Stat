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
    stat_info[3] = desc_no_pay.kurtosis  # 无量纲
    stat_info[4] = desc_no_pay.minmax[0]
    stat_info[5] = desc_no_pay.minmax[1]

    # stat_info.append(desc_no_pay.mean )
    # stat_info.append(desc_no_pay.variance ) # 量纲平方
    # stat_info.append(desc_no_pay.skewness )  # 无量纲
    # stat_info.append(desc_no_pay.kurtosis ) # 无量纲
    # stat_info.append(desc_no_pay.minmax[0] )
    # stat_info.append(desc_no_pay.minmax[1] )

    return stat_info


def get_one_user_odds(file_dir, odds_count_dict, total_count, max_count):
    for payed in range(2):
        pre_fix = ""
        if payed == 0:
            pre_fix = 'pay_'
        file_odds = os.path.join(file_dir, pre_fix + "odds.txt")
        file_line = os.path.join(file_dir, pre_fix + "line.txt")
        file_machine_id = os.path.join(file_dir, pre_fix + "machine_id.txt")
        if (not os.path.exists(file_odds)) or (not os.path.exists(file_machine_id)) or (not os.path.exists(file_line)):
            continue
        f_odds = open(file_odds, 'r')
        f_line = open(file_line, 'r')
        f_machine_id = open(file_machine_id, 'r')
        while True:
            odds_line = f_odds.readline().strip()
            line_line = f_line.readline().strip()
            machine_id_line = f_machine_id.readline().strip()
            if not odds_line:
                break
            # cr_data = np.zeros(3 + seq_len)
            odds_line = odds_line.split(" ")
            line_line = line_line.split(" ")
            machine_id_line = machine_id_line.split(" ")
            # if len(line) < 2:
            #     break
            if odds_line[-1] == "-1":  # 被抛掉的数据
                continue
            # print(line)
            for i in range(len(odds_line)):
                cr_odd= round(float(odds_line[i]),1)
                cr_line = float(line_line[i])
                cr_machine_id = float(machine_id_line[i])

                odds_count_dict[payed][cr_machine_id][cr_line][cr_odd] += 1
                # odds_count_dict[payed][cr_machine_id][cr_odd] += 1
                total_count+= 1
                if(total_count > max_count):
                    return 1
    return 0


def get_odds_vector(max_count=100000):
    base_dir = os.path.join(config.log_result_dir, "slot")
    # base_dir = r"E:\codes\GitHubs\slot\result"
    total_count = 0
    # odds_count_dict = [defaultdict(int), defaultdict(int)]  # 充值的，没充值的，分别是odds->count
    odds_count_dict = [defaultdict(lambda: defaultdict(lambda: defaultdict(float))), defaultdict(lambda: defaultdict(lambda: defaultdict(float)))]
    dir_list = os.listdir(base_dir)
    for cr_uid in dir_list:
        if not cr_uid.isdigit():
            continue
        user_dir = os.path.join(base_dir, cr_uid)
        if not os.path.isdir(user_dir):
            continue
        is_end = get_one_user_odds(user_dir, odds_count_dict, total_count, max_count)
        if is_end:
            print("break");
            return odds_count_dict
    return odds_count_dict


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
            if len(line) < seq_len or line[-1] == "-1":  # 被抛掉的数据
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
    base_dir = os.path.join(config.log_result_dir, "slot")
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
    import numpy as np
    from matplotlib import pyplot as plt
    from utils import file_util, other_util
    from sklearn.preprocessing import StandardScaler
    from matplotlib.ticker import MultipleLocator

    #---------------------------------------------------
    xmajorLocator   = MultipleLocator(0.5) #将x主刻度标签设置为20的倍
    ymajorLocator   = MultipleLocator(50) #将y轴主刻度标签设置为0.5的倍数

    scaler = StandardScaler()
    odds_count_dict = get_odds_vector(10000000000)
    no_pay = odds_count_dict[1]
    pay = odds_count_dict[0]

    no_pay_machine_id_2_odds = {}
    pay_machine_id_2_odds = {}
    for machine_id, machine_odds in no_pay.items():
        path = file_util.get_figure_path("slot", "odds_distribution_no_zero", str(machine_id))
        # machine_line_odds = machine_odds.values()
        for line, machine_line_odds in machine_odds.items():
            machine_line_odds_tmp = sorted(machine_line_odds.items(), key=lambda x : x[0])
            machine_line_odds_tmp = np.array(machine_line_odds_tmp)
            x = machine_line_odds_tmp[:,0]
            y = machine_line_odds_tmp[:,1]
            avg = np.sum(x * y) / np.sum(y)
            if len(x) > 1:
                x = x[1:]
                y = y[1:]
            #---------------------------------------------------
            xmajorLocator   = MultipleLocator(0.5) #将x主刻度标签设置为20的倍

            ymajorLocator   = MultipleLocator(np.max(y) / 8) #将y轴主刻度标签设置为0.5的倍数
            plt.minorticks_on()
            plt.xlabel("odds")
            plt.ylabel("count")
            plt.xlim(xmax=10,xmin=0)
            plt.gcf().set_size_inches(18.5, 10.5)
            plt.bar(x, y, width = 0.1)
            plt.bar(avg, np.max(y), width = 0.1)
            #设置主刻度标签的位置,
            plt.gca().xaxis.set_major_locator(xmajorLocator)
            plt.gca().yaxis.set_major_locator(ymajorLocator)
            # plt.gca().xaxis.set_minor_locator(xminorLocator)
            # plt.gca().yaxis.set_minor_locator(yminorLocator)
            # plt.show()
            plt.savefig(os.path.join(path, str(line)+ "_no_pay.png"))
            # plt.show()
            plt.cla()
        machine_odds_tmp = other_util.union_dict(*machine_odds.values())
        no_pay_machine_id_2_odds[machine_id] = machine_odds_tmp
        machine_odds_tmp = sorted(machine_odds_tmp.items(), key=lambda x : x[0])
        machine_odds_tmp = np.array(machine_odds_tmp)
        x = machine_odds_tmp[:,0]
        y = machine_odds_tmp[:,1]
        avg = np.sum(x * y) / np.sum(y)
        if len(x) > 1:
            x = x[1:]
            y = y[1:]

        xmajorLocator   = MultipleLocator(0.5) #将x主刻度标签设置为20的倍
        ymajorLocator   = MultipleLocator(np.max(y) / 8) #将y轴主刻度标签设置为0.5的倍数
        plt.minorticks_on()
        plt.xlabel("odds")
        plt.ylabel("count")
        plt.xlim(xmax=10,xmin=0)
        plt.gcf().set_size_inches(18.5, 10.5)
        plt.bar(x, y, width = 0.1)
        plt.bar(avg, np.max(y), width = 0.1)
        #设置主刻度标签的位置,
        plt.gca().xaxis.set_major_locator(xmajorLocator)
        plt.gca().yaxis.set_major_locator(ymajorLocator)
        # plt.gca().xaxis.set_minor_locator(xminorLocator)
        # plt.gca().yaxis.set_minor_locator(yminorLocator)
        plt.savefig(os.path.join(path, "all_no_pay"))
        # plt.show()
        plt.cla()


    for machine_id, machine_odds in pay.items():
        path = file_util.get_figure_path("slot", "odds_distribution_no_zero", str(machine_id))
        # machine_line_odds = machine_odds.values()
        for line, machine_line_odds in machine_odds.items():
            machine_line_odds_tmp = sorted(machine_line_odds.items(), key=lambda x : x[0])
            machine_line_odds_tmp = np.array(machine_line_odds_tmp)
            x = machine_line_odds_tmp[:,0]
            y = machine_line_odds_tmp[:,1]
            avg = np.sum(x * y) / np.sum(y)
            if len(x) > 1:
                x = x[1:]
                y = y[1:]
            xmajorLocator   = MultipleLocator(0.5) #将x主刻度标签设置为20的倍
            ymajorLocator   = MultipleLocator(np.max(y) / 8) #将y轴主刻度标签设置为0.5的倍数
            plt.minorticks_on()
            plt.xlabel("odds")
            plt.ylabel("count")
            plt.xlim(xmax=10,xmin=0)
            plt.gcf().set_size_inches(18.5, 10.5)
            plt.bar(x, y, width = 0.1)
            plt.bar(avg, np.max(y), width = 0.1)
            #设置主刻度标签的位置,
            plt.gca().xaxis.set_major_locator(xmajorLocator)
            plt.gca().yaxis.set_major_locator(ymajorLocator)
            # plt.gca().xaxis.set_minor_locator(xminorLocator)
            # plt.gca().yaxis.set_minor_locator(yminorLocator)
            plt.savefig(os.path.join(path, str(line) + "_pay.png"))
            # plt.show()
            plt.cla()
        machine_odds_tmp = other_util.union_dict(*machine_odds.values())
        pay_machine_id_2_odds[machine_id] = machine_odds_tmp
        machine_odds_tmp = sorted(machine_odds_tmp.items(), key=lambda x : x[0])
        machine_odds_tmp = np.array(machine_odds_tmp)
        x = machine_odds_tmp[:,0]
        y = machine_odds_tmp[:,1]
        avg = np.sum(x * y) / np.sum(y)
        if len(x) > 1:
            x = x[1:]
            y = y[1:]

        xmajorLocator   = MultipleLocator(0.5) #将x主刻度标签设置为20的倍
        ymajorLocator   = MultipleLocator(np.max(y) / 8) #将y轴主刻度标签设置为0.5的倍数
        plt.minorticks_on()
        plt.xlabel("odds")
        plt.ylabel("count")
        plt.xlim(xmax=10,xmin=0)
        plt.gcf().set_size_inches(18.5, 10.5)
        plt.bar(x, y, width = 0.1)
        plt.bar(avg, np.max(y), width = 0.1)
        #设置主刻度标签的位置,
        plt.gca().xaxis.set_major_locator(xmajorLocator)
        plt.gca().yaxis.set_major_locator(ymajorLocator)
        # plt.gca().xaxis.set_minor_locator(xminorLocator)
        # plt.gca().yaxis.set_minor_locator(yminorLocator)
        # plt.show()
        plt.savefig(os.path.join(path, "all_pay"))
        plt.cla()



    
    # all_data = sorted(all_data.items(), key=lambda x : x[0])
    # del odds_count_dict

    # # no_pay = scaler.fit_transform(no_pay)
    # # pay = scaler.fit_transform(pay)
    # # all_data = scaler.fit_transform(no_pay)
    

    # no_pay = np.array(no_pay)
    # no_pay_x = no_pay[:,0]
    # no_pay_y = no_pay[:,1]
    # no_pay_avg = np.sum(no_pay_x * no_pay_y) / np.sum(no_pay_y)
    # # no_pay_var = np.var(no_pay_x * no_pay_y)
    # print("no_pay: mean: %f" %(no_pay_avg))

    # pay = np.array(pay)
    # pay_x = pay[:,0]
    # pay_y = pay[:,1]
    # pay_avg = np.sum(pay_x * pay_y) / np.sum(pay_y)
    # # pay_var = np.var(pay_x * pay_y)
    # print("pay: mean: %f" %(pay_avg))

    # all_data = np.array(all_data)
    # all_data_x = all_data[:,0]
    # all_data_y = all_data[:,1]
    # all_avg = np.sum(all_data_x * all_data_y) / np.sum(all_data_y)
    # # all_var = np.var(all_data_x * all_data_y)
    # print("all: mean: %f" %(all_avg))


    # path = file_util.get_figure_path("slot")

    # # no_pay
    # plt.figure(1)
    # plt.minorticks_on()
    # plt.xlabel("odds")
    # plt.ylabel("count")
    # # plt.ylim(ymax=30,ymin=0)
    # plt.xlim(xmax=2,xmin=0)
    # # plt.scatter(no_pay_x[1:], no_pay_y[1:], marker='.')
    # plt.bar(no_pay_x, no_pay_y, width = 0.1)
    # plt.legend()
    # # plt.savefig(os.path.join(path, "odds_distribution_no_pay_with_zero"))
    # print("no pay zero: ", no_pay_y[0])

    # # pay
    # plt.figure(2)
    # plt.minorticks_on()
    # plt.xlabel("odds")
    # plt.ylabel("count")
    # # plt.ylim(ymax=30,ymin=0)
    # plt.xlim(xmax=2,xmin=0)
    # # plt.scatter(pay_x[1:], pay_y[1:], marker='.')
    # plt.bar(pay_x, pay_y, width = 0.1)
    # plt.legend()
    # # plt.savefig(os.path.join(path, "odds_distribution_pay_with_zero"))
    # print("pay zero: ", pay_y[0])

    # # all
    # plt.figure(3)
    # plt.minorticks_on()
    # plt.xlabel("odds")
    # plt.ylabel("count")
    # # plt.ylim(ymax=30,ymin=0)
    # plt.xlim(xmax=2,xmin=0)
    # # plt.scatter(pay_x[1:], pay_y[1:], marker='.')
    # plt.bar(all_data_x, all_data_y, width = 0.1)
    # plt.legend()
    # # plt.savefig(os.path.join(path, "odds_distribution_all_with_zero"))
    # print("all_zero: ",all_data_y[0])

    # plt.show()
    
    

    # from scipy import stats
    # # from scipy.stat import skew
    # file_names = ['coin', 'is_free', 'level', 'odds',
    #               'time_delta', 'win_bonus', 'win_free']
    # seq_len = 10
    # max_len = 50

    # # mean_time = calc_len_times(seq_len, max_len)
    # # exit()
    # uid_2_vectors = gen_uid_vector(seq_len, max_len)
    # # print(len(uid_2_vectors))
    # # print(uid_2_vectors)
    # stat_uid = {}
    # for uid, vectors in uid_2_vectors.items():
    #     no_pay = np.array(vectors[0])
    #     pay = np.array(vectors[1])

    #     types = ["均值", "方差", "偏度", "峰度", "最小", "最大"]

    #     no_pay_stat = []
    #     pay_stat = []
    #     for i in range(len(types)):
    #         no_pay_stat.append(np.mean(no_pay[:, i]))
    #         pay_stat.append(np.mean(pay[:, i]))

    #     from matplotlib import pyplot as plt
    #     plt.xlabel = "stat type"
    #     plt.ylabel = "stat value"
    #     types = ["均值", "方差", "偏度", "峰度", "最小", "最大"]
    #     plt.plot(no_pay_stat, label="no pay")
    #     plt.plot(pay_stat, label="pay")

    #     plt.legend()
    #     plt.show()

    #     print(pay)

    # from matplotlib import pyplot as plt
    # for uid, cr_data in stat_uid.items():
    #     no_pay = cr_data[0]
    #     pay = cr_data[1]

    #     plt.xlabel = "stat type"
    #     plt.ylabel = "stat value"
    #     types = ["均值", "方差", "偏度", "峰度", "最小", "最大"]
    #     plt.plot(no_pay, types, label="no pay")
    #     plt.plot(pay, types, label="pay")

    #     plt.legend()
    #     plt.show()

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
