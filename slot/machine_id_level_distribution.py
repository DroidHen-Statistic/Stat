import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from enum import Enum, unique
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from utils import *
from functools import reduce


from twh.ready_for_train import *

@unique
class ActionType(Enum):
    LOGIN = 1
    SPIN = 2
    PLAYBONUS = 3
    PURCHASE = 4


@unique
class SpinFormat(Enum):
    DATETIME = 1
    UID = 2
    IS_FREE = 3
    MACHINE_ID = 4
    PAY_IN = 5
    WIN = 6
    COIN = 7
    WIN_BONUS = 8
    WIN_FREE_SPIN = 9
    BET = 10
    LINES = 11
    LEVEL = 12

# 2 20170423000000 1303182 0 24 75000 0 16128079 0 0


@unique
class LoginFormat(Enum):
    DATETIME = 1
    UID = 2


@unique
class PurchaseFormat(Enum):
    DATETIME = 1
    UID = 2


@unique
class PlayBonusFormat(Enum):
    DATETIME = 1
    UID = 2
    MACHINE_ID = 3
    TOTAL_WIN = 4
    COIN = 5


def read_log_from_after_read():
    log_file = os.path.join(config.log_base_dir, "after_read")
    with open(log_file, 'r') as f:
        reader = csv.reader(f, delimiter=" ")
        for line in reader:
            data_type = int(line[0])
            if data_type == 2:
                pass


def read_from_parse_result():
    """
    自定义向量读取，每次要改改这里
    """

    base_dir = os.path.join(config.log_result_dir, "slot")

    uid_2_vectors = {}
    data = [defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float))]  # 0没充值，1充值

    dir_list = os.listdir(base_dir)
    for cr_uid in dir_list:
        if cr_uid == "__waring__":
            continue
        user_data = [defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float))]
        user_dir = os.path.join(base_dir, cr_uid)
        if not os.path.isdir(user_dir):
            continue

        for payed in range(2):
            pre_fix = ""
            if payed == 1:
                pre_fix = "pay_"
            file_machine_id = os.path.join(user_dir, pre_fix + "machine_id.txt")
            file_level = os.path.join(user_dir, pre_fix + "level.txt")
            if os.path.exists(file_machine_id):
                f_machine_id = open(file_machine_id, 'r')
                f_level = open(file_level, 'r')
                while True:
                    line_machine_id = f_machine_id.readline().strip()
                    line_level = f_level.readline().strip()
                    if not line_machine_id:
                        break
                    line_machine_id = line_machine_id.split(" ")
                    line_level = line_level.split(" ")
                    if line_machine_id[-1] == "-1" or line_level[-1] == "-1":
                    # if line_machine_id[-1] == "-1":
                        continue
                    for i in range(len(line_machine_id)):
                        data[payed][line_level[i]][line_machine_id[i]] += 1
                        user_data[payed][line_level[i]][line_machine_id[i]] += 1
                f_machine_id.close()
                f_level.close()  

        drop_dir = os.path.join(base_dir, "__waring__", "drop_pay")
        payed = 0
        user_dir = os.path.join(drop_dir, cr_uid)
        file_machine_id = os.path.join(user_dir, "machine_id.txt")
        file_level = os.path.join(user_dir, "level.txt")
        if os.path.exists(file_machine_id):
            f_machine_id = open(file_machine_id, 'r')
            f_level = open(file_level, 'r')
            while True:
                line_machine_id = f_machine_id.readline().strip()
                line_level = f_level.readline().strip()
                if not line_machine_id:
                    break
                line_machine_id = line_machine_id.split(" ")
                line_level = line_level.split(" ")
                if line_machine_id[-1] == "-1" or line_level[-1] == "-1":
                    continue
                for i in range(len(line_machine_id)):
                    data[1][line_level[i]][line_machine_id[i]] += 1
                    user_data[1][line_level[i]][line_machine_id[i]] += 1
            f_machine_id.close()
            f_level.close()

        uid_2_vectors[cr_uid] = user_data

        # result = []
        # with open("wja_data.txt",'a') as f:
        #     f.write("uid: " + cr_uid + ":\n")
        #     for payed, lv_mid_data in enumerate(user_data):
        #         # f.write("payed: " + str(payed) + ":\n")
        #         # tmp_lv = sorted(lv_mid_data.items(), key = lambda x: x[0])
        #         # for lv_mid in tmp_lv:
        #         #     f.write("lv: " + str(lv_mid[0] + "\n"))
        #         #     f.write(str(list(sorted(lv_mid[1].values()))))
        #         #     f.write("\n")
        #         for lv, mid_count in lv_mid_data.items():
        #             result += list(mid_count.values())
        #     f.write(str(sorted([int(x) for x in result])))
        #     f.write("\n")



    # base_dir = os.path.join(base_dir, "__waring__", "drop_pay")
    # for cr_uid in dir_list:
    #     payed = 0
    #     user_dir = os.path.join(base_dir, cr_uid)
    #     if not os.path.isdir(user_dir):
    #         continue
    #     file_machine_id = os.path.join(user_dir, "machine_id.txt")
    #     file_level = os.path.join(user_dir, "level.txt")
    #     if os.path.exists(file_machine_id):
    #         print("origin write")
    #         f_machine_id = open(file_machine_id, 'r')
    #         f_level = open(file_level, 'r')
    #         while True:
    #             line_machine_id = f_machine_id.readline().strip()
    #             line_level = f_level.readline().strip()
    #             if not line_machine_id:
    #                 break
    #             line_machine_id = line_machine_id.split(" ")
    #             line_level = line_level.split(" ")
    #             if line_machine_id[-1] == -1 or line_level[-1] == -1:
    #                 continue
    #             for i in range(len(line_machine_id)):
    #                 data[1][line_level[i]][line_machine_id[i]] += 1
    #         f_machine_id.close()
    #         f_level.close()
    return data

def level_distribution():
    # pay和no_pay的图
    for i in range(2):
        if i == 1:
            path = file_util.get_figure_path("slot","machine_level_distribution", "pay")
            file_name = "_pay.PNG"
        else:
            path = file_util.get_figure_path("slot","machine_level_distribution", "no_pay")
            file_name = "_not_pay.PNG"
        for level, machine_ids in data[i].items():
            tmp = sorted(machine_ids.items(), key = lambda x : float(x[0]))
            tmp = list(zip(*tmp))
            ids = [float(x) for x in tmp[0]]
            count = tmp[1]
            plt.bar(ids, count, width = 1)
            plt.gca().xaxis.set_major_locator(xmajorLocator)
            plt.gca().set_xlabel("machine_id")
            plt.gca().set_ylabel("count")
            plt.gcf().set_size_inches(18.5, 10.5)
            plt.savefig(os.path.join(path, str(level) + file_name))
            plt.cla()

    keys = set(sum([list(obj.keys()) for obj in data], []))
    total = {}
    for key in keys:
        total[key] = reduce(other_util.union_dict, [obj.get(key, {}) for obj in data])

    # pay和nopay合起来的图
    path = file_util.get_figure_path("slot","machine_level_distribution", "total")
    for level, machine_ids in total.items():
        tmp = sorted(machine_ids.items(), key = lambda x : float(x[0]))
        tmp = list(zip(*tmp))
        ids = [float(x) for x in tmp[0]]
        count = tmp[1]
        plt.bar(ids, count, width = 1)
        plt.gca().xaxis.set_major_locator(xmajorLocator)
        plt.gca().set_xlabel("machine_id")
        plt.gca().set_ylabel("count")
        plt.gcf().set_size_inches(18.5, 10.5)
        plt.savefig(os.path.join(path, str(level) + "_total.PNG"))
        plt.cla()
    #     
    # fig = plt.figure()
    # levels = sorted(list(total.keys()), key = lambda x: float(x))
    # levels = np.arange(float(levels[-1]))
    # ids = sorted(set(sum([list(obj.keys()) for obj in total.values()], [])), key = lambda x : float(x))
    # ids = np.arange(float(ids[-1]))
    # levels, ids = np.meshgrid(levels, ids)
    # machine_ids = set(sum([list(obj.keys()) for obj in data], []))
    # count = np.zeros((len(levels), len(ids)))
    # for i in range(levels.shape[1]):
    #     for j in range(ids.shape[0]):
    #         count[i][j] = total.get(i,{}).get(j,0)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # print(len(levels))
    # print(len(ids))
    # print(count.shape)
    # surf = ax.plot_surface(levels, ids, count, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # plt.show()

    # for level, machine_ids in total.items():

    #     tmp = sorted(machine_ids.items(), key = lambda x : float(x[0]))
    #     tmp = list(zip(*tmp))
    #     ids = [float(x) for x in tmp[0]]
    #     count = tmp[1]
        
    #     ax = fig.gca(projection='3d')
    #     ax.bar(ids, count, zs=float(level), zdir='y', alpha=0.8)
    #     plt.gca().set_xlabel("machine_id")
    #     plt.gca().set_ylabel("level")
    #     plt.gcf().set_size_inches(18.5, 10.5)
    #     # plt.savefig(os.path.join(path, str(level) + "pay.PNG"))
    # plt.show()



    # 所有level合起来的图
    path = file_util.get_figure_path("slot","machine_level_distribution")
    count_dict = list(total.values())
    total_count = other_util.union_dict(*count_dict)
    ids = [float(x) for x in list(total_count.keys())]
    count = list(total_count.values())
    plt.bar(ids, count, width = 1)
    plt.gca().xaxis.set_major_locator(xmajorLocator)
    plt.gca().set_xlabel("machine_id")
    plt.gca().set_ylabel("count")
    plt.gcf().set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(path, "total_count.PNG"))
    plt.cla()

def level_range_distribution():
        # for i in range(2):
    #     if i == 1:
    #         path = file_util.get_figure_path("slot","machine_level_section_distribution", "pay")
    #         file_name = "_pay.PNG"
    #     else:
    #         path = file_util.get_figure_path("slot","machine_level_section_distribution", "no_pay")
    #         file_name = "_not_pay.PNG"
    #     data_sorted = sorted(data[i].items(), key = lambda x:float(x[0]))
    #     unlock_index = 1
    #     total_section = {}
    #     for level, machine_ids in data_sorted:
    #         if (not unlock_index >= len(unlock_level)) and float(level) >= unlock_level[unlock_index]:
    #             tmp = sorted(total_section.items(), key = lambda x : float(x[0]))
    #             tmp = list(zip(*tmp))
    #             ids = [float(x) for x in tmp[0]]
    #             count = tmp[1]
    #             plt.bar(ids, count, width = 1)
    #             plt.gca().xaxis.set_major_locator(xmajorLocator)
    #             plt.gca().set_xlabel("machine_id")
    #             plt.gca().set_ylabel("count")
    #             plt.gcf().set_size_inches(18.5, 10.5)
    #             plt.savefig(os.path.join(path, str(unlock_level[unlock_index - 1]) + "_" + str(int(float(level)) - 1) + file_name))
    #             plt.cla()
    #             total_section = {}
    #             unlock_index += 1
    #         total_section = other_util.union_dict(total_section, machine_ids)
    #     tmp = sorted(total_section.items(), key = lambda x : float(x[0]))
    #     tmp = list(zip(*tmp))
    #     ids = [float(x) for x in tmp[0]]
    #     count = tmp[1]
    #     plt.bar(ids, count, width = 1)
    #     plt.gca().xaxis.set_major_locator(xmajorLocator)
    #     plt.gca().set_xlabel("machine_id")
    #     plt.gca().set_ylabel("count")
    #     plt.gcf().set_size_inches(18.5, 10.5)
    #     plt.savefig(os.path.join(path, str(unlock_level[-1]) + "_" + "200" + file_name))
    #     plt.cla()

    keys = set(sum([list(obj.keys()) for obj in data], []))
    total = defaultdict(lambda: defaultdict(float))
    for key in keys:
        total[key] = reduce(other_util.union_dict, [obj.get(key, defaultdict(float)) for obj in data])
    # path = file_util.get_figure_path("slot","machine_level_section_distribution", "total")   
    # data_sorted = sorted(total.items(), key = lambda x:float(x[0]))
    # unlock_index = 1
    # total_section = {}
    # for level, machine_ids in data_sorted:
    #     if (not unlock_index >= len(unlock_level)) and float(level) >= unlock_level[unlock_index]:
    #         tmp = sorted(total_section.items(), key = lambda x : float(x[0]))
    #         tmp = list(zip(*tmp))
    #         ids = [float(x) for x in tmp[0]]
    #         count = tmp[1]
    #         plt.bar(ids, count, width = 1)
    #         plt.gca().xaxis.set_major_locator(xmajorLocator)
    #         plt.gca().set_xlabel("machine_id")
    #         plt.gca().set_ylabel("count")
    #         plt.gcf().set_size_inches(18.5, 10.5)
    #         plt.savefig(os.path.join(path, str(unlock_level[unlock_index - 1]) + "_" + str(int(float(level)) - 1) + "total"))
    #         plt.cla()
    #         total_section = {}
    #         unlock_index += 1
    #     total_section = other_util.union_dict(total_section, machine_ids)
    # tmp = sorted(total_section.items(), key = lambda x : float(x[0]))
    # tmp = list(zip(*tmp))
    # ids = [float(x) for x in tmp[0]]
    # count = tmp[1]
    # plt.bar(ids, count, width = 1)
    # plt.gca().xaxis.set_major_locator(xmajorLocator)
    # plt.gca().set_xlabel("machine_id")
    # plt.gca().set_ylabel("count")
    # plt.gcf().set_size_inches(18.5, 10.5)
    # plt.savefig(os.path.join(path, str(unlock_level[-1]) + "_" + "200" + "total"))
    # plt.cla()

    # # 所有level合起来的图
    # path = file_util.get_figure_path("slot","machine_level_section_distribution")
    # count_dict = list(total.values())
    # total_count = other_util.union_dict(*count_dict)
    # ids = [float(x) for x in list(total_count.keys())]
    # count = list(total_count.values())
    # plt.bar(ids, count, width = 1)
    # plt.gca().xaxis.set_major_locator(xmajorLocator)
    # plt.gca().set_xlabel("machine_id")
    # plt.gca().set_ylabel("count")
    # plt.gcf().set_size_inches(18.5, 10.5)
    # plt.savefig(os.path.join(path, "total_count.PNG"))
    # plt.cla()

if __name__ == '__main__':
    unlock_level = [0,1,2,4,5,6,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55,58,61,64,67,70,73,76,79,82,85,88,91]
    unlock_id = [0,1,4,5,6,7,11,8,10,9,2,3,12,13,17,14,16,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
    id_to_level = {}
    # print(len(unlock_level))
    # print(len(unlock_id))
    for i in range(len(unlock_level)):
        id_to_level[str(float(unlock_id[i]))] = i
    id_to_level['-1'] = 0
    data = read_from_parse_result()
    
    xmajorLocator = MultipleLocator(1) #将x轴刻度标签设置为0.5的倍数

    keys = set(sum([list(obj.keys()) for obj in data], []))
    total = defaultdict(lambda: defaultdict(float))
    for key in keys:
        total[key] = reduce(other_util.union_dict, [obj.get(key, defaultdict(float)) for obj in data])

    result = []


    total_sorted = sorted(total.items(), key = lambda x : float(x[0]))
    level_count_total = {}
    level_range_mid_count = {}
    
    mids = set()
    unlock_index = 1
    sum_count = 0
    level_range_mid_count_tmp = {}
    for level,  machine_ids in total_sorted:
        mids = mids | set(machine_ids.keys())
        if (not unlock_index >= len(unlock_level)) and float(level) >= unlock_level[unlock_index]:
            level_count_total[unlock_level[unlock_index - 1]] = sum_count
            level_range_mid_count[unlock_level[unlock_index - 1]] = level_range_mid_count_tmp
            sum_count = 0
            unlock_index += 1
            level_range_mid_count_tmp = {}
        sum_count += sum(machine_ids.values(),0)
        level_range_mid_count_tmp = other_util.union_dict(level_range_mid_count_tmp, machine_ids)
    level_count_total[unlock_level[-1]] = sum_count
    level_range_mid_count[unlock_level[-1]] = level_range_mid_count_tmp #这里存的是等级分组后的各个mid的count值
    # print(level_range_mid_count)
    # 
    with open("wja_data_total.txt", 'w') as f:
        for level, mid_count in level_range_mid_count.items():
            result += [int(c) for c in list(mid_count.values())]
        f.write(str(sorted(result)))
        f.write("\n")
    exit()

    path = file_util.get_figure_path("slot","machine_level_ratio","all_level")
    mids = sorted(mids, key = lambda x : float(x))
    levels = sorted(list(total.keys()), key = lambda x : float(x))
    tmp = 0
    for mid in mids:
        y = []
        for level in unlock_level:
            y.append(level_range_mid_count.get(level, {}).get(mid,0)/level_count_total[level])
        tmp += y[-1]
        inr = 1 / np.arange(1,len(unlock_level))
        plt.plot(range(len(unlock_level)), y)
        plt.plot(range(1,len(unlock_level)),inr)
        plt.bar(id_to_level[mid], max(y) + 0.1, width = 0.05, color = 'k')
        plt.ylim(0, max(y) + 0.1)
        plt.xticks(range(len(unlock_level)), unlock_level)
        # plt.xlim(id_to_level[mid], len(unlock_level) - 1)
        # plt.plot(inr)
        # plt.gca().xaxis.set_major_locator(xmajorLocator)
        plt.gca().set_xlabel("level")
        plt.gca().set_ylabel("ratio")
        plt.gcf().set_size_inches(18.5, 10.5)
        plt.savefig(os.path.join(path, str(mid) + ".PNG"))
        # plt.show()
        plt.cla()