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


def _gen_defaultdict():
    # return copy.copy(defaultdict(int))
    return defaultdict(int)

class Mid_Bonus_Count_Reader(v_reader):
    def __init__(self, input_dir='null', out_put_dir='null'):
        v_reader.__init__(self, input_dir, out_put_dir)

    def _do_read_user_data(self, file_machine, file_bonus, file_payin, file_line, seq_len, max_len, mid_bonus_count, mid_count, mid_bonus_win):
        """
        max_len : 原始文件的最大序列长度
        """
        # mid_lv_count = [defaultdict(_gen_defaultdict),
        #                 defaultdict(_gen_defaultdict)]
        if (not os.path.exists(file_machine)):
            return False
        f_machine = open(file_machine, 'r')
        f_bonus = open(file_bonus, 'r')
        f_payin = open(file_payin, 'r')
        f_line = open(file_line, 'r')
#            has_zero = False
        while True:
            machine_line = f_machine.readline().strip()
            if not machine_line:
                break
            bonus_line = f_bonus.readline().strip()
            payin_line = f_payin.readline().strip()
            line_line = f_line.readline().strip()

            machine_line = machine_line.split(" ")
            bonus_line = bonus_line.split(" ")
            payin_line = payin_line.split(" ")
            line_line = line_line.split(" ")
            # if len(line) < seq_len or line[-1] == "-1":  # 被抛掉的数据
            if machine_line[-1] == '-1':
                continue
            cr_len = len(machine_line)
            # for in line[::-1]
            for i in range(seq_len):
                # cr_index = max(0, cr_len - seq_len + i)
                cr_index = cr_len - seq_len + i
                if cr_index >= cr_len or cr_index < 0:
                    continue
                # print(line[cr_index])
                try:
                    cr_mid = int(float(machine_line[cr_index]))
                    cr_bonus = int(float(bonus_line[cr_index]))
                    cr_payin = int(float(payin_line[cr_index]))
                    cr_line = int(float(line_line[cr_index]))
                    mid_count[cr_mid] += 1
                    if cr_bonus != 0:
                        mid_bonus_count[cr_mid] += 1
                        cr_bet = round(float(cr_payin / cr_line),2)
                        mid_bonus_win[cr_mid] += (cr_bonus / cr_bet)

                except:
                    break
            #         # [cr_mid] += 1
            # f_machine.close()
            # f_bonus.close()
            # f_coin.close()
            # f_time.close()
        # return [data, lable]
        f_machine.close()
        f_bonus.close()
        return True

    """
    读玩家数据，继承类可以重写这个函数
    """

    def _read_user_data(self, file_dir, seq_len, max_len):
        """
        max_len : 原始文件的最大序列长度
        """

        arr_dir = file_dir.split(os.path.sep)
        uid = arr_dir[-1]
        # print(uid)
        # exit()

        mid_bonus_count = defaultdict(int)
        mid_count = defaultdict(int)
        mid_bonus_win = defaultdict(int)
        for payed in range(2)[::-1]:
            pre_fix = ""
            if payed == 1:
                pre_fix = 'pay_'
            file_machine = os.path.join(file_dir, pre_fix + "machine_id.txt")
            if (not os.path.exists(file_machine)):
                continue
            file_bonus = os.path.join(file_dir, pre_fix + "win_bonus.txt")
            file_payin = os.path.join(file_dir, pre_fix + "pay_in.txt")
            file_line = os.path.join(file_dir, pre_fix + "line.txt")

            # 这里加一个bonus赢的钱

            self._do_read_user_data(
                file_machine, file_bonus, file_payin, file_line, seq_len, max_len, mid_bonus_count, mid_count, mid_bonus_win)

        # 读drop out的pay记录
        drop_dir = file_dir + os.path.sep + '..' + os.path.sep + \
            '__waring__' + os.path.sep + 'drop_pay' + os.path.sep + uid
        # print(drop_dir)
        file_machine = os.path.join(drop_dir, "machine_id.txt")
        if (os.path.exists(file_machine)):
            file_bonus = os.path.join(drop_dir, "win_bonus.txt")
            file_payin = os.path.join(drop_dir, pre_fix + "pay_in.txt")
            file_line = os.path.join(drop_dir, pre_fix + "line.txt")
            self._do_read_user_data(
                file_machine, file_bonus, file_payin, file_line, seq_len, max_len, mid_bonus_count, mid_count, mid_bonus_win)
        
        # 这里是为了找错误写的
        # result = []
        # with open("twh_data.txt",'a') as f:
        #     f.write("uid: " + str(uid) + ":\n")
        #     for payed, lv_mid_data in enumerate(mid_lv_count):
        #         # f.write("payed: " + str(payed) + ":\n")
        #         # tmp_lv = sorted(lv_mid_data.items(), key = lambda x: x[0])
        #         # for lv_mid in tmp_lv:
        #         #     f.write("lv: " + str(lv_mid[0] + "\n"))
        #         #     f.write(str(list(sorted(lv_mid[1].values()))))
        #         #     f.write("\n")
        #         for lv, mid_count in lv_mid_data.items():
        #             result += list(mid_count.values())
        #     f.write(str(sorted(result)))
        #     f.write("\n")

        return [mid_bonus_count, mid_count, mid_bonus_win]

class Mid_Free_Count_Reader(v_reader):
    def __init__(self, input_dir='null', out_put_dir='null'):
        v_reader.__init__(self, input_dir, out_put_dir)

    def _do_read_user_data(self, file_machine, file_free, seq_len, max_len, mid_free_count):
        """
        max_len : 原始文件的最大序列长度
        """
        # mid_lv_count = [defaultdict(_gen_defaultdict),
        #                 defaultdict(_gen_defaultdict)]
        if (not os.path.exists(file_machine)):
            return False
        f_machine = open(file_machine, 'r')
        f_free = open(file_free, 'r')
#            has_zero = False
        while True:
            line = f_machine.readline().strip()
            if not line:
                break
            free_line = f_free.readline().strip()

            line = line.split(" ")
            free_line = free_line.split(" ")
            # if len(line) < seq_len or line[-1] == "-1":  # 被抛掉的数据
            if line[-1] == '-1':
                continue
            cr_len = len(line)
            # for in line[::-1]
            for i in range(seq_len):
                # cr_index = max(0, cr_len - seq_len + i)
                cr_index = cr_len - seq_len + i
                if cr_index >= cr_len or cr_index < 0:
                    continue
                # print(line[cr_index])
                try:
                    cr_mid = int(float(line[cr_index]))
                    cr_free = int(float(free_line[cr_index]))
                    if cr_free != 0:
                        mid_free_count[cr_mid] += 1
                except:
                    break
            #         # [cr_mid] += 1
            # f_machine.close()
            # f_bonus.close()
            # f_coin.close()
            # f_time.close()
        # return [data, lable]
        f_machine.close()
        f_free.close()
        return True

    """
    读玩家数据，继承类可以重写这个函数
    """

    def _read_user_data(self, file_dir, seq_len, max_len):
        """
        max_len : 原始文件的最大序列长度
        """

        arr_dir = file_dir.split(os.path.sep)
        uid = arr_dir[-1]
        # print(uid)
        # exit()

        mid_free_count = defaultdict(int)
        for payed in range(2)[::-1]:
            pre_fix = ""
            if payed == 1:
                pre_fix = 'pay_'
            file_machine = os.path.join(file_dir, pre_fix + "machine_id.txt")
            if (not os.path.exists(file_machine)):
                continue
            file_free = os.path.join(file_dir, pre_fix + "win_free.txt")
            self._do_read_user_data(
                file_machine, file_free, seq_len, max_len, mid_free_count)

        # 读drop out的pay记录
        drop_dir = file_dir + os.path.sep + '..' + os.path.sep + \
            '__waring__' + os.path.sep + 'drop_pay' + os.path.sep + uid
        # print(drop_dir)
        file_machine = os.path.join(drop_dir, "machine_id.txt")
        if (os.path.exists(file_machine)):
            file_free = os.path.join(drop_dir, "win_free.txt")
            self._do_read_user_data(
                file_machine, file_free, seq_len, max_len, mid_free_count)

        return mid_free_count

class Mid_Count_Reader(v_reader):
    def __init__(self, input_dir='null', out_put_dir='null'):
        v_reader.__init__(self, input_dir, out_put_dir)

    def _do_read_user_data(self, file_machine, seq_len, max_len, mid_count):
        """
        max_len : 原始文件的最大序列长度
        """
        # mid_lv_count = [defaultdict(_gen_defaultdict),
        #                 defaultdict(_gen_defaultdict)]
        if (not os.path.exists(file_machine)):
            return False
        f_machine = open(file_machine, 'r')
#            has_zero = False
        while True:
            line = f_machine.readline().strip()
            if not line:
                break

            line = line.split(" ")
            # if len(line) < seq_len or line[-1] == "-1":  # 被抛掉的数据
            if line[-1] == '-1':
                continue
            cr_len = len(line)
            # for in line[::-1]
            for i in range(seq_len):
                # cr_index = max(0, cr_len - seq_len + i)
                cr_index = cr_len - seq_len + i
                if cr_index >= cr_len or cr_index < 0:
                    continue
                # print(line[cr_index])
                try:
                    cr_mid = int(float(line[cr_index]))
                    mid_count[cr_mid] += 1
                except:
                    break
            #         # [cr_mid] += 1
            # f_machine.close()
            # f_bonus.close()
            # f_coin.close()
            # f_time.close()
        # return [data, lable]
        f_machine.close()
        return True

    """
    读玩家数据，继承类可以重写这个函数
    """

    def _read_user_data(self, file_dir, seq_len, max_len):
        """
        max_len : 原始文件的最大序列长度
        """

        arr_dir = file_dir.split(os.path.sep)
        uid = arr_dir[-1]
        # print(uid)
        # exit()

        mid_count = defaultdict(int)
        for payed in range(2)[::-1]:
            pre_fix = ""
            if payed == 1:
                pre_fix = 'pay_'
            file_machine = os.path.join(file_dir, pre_fix + "machine_id.txt")
            if (not os.path.exists(file_machine)):
                continue
            self._do_read_user_data(
                file_machine, seq_len, max_len, mid_count)

        # 读drop out的pay记录
        drop_dir = file_dir + os.path.sep + '..' + os.path.sep + \
            '__waring__' + os.path.sep + 'drop_pay' + os.path.sep + uid
        # print(drop_dir)
        file_machine = os.path.join(drop_dir, "machine_id.txt")
        if (os.path.exists(file_machine)):
            self._do_read_user_data(
                file_machine, seq_len, max_len, mid_count)
        

        return mid_count



if __name__ == "__main__":

    bonus_reader = Mid_Bonus_Count_Reader()
    free_reader = Mid_Free_Count_Reader()
    mid_count_reader = Mid_Count_Reader()

    seq_len = 50
    max_len = 50

    import pickle
    vector_file = "mid_bonus_count.data"
    if not os.path.exists(vector_file):
        bonus_uid_2_vectors = bonus_reader.gen_uid_vector(seq_len, max_len)
        # with open(vector_file, 'wb') as f:
        #     pickle.dump(bonus_uid_2_vectors, f)
    else:
        with open(vector_file, 'rb') as f:
            bonus_uid_2_vectors = pickle.load(f)

    vector_file = "mid_free_count.data"
    if not os.path.exists(vector_file):
        free_uid_2_vectors = free_reader.gen_uid_vector(seq_len, max_len)
        with open(vector_file, 'wb') as f:
            pickle.dump(free_uid_2_vectors, f)
    else:
        with open(vector_file, 'rb') as f:
            free_uid_2_vectors = pickle.load(f)

    vector_file = "mid_count.data"
    if not os.path.exists(vector_file):
        mid_count_uid_2_vectors = mid_count_reader.gen_uid_vector(seq_len, max_len)
        with open(vector_file, 'wb') as f:
            pickle.dump(mid_count_uid_2_vectors, f)
    else:
        with open(vector_file, 'rb') as f:
            mid_count_uid_2_vectors = pickle.load(f)
    exit()

    # print(sum(uid_2_vectors.values(), []))
    mid_count = other_util.union_dict(*[x[1] for x in bonus_uid_2_vectors.values()])
    mid_bonus_count = other_util.union_dict(*[x[0] for x in bonus_uid_2_vectors.values()])
    mid_bonus_win = other_util.union_dict(*[x[2] for x in bonus_uid_2_vectors.values()])
    mid_free_count = other_util.union_dict(*free_uid_2_vectors.values())
    # print(list(uid_2_vectors.values()))
    sorted_mid_count = sorted(mid_count.items(), key = lambda x : float(x[0]))
    sorted_mid_bonus_count = sorted(mid_bonus_count.items(), key = lambda x : float(x[0]))
    sorted_mid_free_count = sorted(mid_free_count.items(), key = lambda x : float(x[0]))    
    sorted_mid_bonus_win = sorted(mid_bonus_win.items(), key = lambda x : float(x[0]))
    mids = np.array([x[0] for x in sorted_mid_count])
    spin_count = np.array([x[1] for x in sorted_mid_count])
    bonus_count = np.array([x[1] for x in sorted_mid_bonus_count])
    free_count = np.array([x[1] for x in sorted_mid_free_count])
    bonus_win = np.array([x[1] for x in sorted_mid_bonus_win])
    bonus_win_avg = bonus_win / bonus_count
    bonus_ratio = bonus_count / spin_count
    free_ratio = free_count / spin_count

    figure_path = file_util.get_figure_path("slot","bonus_free_ratio")

    # plt.gcf().set_size_inches(18, 9)
    # ax1 = plt.gca()
    # ax1.set_xlabel("machine id")
    # ax1.set_ylabel("spin count")
    # ax1.bar(mids, spin_count)
    # ax2 = ax1.twinx()
    # ax2.set_ylabel("ratio")
    # ax2.plot(mids, bonus_ratio, 'r.-', label = 'bonus_ratio')
    # ax2.plot(mids, free_ratio, 'g.-', label = 'free_ratio')
    # plt.legend(loc = "upper right")
    # # plt.bar(mids, spin_count)
    # title = "bonus_free_ratio"
    # plt.gca().set_title(title)
    # plt.savefig(os.path.join(figure_path, title + ".png"))
    # print(mid_bonus_count)
    # 
    # 
    
    plt.gcf().set_size_inches(18, 9)
    ax1 = plt.gca()
    ax1.set_xlabel("machine id")
    ax1.set_ylabel("bonus_win_average")
    title = "bonus_win_average"
    plt.gca().set_title(title)
    ax1.bar(mids, bonus_win_avg)
    plt.savefig(os.path.join(figure_path, title + ".png"))