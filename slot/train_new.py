import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # 服务器上跑
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


import copy

# class MyClass():
#         classVar = 2
#         def test(self):
#                 print( "self:%s class:%s" % (id(self.classVar) , id(MyClass.classVar)))
#                 self.classVar = self.classVar+1
#                 # print(id(self.classVar))
#                 # print(self.classVar)
# a = MyClass()
# a.test()
# a.test()
# # MyClass().test()
# exit()

# plt.figure(2)


def _gen_defaultdict():
    # return copy.copy(defaultdict(int))
    return defaultdict(int)

# 读机器id


class Machine_Vector_Reader(v_reader):
    # Machine_Id_2_Lv = {1:1, 4:2, 5:4, 6:5, 7:6, 11:7, 8:10, 10:13, 9:16,
    # 2:19, 3:22, 12:25, 13:28, 17:31, 14:34, 16:37, 15:40, 18:43, 19:46,
    # 20:49, 21:52, 22:55, 23:58, 24:61, 25:64, 26:67, 27:70, 28:73, 29:76,
    # 30:79, 31:82, 32:85, 33:88,34:91}
    # 已经排好序
    # max_group = 999
    Machine_Id_2_Lv = [(1, 1), (4, 2), (5, 4), (6, 5), (7, 6), (11, 7), (8, 10), (10, 13), (9, 16), (2, 19), (3, 22), (12, 25), (13, 28), (17, 31), (14, 34), (16, 37), (15, 40), (
        18, 43), (19, 46), (20, 49), (21, 52), (22, 55), (23, 58), (24, 61), (25, 64), (26, 67), (27, 70), (28, 73), (29, 76), (30, 79), (31, 82), (32, 85), (33, 88), (34, 91)]

    Mid_2_Lv_Pos_Dict = {}  # mid第几个等级开发
    Lv_Group_Machine_Count = defaultdict(int)  # 等级段开放的机器个数
    Lv_Group_Machine_Percent = defaultdict(float)   # 等级段开放的机器比例，开放两个就是50%

    def print_mid_unlock_lv(self):
        a = self.Machine_Id_2_Lv
        print("mid unlock lv:%s" % a)

    @staticmethod
    def lv_group_list():
        return [v[1] for v in Machine_Vector_Reader.Machine_Id_2_Lv]

    @staticmethod
    def lv_group_pos(lv_group):
        pos = 0
        for m_id, cr_lv in Machine_Vector_Reader.Machine_Id_2_Lv:
            if cr_lv == lv_group:
                return pos
            pos += 1
        return pos

    @staticmethod
    def lv_2_machine_id(lv):
        # lv_2_id = other_util.flip_dict(Machine_Vector_Reader.Machine_Id_2_Lv)
        m_ids = []
        for m_id, cr_lv in Machine_Vector_Reader.Machine_Id_2_Lv:
            # for m_id, cr_lv in
            # sorted(Machine_Vector_Reader.Machine_Id_2_Lv.items(), key = lambda x
            # : x[1] ):
            if lv >= cr_lv:
                m_ids.append(m_id)
            else:
                break
        return m_ids

    # 传入等级，返回等级分组
    @staticmethod
    def lv_2_group(lv):
        pre_lv = 0
        for m_id, cr_lv in Machine_Vector_Reader.Machine_Id_2_Lv:
            if lv < cr_lv:
                return pre_lv
            elif lv == cr_lv:
                return pre_lv
            else:
                pre_lv = cr_lv
        # return Machine_Vector_Reader.max_group
        return pre_lv

    @staticmethod
    def calc_mid_2_start_lv_pos():
        pos = 0
        for m_id, cr_lv in Machine_Vector_Reader.Machine_Id_2_Lv:
            Machine_Vector_Reader.Mid_2_Lv_Pos_Dict[m_id] = pos
            # Machine_Vector_Reader.Mid_2_Lv_Pos_Dict[m_id] = pos
            pos += 1

    @staticmethod
    def mid_2_start_lv_pos(mid):
        return Machine_Vector_Reader.Mid_2_Lv_Pos_Dict[mid]

    @staticmethod
    def calc_lv_group_m_count():
        for m_id, cr_lv in Machine_Vector_Reader.Machine_Id_2_Lv:
            m_count = len(Machine_Vector_Reader.lv_2_machine_id(cr_lv))
            Machine_Vector_Reader.Lv_Group_Machine_Count[cr_lv] = m_count
            Machine_Vector_Reader.Lv_Group_Machine_Percent[
                cr_lv] = round(1. / m_count, 4)
        # Machine_Vector_Reader.Lv_Group_Machine_Count[]

    def __init__(self, input_dir='null', out_put_dir='null'):
        v_reader.__init__(self, input_dir, out_put_dir)

    def _do_read_user_data(self, file_machine, file_lv, seq_len, max_len, mid_lv_count, payed):
        """
        max_len : 原始文件的最大序列长度
        """
        # mid_lv_count = [defaultdict(_gen_defaultdict),
        #                 defaultdict(_gen_defaultdict)]
        if (not os.path.exists(file_machine)):
            return False
        f_machine = open(file_machine, 'r')
        f_lv = open(file_lv, 'r')
#            has_zero = False
        while True:
            line = f_machine.readline().strip()
            if not line:
                break
            lv_line = f_lv.readline().strip()

            line = line.split(" ")
            lv_line = lv_line.split(" ")
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
                    cr_lv = int(float(lv_line[cr_index]))
                    cr_lv_group = self.lv_2_group(cr_lv)
                    # if cr_lv_group >= len(Machine_Vector_Reader.Machine_Id_2_Lv) or cr_lv_group <= 0 :
#                            print("uid lv:%s group:%s" % (cr_lv, cr_lv_group))
#                           has_zero = True
                    #    continue
                    mid_lv_count[payed][cr_mid][cr_lv_group] += 1
                    # mid_lv_count[payed][cr_mid][cr_lv] += 1
                except:
                    break
            #         # [cr_mid] += 1
            # f_machine.close()
            # f_bonus.close()
            # f_coin.close()
            # f_time.close()
        # return [data, lable]
        f_machine.close()
        f_lv.close()
        return True

    """
    读玩家数据，继承类可以重写这个函数
    """

    def _read_user_data(self, file_dir, seq_len, max_len):
        """
        max_len : 原始文件的最大序列长度
        """
        # pay_count = 0
        # no_pay_count = 0

        arr_dir = file_dir.split(os.path.sep)
        uid = arr_dir[-1]
        # print(uid)
        # exit()

        mid_lv_count = [defaultdict(_gen_defaultdict),
                        defaultdict(_gen_defaultdict)]
        for payed in range(2)[::-1]:
            pre_fix = ""
            if payed == 1:
                pre_fix = 'pay_'
            file_machine = os.path.join(file_dir, pre_fix + "machine_id.txt")
            if (not os.path.exists(file_machine)):
                continue
            file_lv = os.path.join(file_dir, pre_fix + "level.txt")
            self._do_read_user_data(
                file_machine, file_lv, seq_len, max_len, mid_lv_count, payed)

            # f_machine = open(file_machine, 'r')
            # f_lv = open(file_lv, 'r')
# #            has_zero = False
#             while True:
#                 line = f_machine.readline().strip()
#                 if not line:
#                     break
#                 lv_line = f_lv.readline().strip()

#                 line = line.split(" ")
#                 lv_line = lv_line.split(" ")
#                 # if len(line) < seq_len or line[-1] == "-1":  # 被抛掉的数据
#                 if line[-1] == '-1':
#                     break
#                 cr_len = len(line)
#                 # for in line[::-1]
#                 for i in range(seq_len):
#                     # cr_index = max(0, cr_len - seq_len + i)
#                     cr_index = cr_len - seq_len + i
#                     if cr_index >= cr_len or cr_index < 0:
#                         continue
#                     # print(line[cr_index])
#                     try:
#                         cr_mid = int(float(line[cr_index]))
#                         cr_lv = int(float(lv_line[cr_index]))
#                         cr_lv_group = self.lv_2_group(cr_lv)
#                         # if cr_lv_group >= len(Machine_Vector_Reader.Machine_Id_2_Lv) or cr_lv_group <= 0 :
# #                            print("uid lv:%s group:%s" % (cr_lv, cr_lv_group))
#  #                           has_zero = True
#                         #    continue
#                         mid_lv_count[payed][cr_mid][cr_lv_group] += 1
#                     except:
#                         break
        # 读drop out的pay记录
        drop_dir = file_dir + os.path.sep + '..' + os.path.sep + \
            '__waring__' + os.path.sep + 'drop_pay' + os.path.sep + uid
        # print(drop_dir)
        file_machine = os.path.join(drop_dir, "machine_id.txt")
        if (os.path.exists(file_machine)):
            file_lv = os.path.join(drop_dir, "level.txt")
            self._do_read_user_data(
                file_machine, file_lv, seq_len, max_len, mid_lv_count, payed=1)

            #         # [cr_mid] += 1
            # f_machine.close()
            # f_bonus.close()
            # f_coin.close()
            # f_time.close()
        # return [data, lable]
        # exit()
        # 
        
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

        return mid_lv_count


if __name__ == '__main__':
    # d = {3:33, 4:44}
    # k = other_util.flip_dict(d)
    # print(k)
    # Machine_Vector_Reader.calc_mid_2_start_lv(3)
    Machine_Vector_Reader.calc_mid_2_start_lv_pos()
    mv_reader = Machine_Vector_Reader()
    mv_reader.print_mid_unlock_lv()
    # exit()
    mv_reader.calc_lv_group_m_count()
    seq_len = 50
    max_len = 50
    mid_lv_count = [defaultdict(_gen_defaultdict),
                    defaultdict(_gen_defaultdict),
                    defaultdict(_gen_defaultdict),
                    defaultdict(_gen_defaultdict)]
    pay_user_mid_lv_count = defaultdict(_gen_defaultdict)
    nopay_user_mid_lv_count = defaultdict(_gen_defaultdict)
    pay_user_lv_total = defaultdict(int)
    nopay_user_lv_total = defaultdict(int)

    uid_2_vectors = []
    mid_lv_pay_uid_count= defaultdict(_gen_defaultdict)
    # 从文件里读
    import pickle
    vector_file = "vector.data"
    if not os.path.exists(vector_file):
        uid_2_vectors = mv_reader.gen_uid_vector(seq_len, max_len)
        with open(vector_file, 'wb') as f:
            pickle.dump(uid_2_vectors, f)
    else:
        with open(vector_file, 'rb') as f:
            uid_2_vectors = pickle.load(f)

    lv_total = defaultdict(_gen_defaultdict)
    pay_uids = mv_reader.get_pay_uids()
    for cr_uid, vector in uid_2_vectors.items():
        #        if int(cr_uid) != 1650378:
        #            continue
        #        print(cr_uid)
        for payed, cr_mid_lv_count in enumerate(vector):
            for cr_mid, lv_count in cr_mid_lv_count.items():
                for cr_lv, count in lv_count.items():
                    if cr_lv == 0:
                        # print("uid:%s mid:%s lv:0" % (cr_uid, cr_mid))
                        continue
                    mid_lv_count[payed][cr_mid][cr_lv] += count
                    lv_total[payed][cr_lv] += count
                    mid_lv_count[2][cr_mid][cr_lv] += count
                    lv_total[2][cr_lv] += count
                    if cr_uid in pay_uids:
                        mid_lv_count[3][cr_mid][cr_lv] += count
                        lv_total[3][cr_lv] += count
                        pay_user_mid_lv_count[cr_mid][cr_lv] += count
                        pay_user_lv_total[cr_lv] += count
                        mid_lv_pay_uid_count[cr_mid][cr_lv] += 1
                    else:
                        nopay_user_mid_lv_count[cr_mid][cr_lv] += count
                        nopay_user_lv_total[cr_lv] += count


    result = []
    with open("twh_data_total.txt", 'w') as f:
        for level, mid_count in mid_lv_count[2].items():
            result += [int(c) for c in list(mid_count.values())]
        f.write(str(sorted(result)))
        f.write("\n")
        # break
    # X = Machine_Vector_Reader.lv_group_list()
    # X = range(len(Machine_Vector_Reader.Machine_Id_2_Lv) ) # 要大于所有的 #错了
    figure_path = file_util.get_figure_path("slot", "machine_used")
    # if not os.path.exists(figure_path):
    #     os.mkdir(figure_path)

    X_label = [x for x in Machine_Vector_Reader.Lv_Group_Machine_Percent.keys()]
    import copy
    y_expect = [
        x for x in Machine_Vector_Reader.Lv_Group_Machine_Percent.values()]
    fig_count = 0
    # for payed, cr_mid_lv_count in enumerate(mid_lv_count):
    #     # title_raw = r"machine:%d" + (" (pay)" if payed else "")
    #     # postfix = "(pay)" if payed else "(no_pay)"
    #     if payed == 1:
    #         postfix = "(pay)"
    #     elif payed == 0:
    #         postfix = "(no_pay)"
    #     elif payed == 2:
    #         postfix = "(total)"
    #     else:
    #         postfix = "(pay_user)"
    #     for cr_mid, lv_count in cr_mid_lv_count.items():
    #         if len(lv_count) <= 0:
    #             continue
    #         title = "machine_%d%s" % (cr_mid, postfix)
    #         # title += str(cr_mid)
    #         cr_X_label = copy.copy(X_label)
    #         y = [0] * len(y_expect)
    #         total = ['0'] * len(y_expect)
    #         # print(lv_count)
    #         for cr_lv, count in lv_count.items():
    #             pos = Machine_Vector_Reader.lv_group_pos(cr_lv)
    #          #  print(Machine_Vector_Reader.lv_group_pos(cr_lv))
    #             cr_lv_total = lv_total[payed][cr_lv]
    #             if pos >= len(y) or pos < 0:
    #                 print("err mid:%s pos:%s lv:%s" %
    #                       (str(cr_mid), str(pos), str(cr_lv)))
    #                 continue
    #             y[pos] = round(count / cr_lv_total, 4) if cr_lv_total != 0 else 0
    #             # cr_X_label[pos] = str(cr_X_label[pos]) + "(%s)" % cr_lv_total
    #             cr_X_label[pos] = cr_X_label[pos]
    #             total[pos] = "1k+" if cr_lv_total > 1000 else str(cr_lv_total)
    #            # total[pos] = cr_lv_total

    #         #gcf = plt.figure(fig_count)
    #         plt.gcf().set_size_inches(12,6)

    #         # gcf = plt.figure(fig_count, figsize=(12, 6))
    #         # 
    #         ax = plt.gca()
    #         axisx = ax.xaxis
    #         ax.set_title(title)
    #         ax.set_xlabel("Lv group")
    #         # xlim=(0, X_label[-1])
    #         # ax.set_xlim(xlim)
    #         ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    #         ax.set_ylabel("use percent")

    #         start_pos = Machine_Vector_Reader.mid_2_start_lv_pos(cr_mid)
    #         cr_y_expect = [0.] * len(y)
    #         for i in range(start_pos, len(y)):
    #             cr_y_expect[i] = y_expect[i]

    #         ax.plot(cr_y_expect, '--.', label="expect")
    #         ax.plot(y, '-', label="real")

    #         ax.set_xticklabels([-1, 0] + cr_X_label)

    #         for label in axisx.get_ticklabels():
    #             # label.set_color("red")
    #             label.set_rotation(45)
    #             label.set_fontsize(8)
    #         # ax.set_xticklabels.set_ticklabels(['a','b','c','d','e'])
    #         # s1 = plt.subplot(111)
    #         # s1.xaxis.set_ticklabels(Machine_Vector_Reader.lv_group_list())

    #         # s1.set_ylabel('odds_mean')
    #         # s1.plt(X,y_expect, '--.', lable="use count")
    #         # s1.plt(X, y, '-o', lable="use count")

    #         # 加上标注
    #         for pos in range(len(y)):
    #             # plt.text(pos, y , total[pos] ,color='b',fontsize=2)
    #             if total[pos] != '0':
    #                 # plt.text(pos, y[pos], "total: %s" % total[pos])
    #                 plt.text(pos, y[pos], total[pos])

    #         # plt.text(15, 1, s='Numbers above curve are total counts',
    #         #          color='blue', va="top", ha="center")
    #         # plt.annotate('total counts above curve',xy=(0,0),xytext=(0.2,0.2),arrowprops=dict(facecolor='blue', shrink=0.1))

    #         ax.legend(loc="upper right")
    #         # plt.show()
    #         file_name = os.path.join(figure_path, title + ".png")
    #         # gcf.savefig(file_name, dpi= 160)
    #         # if payed >= 1:
    #         # gcf.savefig(file_name)
    #         plt.savefig(file_name)
    #         # plt.close(fig_count)
    #         plt.cla()
    #         # plt.close('all')
    #         # fig_count += 1
    #         # gcf.close()s
    #         # exit()
    #     # break
    #     # plt.figure()


    for cr_mid, lv_count in pay_user_mid_lv_count.items():
        if len(lv_count) <= 0:
            continue
        title = "machine_%d%s" % (cr_mid, "(pay user and nopay user)")
        # title += str(cr_mid)
        cr_X_label = copy.copy(X_label)
        y_pay = [0] * len(y_expect)
        y_nopay = [0] * len(y_expect)
        pay_total = nopay_total = ['0'] * len(y_expect)
        # print(lv_count)
        for cr_lv, count in lv_count.items():
            pay_pos = Machine_Vector_Reader.lv_group_pos(cr_lv)
         #  print(Machine_Vector_Reader.lv_group_pos(cr_lv))
            cr_lv_total = pay_user_lv_total[cr_lv]
            if pay_pos >= len(y_pay) or pay_pos < 0:
                print("err mid:%s pos:%s lv:%s" %
                      (str(cr_mid), str(pay_pos), str(cr_lv)))
                continue
            y_pay[pay_pos] = round(count / cr_lv_total, 4) if cr_lv_total != 0 else 0
            # cr_X_label[pos] = str(cr_X_label[pos]) + "(%s)" % cr_lv_total
            cr_X_label[pay_pos] = cr_X_label[pay_pos]
            pay_total[pay_pos] = "1k+" if cr_lv_total > 1000 else str(cr_lv_total)
            pay_total[pay_pos] += ("_" + str(mid_lv_pay_uid_count[cr_mid][cr_lv]))
            # print(pay_total[pay_pos])
           # total[pos] = cr_lv_total
        
        for cr_lv, count in nopay_user_mid_lv_count[cr_mid].items():
            nopay_pos = Machine_Vector_Reader.lv_group_pos(cr_lv)
         #  print(Machine_Vector_Reader.lv_group_pos(cr_lv))
            cr_lv_total = nopay_user_lv_total[cr_lv]
            if nopay_pos >= len(y_nopay) or nopay_pos < 0:
                print("err mid:%s pos:%s lv:%s" %
                      (str(cr_mid), str(nopay_pos), str(cr_lv)))
                continue
            y_nopay[nopay_pos] = round(count / cr_lv_total, 4) if cr_lv_total != 0 else 0
            # cr_X_label[pos] = str(cr_X_label[pos]) + "(%s)" % cr_lv_total
            cr_X_label[nopay_pos] = cr_X_label[nopay_pos]
            # nopay_total[nopay_pos] = "1k+" if cr_lv_total > 1000 else str(cr_lv_total)
           # total[pos] = cr_lv_total

        #gcf = plt.figure(fig_count)
        plt.gcf().set_size_inches(12,6)

        # gcf = plt.figure(fig_count, figsize=(12, 6))
        # 
        ax = plt.gca()
        axisx = ax.xaxis
        ax.set_title(title)
        ax.set_xlabel("Lv group")
        # xlim=(0, X_label[-1])
        # ax.set_xlim(xlim)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.set_ylabel("use percent")

        start_pos = Machine_Vector_Reader.mid_2_start_lv_pos(cr_mid)
        cr_y_expect = [0.] * len(y_pay)
        for i in range(start_pos, len(y_pay)):
            cr_y_expect[i] = y_expect[i]

        ax.plot(cr_y_expect, '--.', label="expect")
        ax.plot(y_pay, '-', label="pay_real")
        ax.plot(y_nopay, '-', label="nopay_real")
        ax.set_xticklabels([-1, 0] + cr_X_label)

        for label in axisx.get_ticklabels():
            # label.set_color("red")
            label.set_rotation(45)
            label.set_fontsize(8)
        # ax.set_xticklabels.set_ticklabels(['a','b','c','d','e'])
        # s1 = plt.subplot(111)
        # s1.xaxis.set_ticklabels(Machine_Vector_Reader.lv_group_list())

        # s1.set_ylabel('odds_mean')
        # s1.plt(X,y_expect, '--.', lable="use count")
        # s1.plt(X, y, '-o', lable="use count")

        # 加上标注
        for pos in range(len(y_pay)):
            # plt.text(pos, y , total[pos] ,color='b',fontsize=2)
            if pay_total[pos] != '0':
                # plt.text(pos, y[pos], "total: %s" % total[pos])
                # print(pay_total[pos])
                plt.text(pos, y_pay[pos], pay_total[pos])

        # plt.text(15, 1, s='Numbers above curve are total counts',
        #          color='blue', va="top", ha="center")
        # plt.annotate('total counts above curve',xy=(0,0),xytext=(0.2,0.2),arrowprops=dict(facecolor='blue', shrink=0.1))

        ax.legend(loc="upper right")
        # plt.show()
        figure_path = file_util.get_figure_path("slot", "machine_used", "user_pay_and_nopay")
        file_name = os.path.join(figure_path, title + ".png")
        # gcf.savefig(file_name, dpi= 160)
        # if payed >= 1:
        # gcf.savefig(file_name)
        plt.savefig(file_name)
        # plt.close(fig_count)
        plt.cla()
        # plt.close('all')
        # fig_count += 1
        # gcf.close()s
        # exit()
    # break
    # plt.figure()

    k = Machine_Vector_Reader.lv_2_machine_id(37)
    # print(k)

    # exit()
