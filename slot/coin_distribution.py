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
import logging


def _gen_defaultdict():
    # return copy.copy(defaultdict(int))
    return defaultdict(int)

class Coin_Reader(v_reader):
    def __init__(self, input_dir='null', out_put_dir='null'):
        v_reader.__init__(self, input_dir, out_put_dir)

    def _do_read_user_data(self, file_machine, file_bonus, file_payin, file_odds, file_line, seq_len, max_len, mid_line_odds_count):
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
        f_odds = open(file_odds, 'r')
        f_line = open(file_line, 'r')
#            has_zero = False
        while True:
            machine_line = f_machine.readline().strip()
            if not machine_line:
                break
            bonus_line = f_bonus.readline().strip()
            payin_line = f_payin.readline().strip()
            odds_line = f_odds.readline().strip()
            line_line = f_line.readline().strip()

            machine_line = machine_line.split(" ")
            bonus_line = bonus_line.split(" ")
            payin_line = payin_line.split(" ")
            odds_line = odds_line.split(" ")
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
                    cr_odds = float(odds_line[cr_index])
                    cr_line = int(float(line_line[cr_index]))
                    if cr_bonus != 0:
                        cr_odds = (cr_odds * cr_payin + cr_bonus) / cr_payin if cr_payin != 0 else 0
                    cr_odds = round(cr_odds, 1)
                    mid_line_odds_count[cr_mid][cr_line][cr_odds] += 1

                except Exception as ex:
                    logging.exception(ex)
                    print(ex)
                    break
            #         # [cr_mid] += 1
            # f_machine.close()
            # f_bonus.close()
            # f_coin.close()
            # f_time.close()
        # return [data, lable]
        f_machine.close()
        f_bonus.close()
        f_line.close()
        f_payin.close()
        f_odds.close()
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

        mid_line_odds_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for payed in range(2)[::-1]:
            pre_fix = ""
            if payed == 1:
                pre_fix = 'pay_'
            file_machine = os.path.join(file_dir, pre_fix + "machine_id.txt")
            if (not os.path.exists(file_machine)):
                continue
            file_bonus = os.path.join(file_dir, pre_fix + "win_bonus.txt")
            file_payin = os.path.join(file_dir, pre_fix + "pay_in.txt")
            file_odds = os.path.join(file_dir, pre_fix + "odds.txt")
            file_line = os.path.join(file_dir, pre_fix + "line.txt")
            self._do_read_user_data(
                file_machine, file_bonus, file_payin, file_odds, file_line, seq_len, max_len, mid_line_odds_count)
        # 读drop out的pay记录
        drop_dir = file_dir + os.path.sep + '..' + os.path.sep + \
            '__waring__' + os.path.sep + 'drop_pay' + os.path.sep + uid
        # print(drop_dir)
        file_machine = os.path.join(drop_dir, "machine_id.txt")
        if (os.path.exists(file_machine)):
            file_bonus = os.path.join(drop_dir, "win_bonus.txt")
            file_payin = os.path.join(drop_dir, "pay_in.txt")
            file_odds = os.path.join(drop_dir, "odds.txt")
            file_line = os.path.join(drop_dir, "line.txt")
            self._do_read_user_data(
                file_machine, file_bonus, file_payin, file_odds, file_line, seq_len, max_len, mid_line_odds_count)
        

        return mid_line_odds_count