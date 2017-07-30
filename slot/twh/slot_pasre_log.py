import sys
import os
head_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
print(head_path)
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import config
# from MysqlConnection import MysqlConnection
import numpy as np
from utils import *
# import copy
import pickle
from enum import Enum, unique

from scipy.ndimage.interpolation import shift
# import matplotlib.pyplot as plt
# from sklearn import preprocessing
# from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import tree
# import pydotplus


def check_and_mk_dir(path):
    isExists = os.path.exists(path)
    # 判断结果
    if isExists:
        return False
    else:
        # 如果不存在则创建目录
        # print path + ' 创建成功'
        # 创建目录操作函数
        os.makedirs(path)
        return True


# print(sys.path)
# file_util.check_and_mk_dir("abc/cdc/");
# k = date_util.int_to_timestamp(20170423000000)
# print(k)
# exit()

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


# 末尾
class FixedQueueArray(object):

    def __init__(self, capacity):
        self.capacity = capacity
        # self.sequence = np.zeros(capacity) - 1
        self.sequence = np.full(capacity, fill_value=np.infty, dtype=float)
        self.len = 0
        # np.full([3,4], fill_value=sys.maxsize, dtype=float)

    def push(self, a):
        if not self.full():
            self.len += 1    
        self.sequence = np.roll(self.sequence, -1)
        self.sequence[self.capacity - 1] = a

    def pop(self):
        v = self.sequence[self.capacity - 1]
        self.sequence = shift(self.sequence, 1, cval=np.infty)
        self.len -= 1
        return v

    def full(self):
        return self.sequence[0] != np.inf

    def empty(self):
        return self.sequence[self.capacity - 1] == np.inf

    def clear(self):
        self.sequence = np.full(
            self.capacity, fill_value=np.infty, dtype=float)
        self.len = 0

    def update_last(self, v):
        self.sequence[self.capacity - 1] = v

    def __str__(self):
        cr_str = ""
        sp = ""
        for v in self.sequence:
            # print(v)
            if v != np.inf:
                cr_str += (sp + str(v))
                sp = " "
        return cr_str

    def head_str(self, len_):  # 打印前n个
        cr_str = ""
        sp = ""
        start_pos = 0 if len_ > self.len else self.capacity - len_
        for i in range(start_pos, self.capacity):
            v = self.sequence[i]
            if v != np.inf:
                cr_str += (sp + str(v))
                sp = " "
        return cr_str
# test_que = FixedQueueArray(4)
# test_que.push(1)
# test_que.push(2)
# test_que.push(3)
# str_ = test_que.head_str(2)
# print(str_)
# exit()
# print(test_que)
# v = test_que.empty()
# print(v)
# test_que.push(3)
# v = test_que.empty()
# print(v)
# test_que.push(2)
# test_que.push(1)
# # v = test_que.full()
# # print(v)
# test_que.push(0)
# print(test_que)
# test_que.clear()
# print(test_que)
# v = test_que.full()
# print(v)
# v = test_que.empty()
# print(v)
# exit()

# test_que.push(3)
# print(test_que)
# v = test_que.full()
# print(v)

# v = test_que.pop()
# print(v)
# print(test_que)
# v = test_que.full()
# print(v)
# exit()
# # v = test_que.empty()
# # print(v)
# print(test_que)
# v = test_que.full()
# print(v)

# print(test_que)

# exit()


@unique
class FeaturePosFormat(Enum):
    ODDS = 0
    USE_FREE = 1
    WIN_FREE = 2
    TIME_DELTA = 3
    WIN_BONUS = 4
    COIN = 5
    LEVEL = 6
    PAY_IN = 7
    MACHINE_ID = 8
    LINE = 9

import csv
import copy
from collections import defaultdict

import shutil


class Log_Parser(object):
    features = ["odds", "is_free", "win_free",
                "time_delta", "win_bonus", "coin", "level", "pay_in", "machine_id", "line"]

    def __init__(self, log_file, out_put_dir, sequence_len, sequence_len_min, time_threshold=600):
        self.log_file = log_file
        self.sequence_len = sequence_len
        self.sequence_len_min = sequence_len_min  # 序列最短长度
        self.out_put_dir = out_put_dir

        if os.path.exists(out_put_dir):
            shutil.rmtree(out_put_dir)
        os.mkdir(out_put_dir)

        feature_counts = len(Log_Parser.features)
        self.out_files = np.zeros(feature_counts, dtype=str)
        self.time_threshold = time_threshold  # 一个序列的时间长度，超过且大于最短序列则强制中断输出

        # index = 0
        # for k in Log_Parser.features:
        #     self.out_files[index] = out_put_files[k]

        # self.odds_out_file, self.is_free_out_files = out_put_files
        #self.sequence = np.full(capacity, fill_value=np.infty, dtype=float)
        # self.feature_uid_seq = np.full(
        #     (feature_counts), fill_value=defaultdict(self.FixedQueueFactory))

        self.uid_last_spin = defaultdict(int)  # 上次转的时间戳
        self.uid_seq_seconds = defaultdict(int)  # 序列累计的时间
        self.uid_negtive_seq_len = defaultdict(int)  # 当前序列有多长
        # self.uid_positive_seq_len = defaultdict(int) # 当前记录

        self.feature_uid_seq = np.zeros(feature_counts, dtype=defaultdict)
        for i in range(feature_counts):
            self.feature_uid_seq[i] = defaultdict(self.FixedQueueFactory)

    def get_out_put_file(self, uid, feature_pos, cr_out_dir, pay=0):
        # base_file = self.out_files[feature_pos]
        base_file = Log_Parser.features[feature_pos] + ".txt"
        if pay:
            base_file = "pay_" + base_file
        cr_out_file = os.path.join(cr_out_dir, base_file)
        return cr_out_file

    def get_seq(self, uid, feature_pos):
        return self.feature_uid_seq[feature_pos][uid]
#        self.odds_sequence = defaultdict(self.FixedQueueFactory, )
#        self.is_free_sequence = defaultdict(self.FixedQueueFactory,)
        # self.online_uids = set()

    def __str__(self):
        show_str = ""
        show_str += "from file:%s" % self.log_file + "\n"
        show_str += "out put dir:%s" % self.out_put_dir + "\n"
        show_str += "seq len:%s" % self.sequence_len + "\n"
        show_str += "seq len min:%s" % self.sequence_len_min + "\n"
        # show_str += "ods out file:%s" % self.odds_out_file + "\n"
        # show_str += "is free out files:%s" % self.odds_out_file + "\n"
        return show_str

    def FixedQueueFactory(self):
        return copy.deepcopy(FixedQueueArray(self.sequence_len))

    def out_put_to_files(self, uid, pay=0, clear=0):
        # 输出到文件
        cr_out_dir = os.path.join(
            self.out_put_dir, str(uid))
        file_util.check_and_mk_dir(cr_out_dir)

        for feature_pos in range(len(Log_Parser.features)):
            cr_sq = self.get_seq(uid, feature_pos)
            # cr_out_file = os.path.join(cr_out_dir, self.get_out_put_file(uid, feature_pos, cr_out_dir))
            cr_out_file = self.get_out_put_file(
                uid, feature_pos, cr_out_dir, pay)
            prefix = ""
            if os.path.isfile(cr_out_file) and os.path.getsize(cr_out_file) > 0:
                prefix = "\n"
            with open(cr_out_file, "a") as f:
                if pay:
                    f.write(prefix + str(cr_sq))
                    no_pay_out_file = self.get_out_put_file(
                        uid, feature_pos, cr_out_dir, 0)
                    # out_put_len = self.uid
                    if os.path.isfile(no_pay_out_file) and os.path.getsize(no_pay_out_file) > 0:
                        with open(no_pay_out_file, "a") as f_no_pay:
                            f_no_pay.write(" -1")  # 标致上一条非充值的序列无效(和pay存在重叠的可能)
                else:
                    f.write(
                        prefix + cr_sq.head_str(self.uid_negtive_seq_len[uid]))
            if(clear):  # 充值才清空
                cr_sq.clear()
        # if pay:
        #     self.uid_negtive_seq_len[uid] = 0
        # else:
        #     self.uid_positive_seq_len[uid] = 0
        self.uid_negtive_seq_len[uid] = 0

    def clear_uid_seq(self, uid):
        for feature_pos in range(len(Log_Parser.features)):
            cr_sq = self.get_seq(uid, feature_pos)
            cr_sq.clear()
    #    exit() #TODO

    def uid_seq_full(self, uid):
        return self.feature_uid_seq[FeaturePosFormat.ODDS.value][uid].full()

    def uid_seq_can_out_put(self, uid):
        return self.feature_uid_seq[FeaturePosFormat.ODDS.value][uid].len >= self.sequence_len_min

    def parse_spin(self, line):
        date_int = int(line[SpinFormat.DATETIME.value])
        uid = int(line[SpinFormat.UID.value])

        timestamp = date_util.int_to_timestamp(date_int)
        last_timestamp = self.uid_last_spin[uid]
        seconds_past = 0 if last_timestamp == 0 else timestamp - last_timestamp
        self.uid_last_spin[uid] = timestamp

        machine_id = int(line[SpinFormat.MACHINE_ID.value])
        pay_in = int(line[SpinFormat.PAY_IN.value])
        use_free = int(line[SpinFormat.IS_FREE.value])
        win = int(line[SpinFormat.WIN.value])
        coin = int(line[SpinFormat.COIN.value])
        win_bonus = int(line[SpinFormat.WIN_BONUS.value])
        win_free_spin = int(line[SpinFormat.WIN_FREE_SPIN.value])
        bet = int(line[SpinFormat.BET.value])
        lines = int(line[SpinFormat.LINES.value])
        level = int(line[SpinFormat.LEVEL.value])

        pay_in = int(bet * lines)
        odds = round(win / pay_in, 2)

        # self.uid_negtive_seq_len[uid] = 0
# @unique
#     class FeaturePosFormat(Enum):
#         ODDS = 0
#         USE_FREE = 1
#         WIN_FREE = 2
#         TIME_DELTA = 3
#         WIN_BONUS = 4
#         COIN = 5
#         LEVEL = 6
        self.feature_uid_seq[FeaturePosFormat.ODDS.value][uid].push(odds)
        self.feature_uid_seq[FeaturePosFormat.USE_FREE.value][
            uid].push(use_free)
        self.feature_uid_seq[FeaturePosFormat.WIN_FREE.value][
            uid].push(win_free_spin)
        self.feature_uid_seq[FeaturePosFormat.TIME_DELTA.value][
            uid].push(seconds_past)
        self.feature_uid_seq[FeaturePosFormat.WIN_BONUS.value][
            uid].push(win_bonus)
        self.feature_uid_seq[FeaturePosFormat.COIN.value][uid].push(coin)
        self.feature_uid_seq[FeaturePosFormat.LEVEL.value][uid].push(level)
        self.feature_uid_seq[FeaturePosFormat.PAY_IN.value][uid].push(pay_in)
        self.feature_uid_seq[FeaturePosFormat.MACHINE_ID.value][
            uid].push(machine_id)
        self.feature_uid_seq[FeaturePosFormat.LINE.value][uid].push(lines)
        self.uid_negtive_seq_len[uid] += 1
        self.uid_seq_seconds[uid] += seconds_past

        if self.uid_seq_seconds[uid] >= self.time_threshold:
            if self.uid_negtive_seq_len >= self.sequence_len_min:
                self.out_put_to_files(uid)
            else:
                self.clear_uid_seq(uid)
        elif self.uid_negtive_seq_len[uid] >= self.sequence_len:
            self.out_put_to_files(uid)
        # exit()

    def parse_login(self, line):
        uid = int(line[LoginFormat.UID.value])
        # if self.uid_negtive_seq_len[uid] >= self.sequence_len:
        if self.uid_seq_can_out_put(uid):
            self.out_put_to_files(uid, clear=1)
            self.uid_last_spin[uid] = 0

    def parse_pay(self, line):
        uid = int(line[PurchaseFormat.UID.value])
        if self.uid_seq_can_out_put(uid):
            self.out_put_to_files(uid, pay=1, clear=1)

    def parse_play_bonus(self, line):
        uid = int(line[PurchaseFormat.UID.value])
        win_coin = line[PlayBonusFormat.TOTAL_WIN.value]
        self.feature_uid_seq[FeaturePosFormat.WIN_BONUS.value][
            uid].update_last(win_coin)

    def parse_log(self):
        log_file = self.log_file
        with open(log_file, 'r') as f:
            reader = csv.reader(f, delimiter=" ")
            for line in reader:
                data_type = int(line[0])
                if data_type == ActionType.LOGIN.value:
                    self.parse_login(line)
                elif data_type == ActionType.SPIN.value:
                    self.parse_spin(line)
                elif data_type == ActionType.PURCHASE.value:
                    self.parse_pay(line)
                elif data_type == ActionType.PLAYBONUS.value:
                    self.parse_play_bonus(line)
        # exit()

if __name__ == '__main__':
    log_file = os.path.join(config.log_base_dir, "after_read")
    # out_file_odds = os.path.join(config.log_result_dir, "odds")
    # seq_len = 50
    # seq_len_min = 10
    seq_len = 2
    seq_len_min = 2
    parser = Log_Parser(log_file, sequence_len = seq_len, sequence_len_min = seq_len_min, out_put_dir = os.path.join(
        config.log_result_dir, "slot"))
    # print(parser)
    parser.parse_log()

    exit()
