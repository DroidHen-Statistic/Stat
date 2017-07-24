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
    lable = []

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
            cr_data = [0] * (3 + seq_len)
            line = line.split(" ")
            cr_data[0] = float(line[-1])

            line = f_bonus.readline().strip()
            line = list(map(float, line.split(" ")[max_len - seq_len::]))
            cr_data[1] = float(np.sum(line[::-1]))

            line = f_time.readline().strip()
            line = list(map(float, line.split(" ")[max_len - seq_len::]))
            cr_data[2] = float(np.sum(line))

            line = f_odds.readline().strip()
            line = line.split(" ")
            for i in range(seq_len):
                cr_data[3 + i] = float(line[max_len - seq_len + i])
            data.append(cr_data)
            lable.append(file_type)
        f_bonus.close()
        f_coin.close()
        f_odds.close()
        f_time.close()
    return [data, lable]

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
if __name__ == '__main__':
    file_names = ['coin', 'is_free', 'level', 'odds',
              'time_delta', 'win_bonus', 'win_free']
    seq_len = 5
    max_len = 10
    uid_2_vectors = gen_uid_vector(seq_len, max_len)

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
