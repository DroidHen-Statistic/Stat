'''
这里用赔率序列做付费预测

'''


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

class Odds_Reader(v_reader):

    def __init__(self, input_dir='null', out_put_dir='null'):
        v_reader.__init__(self, input_dir, out_put_dir)

    def _read_user_data(self, file_dir, seq_len, max_len):
        pay_file = os.path.join(file_dir, "pay_odds.txt")
        data = [[], []]  # 0没充值，1充值

        pay_count = 0
        no_pay_count = 0
        for file_type in range(2):
            pre_fix = ""
            payed = 0
            if file_type == 0:
                pre_fix = 'pay_'
                payed = 1

            file_odds = os.path.join(file_dir, pre_fix + "odds.txt")
            if not os.path.exists(pay_file):
                continue
            f_odds = open(file_odds, 'r')


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

                cr_data = np.array(odds)


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

        # return [data, lable]
        if pay_count < 10:
            return []
        return data