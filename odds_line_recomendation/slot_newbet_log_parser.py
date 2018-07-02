import os
import sys
head_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
# print(head_path)
sys.path.append(head_path)

import config
from utils import *
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict
from MysqlConnection import MysqlConnection
from enum import Enum, unique
import pickle
import csv
import seaborn

pattern = '[%s;%s,%s=%s,%s@%s,%s#%s,%s_%s,%s]'
fields = ["uid","coin","level","vip_level","pay","lidx","lpos", "prize", "gameid", "bet", "isfree"]

def get_log_files(directory):
    assert os.path.isdir(directory),'make sure directory argument should be a directory'
    result = []
    for root,dirs,files in os.walk(directory, topdown=True):
        for fl in files:
            result.append(os.path.join(root,fl))
    return result

class SlotNewbetLogParser(object):
    def __init__(self, newfile, oldfile = "", outfile = ""):
        self.logfiles = newfile
        self.outfile = outfile
        if oldfile == "":
            self.profiles = defaultdict(list)
            self.user_info = defaultdict(lambda:-1)
            self.line_profiles = {}
            self.line_info = defaultdict(dict)
        else:
            with open(oldfile, 'rb') as f:
                tmp = pickle.load(f)
                self.profiles = tmp[0]
                self.user_info = tmp[1]
        self.current_date = date_util.int_to_date(20170301)

    def parse(self):
        for f1 in self.logfiles:
            with open(f1, 'r') as f:
                for line in f.readlines():
                    tmp = line.strip().split(' ')
                    info = tmp[-1][1:-1].split(',')
                    if(len(info) < 6):
                        print("----info size error----")
                        print(line)
                        continue
                    uid = int(info[0].split(';')[0])
                    pay = int(info[2].split("@")[0])
                    if(pay > 0):    
                        if self.user_info[uid] == -1:
                            self.user_info[uid] = pay
                        elif self.user_info[uid] != pay:
                            self.user_info[uid] = pay
                            # lidx = int(info[2].split("@")[1])
                            # lpos = int(info[3].split("#")[0])
                            self.profiles[uid].append(info)

                    lidx = int(info[2].split("@")[1])
                    lpos = int(info[3].split("#")[0])
                    prize = int(info[3].split("#")[1])

                    if lidx not in self.line_profiles:
                        self.line_profiles[lidx] = {}
                        self.line_profiles[lidx]["count"] = 1
                        self.line_profiles[lidx]["prizes"] = [-1] * 200
                        self.line_profiles[lidx]["prizes"][lpos] = prize
                        self.line_info[lidx][uid] = lpos
                        if lpos == 199:
                            self.line_info[lidx].pop(uid)
                    else:
                        if uid not in self.line_info[lidx]:
                            self.line_profiles[lidx]["count"] += 1
                        self.line_info[lidx][uid] = lpos
                        if self.line_profiles[lidx]["prizes"][lpos] == -1:
                            self.line_profiles[lidx]["prizes"][lpos] = prize
                        # else:     
                        #     assert self.line_profiles[lidx]["prizes"][lpos] == prize, "%d %d %d %d"%(lidx, lpos, self.line_profiles[lidx]["prizes"][lpos], prize) #log里有些错误，比如会有lpos连续出现两遍

                        if lpos == 199:
                            self.line_info[lidx].pop(uid)

    def user_line(self):
        user_line_dict = {}
        for f1 in self.logfiles:
            with open(f1, 'r') as f:
                for line in f.readlines():
                    tmp = line.strip().split(' ')
                    info = tmp[-1][1:-1].split(',')
                    if(len(info) < 6):
                        print("----info size error----")
                        print(line)
                        continue
                    uid = int(info[0].split(';')[0])
                    lidx = int(info[2].split("@")[1])
                    if uid in user_line_dict:
                        user_line_dict[uid].add(lidx)
                    else:
                        user_line_dict[uid] = set([lidx])

        with open("data/user_line.data", 'wb') as f:
            pickle.dump(user_line_dict, f)

        return user_line_dict

    def output_to_file(self):
        with open("data/output.data", 'wb') as f:
            pickle.dump([self.profiles,self.line_profiles], f)

                    

if __name__ == "__main__":
    old_dir = os.getcwd()
    os.chdir(os.path.join(config.base_dir, "odds_line_recomendation"))

    log_dir = os.path.join(config.log_base_dir, "slot_newbet_")
    files = get_log_files(log_dir)
    p  = SlotNewbetLogParser(files)
    # p.parse()
    # p.output_to_file()

    # pay_user = p.profiles
    # lines = p.line_profiles

    # # with open("data/output.data", 'rb') as f:
    # #     p = pickle.load(f)
    # # pay_user = p[0]
    # # lines = p[1]

    # print(len(pay_user))
    # print(len(lines))
    # # for uid, lidx in pay_user.items():
    # #     print("uid: %d" %uid)
    # #     for info in lidx:
    # #         print(info)

    # with open("data/line.csv", 'w', newline = '') as f:
    #     writer = csv.writer(f)
    #     row = ["line_id", "count"] + [x for x in range(0, 200)]
    #     writer.writerow(row)
    #     for k, v in lines.items():
    #         row = []
    #         row.append(k)
    #         row.append(v["count"])
    #         row.extend(v["prizes"])
    #         writer.writerow(row)

    # os.chdir(old_dir)


    # user_line_dict = p.user_line()
    with open("data/user_line.data", 'rb') as f:
        user_line_dict = pickle.load(f)
    c = []
    count = 0
    for user, line in user_line_dict.items():
        if(len(line) > 10):
            count += 1
            print("%d : %d" %(user, len(line)))
            print(count)
        c.append(len(line))

    from collections import Counter
    import matplotlib.pyplot as plt
    import seaborn as sns

    # count = Counter(c)

    # count = count.items()

    # sorted(count, key = lambda x : x[1], reverse = True)

    # count = list(zip(*count))

    # plt.bar(count[0], count[1])

    # print(c)
    sns.distplot(c)
    plt.show()