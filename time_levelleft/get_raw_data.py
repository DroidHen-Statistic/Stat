import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from enum import Enum,unique
from MysqlConnection import MysqlConnection
# import utils
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale

@unique
class StageFormat(Enum):
    UID = 0
    STAGE_ID = 1
    PASSED = 2 # 0 未通过，1正常通过，2使用道具通过，3使用道具失败
    PLAY_TIME = 3
    DATA_VERSION = 4
    FIRST_PLAY = 5
    FIRST_PASS = 6
    IP = 7

# @unique
# class TotalDataFormat(Enum):
#     STAGE_ID = 0;
#     LOCALE = 1;
#     DATA_VERSION = 2;
#     USER_TOTAL = 3;
#     USER_PASSED = 4;
#     PLAY_TOTAL = 5;
#     PLAY_PASSED = 6;
#     PLAY_TIME_LENGTH = 7;
#     USE_ITEM_PASSED = 8;
#     USE_ITEM_USER = 9;
#     USER_ITEM_FAIL_COUNT = 10;
#     TODAY_USER = 11;
#     WEEK_USER = 12;
#     MONTH_USER = 13;


def read_log(date_start, date_end, game_id, data_version = -2, server_id = -1):
    files = file_util.get_log_files(date_start, date_end, "stage", game_id, server_id = -1)
    level_info = {}  # {data_version:{'user_count':, 'time_avg':, 'users':[]}}

    # 只有first pass的时间的平均值
    # for file in files:
    #     with open(file, 'r') as f:
    #         for line in f.readlines():
    #             line = line.split()
    #             _data_version = int(line[StageFormat.DATA_VERSION.value])
    #             if data_version != -2 and _data_version != data_version:
    #                 continue
    #             stage_id = int(line[StageFormat.STAGE_ID.value])
    #             if not stage_id in level_info:
    #                 level_info[stage_id] = {'user_count':0, 'time_avg':0, 'users':[], 'user_passed_count':0}
    #             uid = int(line[StageFormat.UID.value])
    #             first_pass = int(line[StageFormat.FIRST_PASS.value])
    #             if not uid in level_info[stage_id]['users']:
    #                 level_info[stage_id]['users'].append(uid)
    #                 level_info[stage_id]['user_count'] += 1
    #             if first_pass:
    #                 count = level_info[stage_id]['user_passed_count']
    #                 level_info[stage_id]['user_passed_count'] += 1
    #                 level_info[stage_id]['time_avg'] = (level_info[stage_id]['time_avg'] * count + int(line[StageFormat.PLAY_TIME.value])) / (count + 1)

    # first play到first pass之间的所有时间的总和
    play_time = {}
    for file in files:
        with open(file, 'r') as f:
            for line in f.readlines():
                line = line.split()
                stage_id = int(line[StageFormat.STAGE_ID.value])
                _data_version = int(line[StageFormat.DATA_VERSION.value])
                if data_version != -2 and _data_version != data_version:
                    continue
                if not stage_id in level_info:
                    level_info[stage_id] = {'user_count':0, 'time_avg':0, 'users':[]}
                    play_time[stage_id] = {"all":[]}
                uid = int(line[StageFormat.UID.value])
                first_play = int(line[StageFormat.FIRST_PLAY.value])
                first_pass = int(line[StageFormat.FIRST_PASS.value])
                if not uid in level_info[stage_id]['users']:
                    level_info[stage_id]['users'].append(uid)
                    level_info[stage_id]['user_count'] += 1
                if first_play or uid in play_time[stage_id]:
                    if first_play:
                        play_time[stage_id][uid] = []
                    play_time[stage_id][uid].append(int(line[StageFormat.PLAY_TIME.value]))
                if first_pass:
                    if uid in play_time[stage_id]:
                        play_time[stage_id]['all'].append(np.sum(play_time[stage_id][uid]))
                        play_time[stage_id].pop(uid)
    for stage_id in level_info.keys():
        level_info[stage_id]['time_avg'] = np.average(play_time[stage_id]['all']) if play_time[stage_id]['all'] != [] else 0
        

    connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
    level_left_table = "log_level_left_s_100712_1"
    sql = "select level, sum(user_7day) as sum from " + level_left_table + " where date >= %s and date <= %s group by level"
    result = connection.query(sql, (date_start, date_end))
    level_left = {}
    for x in result:
        level_left[x['level']] = int(x['sum'])
    for level in level_info.keys():
        # sql = "select sum(user_7day) as sum from " + level_left_table + " where level = %s and date > %s and date < %s"       
        # result = connection.query(sql, (int(level), date_start, date_end))
        level_info[level]["level_left"] = level_left[level]/level_info[level]['user_count'] * 100 if level in level_left else 0

    return level_info

def plot(level_info, level_start, level_end):
    tmp  = [(k,level_info[k]) for k in sorted(level_info.keys())]
    levels = [x[0] for x in tmp][level_start:level_end]
    time_avg = np.array([x[1]['time_avg'] for x in tmp][level_start:level_end])
    level_left = np.array([x[1]['level_left'] for x in tmp][level_start:level_end])
    # time_avg = (time_avg - np.average(time_avg))/np.std(time_avg)
    # level_left = (level_left - np.average(level_left))/np.std(level_left)
    time_avg = scale(time_avg)
    level_left = scale(level_left)
    
    plt.plot(levels,time_avg,'ro-',label = "time_avg")
    plt.plot(levels,level_left,'bo-',label = "level_left")
    # plt.plot(time_avg,level_left,'ro')
    plt.legend(loc = "upper right")

    # tmp = zip(time_avg, level_left)
    # tmp = sorted(tmp, key = lambda x:x[0])
    # x = [m[0] for m in tmp]
    # y = [m[1] for m in tmp]
    # print(x)
    # print(y)
    # plt.plot(x,y,'ro')

    plt.grid(True)
    plt.show()

def calculate_coef(level_info, level_start, level_end):
    tmp = [(k, level_info[k]) for k in sorted(level_info.keys())]
    levels = [x[0] for x in tmp][level_start:level_end]
    time_avg = [x[1]['time_avg'] for x in tmp][level_start:level_end]
    level_left = [x[1]['level_left'] for x in tmp][level_start:level_end]

    # print("level:",levels)
    #print(time_avg)
    #print(level_left)
    if any(time_avg) and any(level_left):
        data = np.vstack((time_avg, level_left))
        corr_coef = np.corrcoef(data, rowvar = True)
        return corr_coef[0,1]
    else:
        return 0

    



if __name__ == '__main__':
    # date_start = sys.argv[1]
    # date_end = sys.argv[2]
    # game_id = sys.argv[3]

    date_start = 20170601
    date_end = 20170630
    game_id = "s_100712"
    data_version = -2
    level_start = 1
    level_end = 100

    result = read_log(date_start, date_end, game_id, data_version)
    print(result[300]['user_count'])
    plot(result, level_start,level_end)
    #print(result)
    print(calculate_coef(result,level_start, level_end))
    #plot(result)
    