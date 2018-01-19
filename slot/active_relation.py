import os
import sys
head_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
# print(head_path)
sys.path.append(head_path)
from MysqlConnection import MysqlConnection
import numpy as np
import config
import pymysql
from datetime import datetime
from scipy.stats import pearsonr

def get_data(features):
    conn = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname,cursorclass = pymysql.cursors.Cursor)
    feature_str = ""
    for feature in features:
        feature_str += (feature + ", ")
    sql = "select " + feature_str[:-2] +" from slot_user_profile_tmp where active_days >= 3"
    X = np.array(conn.query(sql))
    sql = "select active_days, first_active_time from slot_user_profile_tmp where active_days >= 3"
    active_data = np.array(conn.query(sql))
    Y = np.array([[y[0] / ((datetime(2017,8,22,23,59,59) - y[1]).days + 1)] for y in active_data])
    return np.hstack((X,Y))

def get_new_user_data(features):
    conn = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname,cursorclass = pymysql.cursors.Cursor)
    feature_str = ""
    for feature in features:
        feature_str += (feature + ", ")
    sql = "select " + feature_str[:-2] +" from slot_user_profile_tmp where is_new = 1"
    X = np.array(conn.query(sql))
    sql = "select active_days from slot_user_profile_tmp where is_new = 1"
    Y = np.array(conn.query(sql))
    return np.hstack((X,Y))

def cal_coef(data):
    for i in range(data.shape[1] - 1):
        print(pearsonr(data[:,i], data[:,-1]))

    

if __name__ == "__main__":
    # features = ["average_bonus_win", "bonus_ratio", "free_spin_ratio"]
    features = ["3day_free_spin_ratio", "3day_bonus_ratio", "3day_spin_times","3day_free_spin_times", "3day_bonus_times"]
    result = get_new_user_data(features)
    # result = np.array([[1,2,3,4,5,6,7],[2,4,6,8,10,12,100]])
    cal_coef(result)
    # print(coef)

