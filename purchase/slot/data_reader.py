import os
import sys

head_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
# print(head_path)
sys.path.append(head_path)
sys.path.append(os.path.dirname(head_path))
import config
import numpy as np
import random

# old_dir = os.getcwd()
# os.chdir(os.path.join(config.base_dir, "purchase", "slot"))
from MysqlConnection import MysqlConnection


class DataReader(object):

    def read_positive(self, table, features):
        conn = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
        # features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time", "average_login_interval", "average_spin_interval", "average_bonus_win", "average_bet", "bonus_ratio", "spin_per_active_day", "bonus_per_active_day"]
        # features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time", "average_login_interval", "average_spin_interval", "average_bonus_win"]
        x = []
        y = []
        # sql = "select uid, level, coin, purchase_times, active_days, average_day_active_time, average_login_interval, average_spin_interval from slot_user_profile where purchase_times > 0"
        sql = "select * from " + table + " where purchase_times > 0 and active_days > 1"
        result_pay = conn.query(sql)
        pay_num = len(result_pay)
        for record in result_pay:
            d = []
            for feature in features:
                d.append(record[feature])
            x.append(d)
            y.append(1)
        conn.close()
        return np.array(x), np.array(y)
        


    def read_negative(self, table, features):
        conn = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
        # sql = "select uid, level, coin, purchase_times, active_days, average_day_active_time, average_login_interval, average_spin_interval from slot_user_profile where purchase_times = 0"
        sql = "select * from " + table + " where purchase_times = 0 and active_days > 1"
        result_no_pay = conn.query(sql)
        result_no_pay = random.sample(result_no_pay, 5 * pay_num)
        # no_pay_num = len(result_no_pay)
        for record in result_no_pay:
            d = []
            for feature in features:
                d.append(record[feature])
            x.append(d)
            y.append(0)
        conn.close()
        return [np.array(x), np.array(y)]

    def read(self, table, features):
        conn = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
        # features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time", "average_login_interval", "average_spin_interval", "average_bonus_win", "average_bet", "bonus_ratio", "spin_per_active_day", "bonus_per_active_day"]
        # features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time", "average_login_interval", "average_spin_interval", "average_bonus_win"]
        x = []
        y = []
        # sql = "select uid, level, coin, purchase_times, active_days, average_day_active_time, average_login_interval, average_spin_interval from slot_user_profile where purchase_times > 0"
        sql = "select * from " + table + " where purchase_times > 0 and active_days > 1"
        result_pay = conn.query(sql)
        pay_num = len(result_pay)
        for record in result_pay:
            d = []
            for feature in features:
                d.append(record[feature])
            x.append(d)
            y.append(1)

        sql = "select * from " + table + " where purchase_times = 0 and active_days > 1"
        result_no_pay = conn.query(sql)
        result_no_pay = random.sample(result_no_pay, 5 * pay_num)
        no_pay_num = len(result_no_pay)
        for record in result_no_pay:
            d = []
            for feature in features:
                d.append(record[feature])
            x.append(d)
            y.append(0)
        conn.close()

        return np.array(x), np.array(y)



