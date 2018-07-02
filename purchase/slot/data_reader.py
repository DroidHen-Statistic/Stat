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
import pandas as pd

# old_dir = os.getcwd()
# os.chdir(os.path.join(config.base_dir, "purchase", "slot"))
from MysqlConnection import MysqlConnection


class DataReader(object):

    def __init__(self, proportion = 5):
        self.proportion = 5


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
        
    def read(self, table, features = None):
        conn = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
        # features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time", "average_login_interval", "average_spin_interval", "average_bonus_win", "average_bet", "bonus_ratio", "spin_per_active_day", "bonus_per_active_day"]
        # features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time", "average_login_interval", "average_spin_interval", "average_bonus_win"]
        # sql = "select uid, level, coin, purchase_times, active_days, average_day_active_time, average_login_interval, average_spin_interval from slot_user_profile where purchase_times > 0"
        if features == None:
            fs = '*'
        else:
            fs = ",".join(features)

        sql = "select "+ fs +" from " + table + " where purchase_times > 0 and active_days > 1"
        result_pay = conn.query(sql)
        pay_num = len(result_pay)

        df_pay = pd.DataFrame(result_pay)
        df_pay["purchase"] = 1

        sql = "select "+ fs + " from " + table + " where purchase_times = 0 and active_days > 1"
        result_no_pay = conn.query(sql)
        if self.proportion > 0:
            result_no_pay = random.sample(result_no_pay, self.proportion * pay_num)
        no_pay_num = len(result_no_pay)

        df_no_pay = pd.DataFrame(result_no_pay)
        df_no_pay["purchase"] = 0
        df = pd.concat([df_pay, df_no_pay])
        conn.close()

        # return np.array(x), np.array(y)
        return df




if __name__ == "__main__":
    reader = DataReader() 
    features = ["uid","average_day_active_time","average_login_interval", "average_spin_interval", "average_bonus_win", "spin_per_active_day", "bonus_per_active_day","average_bet", "bonus_ratio", "free_spin_ratio", "coin"]
    df = reader.read("slot_purchase_profile_2017", features)
    df = df.set_index("uid")

    print(df)


