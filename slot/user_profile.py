import numpy as np

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
from datetime import datetime, timedelta

from collections import defaultdict
from MysqlConnection import MysqlConnection
from enum import Enum, unique
import pickle

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
    IP = 3


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
    BET = 6

bet_list = [10,25,50,100,200,500,1000,1500,2000,2500,5000,10000,15000,20000,25000,30000,50000,100000,200000,250000]
level_list = [1,2,3,4,5,7,9,15,20,25,30,35,40,50,100,200]
bet_map = {1:200, 2:500, 3:1000, 4:1500, 5:2000, 7:2500, 9:5000, 15:10000, 20: 15000, 25:20000, 30:25000, 35:30000, 40:50000, 50:100000, 100: 200000, 200:250000}


class UserProfile(object):
    def __init__(self, filename):
        self.after_read_file = filename
        self.ipdb = other_util.IPDB(os.path.join(config.base_dir, "ipdb.csv"))
        self.profiles = {}
        self.user_info = {}
        self.current_date = date_util.int_to_date(20170401)

    def max_bet(self, level):
        low, high = 0, len(level_list)-1  
        pos = high
        while low<high:  
            mid = int((low+high)/2 + 1)
            if level_list[mid] <= level:  
                low = mid 
                pos = low
            else:#>
                high = mid - 1    
        return bet_map[level_list[pos]]

    def bet_ratio(self, bet, level):
        tmp = self.max_bet(level) 
        return (bet_list.index(bet) + 1) / (bet_list.index(tmp) + 1)

    def parse_log(self):
        with open(self.after_read_file, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                if int(line[0]) == ActionType.LOGIN.value:
                    self.parse_login(line)
                elif int(line[0]) == ActionType.SPIN.value:
                    self.parse_spin(line)
                elif int(line[0]) == ActionType.PLAYBONUS.value:
                    self.parse_bonus(line)
                elif int(line[0]) == ActionType.PURCHASE.value:
                    self.parse_purchase(line)
        for uid, profile in self.profiles.items():
            if "average_day_active_time" not in profile:
                profile["average_day_active_time"] = self.user_info[uid].get("day_active_time", 0) + self.user_info[uid].get("current_active_time", 0)
                profile["active_days"] = 1
            spin_times = profile.get("spin_times", 0)
            bonus_times = profile.get("bonus_times", 0)
            profile["bonus_ratio"] = bonus_times / spin_times if spin_times != 0 else bonus_times

    def parse_login(self, line):
        time = date_util.int_to_datetime(int(line[LoginFormat.DATETIME.value]))
        current_day = datetime(time.year, time.month, time.day, 0, 0, 0)
        uid = int(line[LoginFormat.UID.value])
        ip = line[LoginFormat.IP.value]
        locale = self.ipdb.ip2cc(ip)
        if uid not in self.profiles:
            self.profiles[uid] = {}
            self.user_info[uid] = {}
            self.user_info[uid]["first_login_time"] = time
            self.profiles[uid]["login_times"] = 1
            self.profiles[uid]["spin_times"] = 0
        else:
            login_interval = (time - self.user_info[uid]["last_login_time"]).total_seconds()
            login_times = self.profiles[uid]["login_times"]
            avg_interval = self.profiles[uid].get("average_login_interval",0)
            self.profiles[uid]["average_login_interval"] = (avg_interval * (login_times - 1) + login_interval) / login_times
            self.profiles[uid]["login_times"] += 1

        #下面是更新active time
        current_active_time = self.user_info[uid].get("current_active_time",0)
        day_active_time = self.user_info[uid].get("day_active_time", 0)
        if current_day - self.user_info[uid].get("last_active_day", current_day) >= timedelta(days = 1): #如果发现已经到了新的一天
            avg_active_time = self.profiles[uid].get("average_day_active_time", 0)
            active_days = self.profiles[uid].get("active_days", 0)
            day_active_time += current_active_time
            self.profiles[uid]["average_day_active_time"] = (avg_active_time * active_days + day_active_time) / (active_days + 1)
            self.profiles[uid]["active_days"] = active_days + 1
            self.user_info[uid]["day_active_time"] = 0
        else:
            self.user_info[uid]["day_active_time"] = day_active_time + current_active_time
        self.profiles[uid]["locale"] = locale
        self.user_info[uid]["current_active_time"] = 0
        self.user_info[uid]["last_login_time"] = time
        self.user_info[uid]["last_spin_time"] = -1
        self.user_info[uid]["last_active_time"] = time
        self.user_info[uid]["last_active_day"] = current_day


    def parse_spin(self, line):
        time = date_util.int_to_datetime(int(line[SpinFormat.DATETIME.value]))
        current_day = datetime(time.year, time.month, time.day, 0, 0, 0)
        uid = int(line[SpinFormat.UID.value])
        machine = int(line[SpinFormat.MACHINE_ID.value])
        pay_in = int(line[SpinFormat.PAY_IN.value])
        win = int(line[SpinFormat.WIN.value])
        coin = int(line[SpinFormat.COIN.value])
        bet = int(line[SpinFormat.BET.value])
        lines = int(line[SpinFormat.LINES.value])
        level = int(line[SpinFormat.LEVEL.value])
        if bet == 0:
            return
        if uid not in self.profiles:
            self.profiles[uid] = {}
            self.user_info[uid] = {}
            self.user_info[uid]["last_login_time"] = time
            self.user_info[uid]["first_login_time"] = time
            self.profiles[uid]["login_times"] = 1
            self.profiles[uid]["spin_times"] = 1
            self.user_info[uid]["last_spin_time"] = time
        else:
            self.profiles[uid]["spin_times"] += 1
            if self.user_info[uid]["last_spin_time"] != -1:
                self.user_info[uid]["spin_intervals"] = self.user_info[uid].get("spin_intervals", 0) + 1
                spin_interval = (time - self.user_info[uid]["last_spin_time"]).total_seconds()
                avg_spin_interval = self.profiles[uid].get("average_spin_interval", 0)
                login_times = self.profiles[uid]["login_times"]
                spin_intervals = self.user_info[uid].get("spin_intervals", 0)
                self.profiles[uid]["average_spin_interval"] = (avg_spin_interval * (spin_intervals - 1) + spin_interval) / spin_intervals
            self.user_info[uid]["last_spin_time"] = time

        bet_tmp = self.bet_ratio(bet, level)
        avg_bet = self.profiles[uid].get("average_bet", 0)
        spin_times = self.profiles[uid].get("spin_times", 0)
        self.profiles[uid]["average_bet"] = (avg_bet * (spin_times - 1) + bet_tmp) / spin_times
        # print(self.profiles[uid]["average_bet"])

        self.profiles[uid]["level"] = level
        self.profiles[uid]["coin"] = coin
        self.profiles[uid]["machine_" + str(machine)] = self.profiles[uid].setdefault("machine_" + str(machine), 0) + 1

        
        current_active_time = self.user_info[uid].get("current_active_time",0)

        if current_day - self.user_info[uid].get("last_active_day", current_day) >= timedelta(days = 1): #如果发现已经到了新的一天
            
            day_active_time = self.user_info[uid]["day_active_time"] + current_active_time
            avg_active_time = self.profiles[uid].get("average_day_active_time", 0)
            active_days = self.profiles[uid].get("active_days", 0)
            self.profiles[uid]["average_day_active_time"] = (avg_active_time * active_days + day_active_time) / (active_days + 1)
            self.user_info[uid]["current_active_time"] = (time - current_day).total_seconds()
            self.profiles[uid]["active_days"] = active_days + 1
            self.user_info[uid]["day_active_time"] = 0
        else:
            self.user_info[uid]["current_active_time"] = current_active_time + (time - self.user_info[uid].get("last_active_time", time)).total_seconds()

        self.user_info[uid]["last_active_day"] = current_day
        self.user_info[uid]["last_active_time"] = time


    def parse_bonus(self, line):
        time = date_util.int_to_datetime(int(line[SpinFormat.DATETIME.value]))
        current_day = datetime(time.year, time.month, time.day, 0, 0, 0)
        uid = int(line[PlayBonusFormat.UID.value])
        win = int(line[PlayBonusFormat.TOTAL_WIN.value])
        bet = int(line[PlayBonusFormat.BET.value]) / 100
        if bet != 0:
            avg_bonus_win = self.profiles[uid].get("average_bonus_win", 0)
            bonus_times = self.profiles[uid].get("bonus_times", 0)
            self.profiles[uid]["average_bonus_win"] = (avg_bonus_win * bonus_times + win / bet) / (bonus_times + 1)
            self.profiles[uid]["bonus_times"] = bonus_times + 1

        self.current_date = datetime(time.year, time.month, time.day)
        current_active_time = self.user_info[uid].get("current_active_time",0)
        if current_day - self.user_info[uid].get("last_active_day", current_day) >= timedelta(days = 1): #如果发现已经到了新的一天
            
            day_active_time = self.user_info[uid]["day_active_time"] + current_active_time
            avg_active_time = self.profiles[uid].get("average_day_active_time", 0)
            active_days = self.profiles[uid].get("active_days", 0)
            self.profiles[uid]["average_day_active_time"] = (avg_active_time * active_days + day_active_time) / (active_days + 1)
            self.user_info[uid]["current_active_time"] = (time - current_day).total_seconds()
            self.profiles[uid]["active_days"] = active_days + 1
            self.user_info[uid]["day_active_time"] = 0
        else:
            self.user_info[uid]["current_active_time"] = current_active_time + (time - self.user_info[uid].get("last_active_time", time)).total_seconds()

        self.user_info[uid]["last_active_day"] = current_day
        self.user_info[uid]["last_active_time"] = time

    def parse_purchase(self, line):
        time = date_util.int_to_datetime(int(line[SpinFormat.DATETIME.value]))
        current_day = datetime(time.year, time.month, time.day, 0, 0, 0)
        uid = int(line[PurchaseFormat.UID.value])
        self.profiles[uid]["purchase_times"] = self.profiles[uid].get("purchase_times", 0) + 1
        self.current_date = datetime(time.year, time.month, time.day)


        current_active_time = self.user_info[uid].get("current_active_time",0)
        if current_day - self.user_info[uid].get("last_active_day", current_day) >= timedelta(days = 1): #如果发现已经到了新的一天
            
            day_active_time = self.user_info[uid]["day_active_time"] + current_active_time
            avg_active_time = self.profiles[uid].get("average_day_active_time", 0)
            active_days = self.profiles[uid].get("active_days", 0)
            self.profiles[uid]["average_day_active_time"] = (avg_active_time * active_days + day_active_time) / (active_days + 1)
            self.user_info[uid]["current_active_time"] = (time - current_day).total_seconds()
            self.profiles[uid]["active_days"] = active_days + 1
            self.user_info[uid]["day_active_time"] = 0
        else:
            self.user_info[uid]["current_active_time"] = current_active_time + (time - self.user_info[uid].get("last_active_time", time)).total_seconds()
        self.user_info[uid]["last_active_day"] = current_day
        self.user_info[uid]["last_active_time"] = time



if __name__ == "__main__":
    after_read_file = os.path.join(config.log_base_dir, "after_read")
    parser = UserProfile(after_read_file)
   
    profile_file = os.path.join(os.path.dirname(__file__), "data", "user_profiles_tmp")
    if not os.path.exists(profile_file):
        parser.parse_log()
        profiles = parser.profiles
        with open(profile_file, 'wb') as f:
            pickle.dump(profiles, f)
    else:
        with open(profile_file, 'rb') as f:
            profiles = pickle.load(f)

    conn = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
    for uid, profile in profiles.items():
        columns = "(uid"
        val_format = "(%s"
        values = [uid]
        for col, val in profile.items():
            columns += (", " + col)
            val_format += (", " + "%s")
            values.append(val)
        columns += ")"
        val_format += ")"
        sql = "insert into slot_user_profile_tmp " + columns + " values " + val_format
        conn.query(sql, values)