import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
# from MysqlConnection import MysqlConnection
import numpy as np
from utils import *
# import copy
import pickle
from enum import Enum, unique
import matplotlib.pyplot as plt
from sklearn import preprocessing

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

class Action(object):

    def __init__(self, action_type, time, uid):
        self.action_type = action_type
        self.time = time
        self.uid = uid

class Login(Action):
    def __init__(self, time, uid):
        Action.__init__(self, ActionType.LOGIN.value, time, uid)

class Spin(Action):
    def __init__(self, time, uid, is_free, machine_id, pay_in, win, coin, win_bonus, win_free_spin):
        Action.__init__(self, ActionType.SPIN.value, time, uid)
        self.is_free = is_free
        self.machine_id = machine_id
        self.pay_in = pay_in
        self.win = win
        self.coin = coin
        self.win_bonus = win_bonus
        self.win_free_spin = win_free_spin
        self.odds = self.win / self.pay_in if self.pay_in != 0 else 0

class PlayBonus(Action):
    def __init__(self, time, uid, machine_id, total_win, coin):
        Action.__init__(self, ActionType.PLAYBONUS.value, time, uid)
        self.machine_id = machine_id
        self.total_win = total_win
        self.coin = coin
class Purchase(Action):
    def __init__(self, time, uid):
        Action.__init__(self, ActionType.PURCHASE.value, time, uid)

class User(object):
    def __init__(self, uid):
        self.uid = uid
        self.action_list = []

class FixedQueue(object):
    def __init__(self, capacity):
        self.sequence = []
        self.capacity = capacity

    def push(self,a):
        if len(self.sequence) >= self.capacity:
            self.sequence.pop(0)
        self.sequence.append(a)

    def size(self):
        return len(self.sequence)

    def full(self):
        return self.size() >= self.capacity

    def empty(self):
        return self.size() == 0

    def fill_to_full(self):
        while(not self.full()):
            self.sequence.insert(0,0)

    def clear(self):
        self.sequence = []

class ActionSequence(object):
    def __init__(self, log_file):
        self.user_dict = {}
        with open(log_file,'r') as f:
            for line in f.readlines():
                line = line.split()
                if len(line) == 0:
                    continue
                action_type = int(line[0])
                time = date_util.int_to_datetime(line[1])
                uid = int(line[2])
                if action_type == ActionType.LOGIN.value:
                    action = Login(time, uid)
                elif action_type == ActionType.SPIN.value:
                    #print(line)
                    is_free = int(line[SpinFormat.IS_FREE.value])
                    machine_id = int(line[SpinFormat.MACHINE_ID.value])
                    pay_in = int(line[SpinFormat.PAY_IN.value])
                    win = int(line[SpinFormat.WIN.value])
                    coin = int(line[SpinFormat.COIN.value])
                    win_bonus = int(line[SpinFormat.WIN_BONUS.value])
                    win_free_spin = int(line[SpinFormat.WIN_FREE_SPIN.value])
                    action = Spin(time, uid, is_free, machine_id, pay_in, win, coin, win_bonus, win_free_spin)
                elif action_type == ActionType.PLAYBONUS.value:
                    machine_id = int(line[PlayBonusFormat.MACHINE_ID.value])
                    total_win = int(line[PlayBonusFormat.TOTAL_WIN.value])
                    coin = int(line[PlayBonusFormat.COIN.value])
                    action = PlayBonus(time, uid, machine_id, total_win, coin)
                elif action_type == ActionType.PURCHASE.value:
                    action = Purchase(time, uid)
                else:
                    print("action_type_error", action_type)
                if uid not in self.user_dict:
                    self.user_dict[uid] = User(uid)
                self.user_dict[uid].action_list.append(action)

def get_data(action_sequence):
    purchase_odds = []
    non_odds = []
    purchase_flag = 0
    for uid, user in action_sequence.user_dict.items():
        odds_list = FixedQueue(10)
        purchase_flag = 0
        for action in user.action_list:
            if isinstance(action, Spin):
                if(not action.is_free):
                    odds_list.push(action.odds)
            elif isinstance(action, Purchase):
                purchase_flag = 1
                if not odds_list.full():
                    odds_list.fill_to_full()
                purchase_odds.append(odds_list.sequence)
                odds_list.clear()
                continue
        if purchase_flag == 0 and not odds_list.empty():
            odds_list.fill_to_full()
            non_odds.append(odds_list.sequence)

    return purchase_odds, non_odds

def get_data2(file):
    user_dict = {}
    purchase_odds = []
    non_odds = []
    purchase_uid = set()
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.split()
            if len(line) == 0:
                continue
            action_type = int(line[0])
            uid = int(line[2])
            if action_type == ActionType.SPIN.value:
                pay_in = int(line[SpinFormat.PAY_IN.value])
                win = int(line[SpinFormat.WIN.value])
                coin = int(line[SpinFormat.COIN.value])
                odds = win / pay_in if pay_in != 0 else 0
                if uid not in user_dict:
                    user_dict[uid] = FixedQueue(10)
                user_dict[uid].push(odds)
            elif action_type == ActionType.PURCHASE.value:
                purchase_uid.add(uid)
                if uid in user_dict:
                    user_dict[uid].fill_to_full()
                    purchase_odds.append(user_dict[uid].sequence)
                    user_dict.pop(uid)
    for uid, odds in user_dict.items():
        if uid not in purchase_uid:
            odds.fill_to_full()
            non_odds.append(odds.sequence)
    return purchase_odds, non_odds

if __name__ == '__main__':
    logfile = file_util.get_path(config.log_base_dir, "slot","after_read")
    # a = ActionSequence(logfile)
    # with open("E://log_tmp//stat//slot//sequence",'wb') as f:
    #     pickle.dump(a, f)
    # with open("E://log_tmp//stat//slot//sequence",'rb') as f:
    #     a = pickle.load(f)
    # purchase_odds,non_odds = get_data(a)
    # 
    print(sys.getsizeof(Login(1,1)))
    purchase_odds, non_odds = get_data2(logfile)
    plt.figure(1)
    for odds in purchase_odds:
        plt.plot(odds)
    plt.show()
    plt.figure(2)
    for odds in non_odds:
        plt.plot(odds)
    plt.show()