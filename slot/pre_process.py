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
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import pydotplus
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr


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
    def __init__(self, time, uid, is_free, machine_id, bet, lines, pay_in, win, coin, win_bonus, win_free_spin, level):
        Action.__init__(self, ActionType.SPIN.value, time, uid)
        self.is_free = is_free
        self.machine_id = machine_id
        self.bet = bet
        self.lines = lines
        self.pay_in = pay_in
        self.win = win
        self.coin = coin
        self.win_bonus = win_bonus
        self.win_free_spin = win_free_spin
        self.odds = self.win / self.pay_in if self.pay_in != 0 else 0
        self.level = level

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

class FixedQueue(object):   #定长队列
    def __init__(self, capacity):
        self.sequence = []
        self.capacity = capacity

    def push(self,a):
        if len(self.sequence) >= self.capacity:
            self.sequence.pop(0)
        self.sequence.append(a)

    def pop(self):
        self.sequence.pop()

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

def get_odds_data(file):
    user_dict = {}
    purchase_odds = []
    non_odds = []
    coins = []
    purchase_uid = set()
    # count = 0
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

                # 只用odds
                if uid not in user_dict:
                    user_dict[uid] = FixedQueue(10)
                user_dict[uid].push(odds)

            elif action_type == ActionType.PURCHASE.value:
                # count += 1
                purchase_uid.add(uid)
                if uid in user_dict:
                    # 只用odds
                    if user_dict[uid].full():
                        purchase_odds.append(user_dict[uid].sequence)
                        user_dict.pop(uid)
                else:
                    print("purchase but not recorded: ", uid)

    for uid, odds in user_dict.items():
        if uid not in purchase_uid:
            if odds.full():
                non_odds.append(odds.sequence)

    path = file_util.get_path(config.log_tmp_dir, "slot")
    with open(os.path.join(path,"odds_purchase_scale"),'wb') as f:
        pickle.dump(np.array(purchase_odds), f)
    with open(os.path.join(path,"odds_not_purchase_scale"),'wb') as f:
        pickle.dump(np.array(non_odds), f)
    return purchase_odds, non_odds

# def get_odds_data(file):
#     user_dict = {}
#     purchase_spins = []
#     non_spins = []
#     coins = []
#     purchase_uid = set()
#     # count = 0
#     with open(file, 'r') as f:
#         for line in f.readlines():
#             line = line.split()
#             if len(line) == 0:
#                 continue
#             action_type = int(line[0])
#             time = date_util.int_to_datetime(line[1])
#             uid = int(line[2])
#             if action_type == ActionType.SPIN.value:
#                 is_free = int(line[SpinFormat.IS_FREE.value])
#                 machine_id = int(line[SpinFormat.MACHINE_ID.value])
#                 pay_in = int(line[SpinFormat.PAY_IN.value])
#                 win = int(line[SpinFormat.WIN.value])
#                 coin = int(line[SpinFormat.COIN.value])
#                 win_bonus = int(line[SpinFormat.WIN_BONUS.value])
#                 win_free_spin = int(line[SpinFormat.WIN_FREE_SPIN.value])
#                 bet = int(line[SpinFormat.BET.value])
#                 lines = int(line[SpinFormat.LINES.value])
#                 level = int(line[SpinFormat.LEVEL.value])

#                 if uid not in user_dict:
#                     user_dict[uid] = FixedQueue(10)
#                 user_dict[uid].push(Spin(time, uid, is_free, machine_id, bet, lines, pay_in, win, coin, win_bonus, win_free_spin, level))


#             elif action_type == ActionType.PURCHASE.value:
#                 # count += 1
#                 purchase_uid.add(uid)
#                 if uid in user_dict:
#                     # 只用odds
#                     if user_dict[uid].full():
#                         purchase_spins.append(user_dict[uid].sequence)
#                         user_dict.pop(uid)
                        
#                     # # odds + coin
#                     # if user_dict[uid][0].full():
#                     #     purchase_odds.append(preprocessing.scale(user_dict[uid][0].sequence + [user_dict[uid][1]/10000]))
#                     #     user_dict.pop(uid)
#                 else:
#                     print("purchase but not recorded: ", uid)

#     for uid, odds in user_dict.items():
#         if uid not in purchase_uid:
#             # if odds[0].full():
#             #     non_odds.append(preprocessing.scale(odds[0].sequence + [odds[1]/10000]))

#             if odds.full():
#                 non_odds.append(odds.sequence)

#     path = file_util.get_path(config.log_tmp_dir, "slot")
#     with open(os.path.join(path,"odds_purchase_scale"),'wb') as f:
#         pickle.dump(np.array(purchase_odds), f)
#     with open(os.path.join(path,"odds_not_purchase_scale"),'wb') as f:
#         pickle.dump(np.array(non_odds), f)
#     return purchase_odds, non_odds


def get_odds_coin_data(file):
    user_dict = {}
    purchase_odds = []
    non_odds = []
    coins = []
    purchase_uid = set()
    # count = 0
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
                
                # # odds + coin
                if uid not in user_dict:
                    user_dict[uid] = [FixedQueue(11),coin]
                user_dict[uid][0].push(odds)
                user_dict[uid][1] = coin

            elif action_type == ActionType.PURCHASE.value:
                # count += 1
                purchase_uid.add(uid)
                if uid in user_dict:
                        
                    # # odds + coin
                    if user_dict[uid][0].full():
                        purchase_odds.append(preprocessing.scale(user_dict[uid][0].sequence + [user_dict[uid][1]/10000]))
                        user_dict.pop(uid)
                else:
                    print("purchase but not recorded: ", uid)

    for uid, odds in user_dict.items():
        if uid not in purchase_uid:
            if odds[0].full():
                non_odds.append(preprocessing.scale(odds[0].sequence + [odds[1]/10000]))

    path = file_util.get_path(config.log_tmp_dir, "slot")
    with open(os.path.join(path,"odds_coin_purchase_scale"),'wb') as f:
        pickle.dump(np.array(purchase_odds), f)
    with open(os.path.join(path,"odds_coin_not_purchase_scale"),'wb') as f:
        pickle.dump(np.array(non_odds), f)
    return purchase_odds, non_odds


def select_feature(x,y):

    SelectKBest(lambda X, Y: np.array(list(map(lambda x:pearsonr(x, Y), X.T))).T, k=2).fit_transform(x, y)

def train(x, y):
    skf = StratifiedKFold(n_splits=10)
    for train, test in skf.split(x, y):     #这里train 和 test分别保存了训练集和验证集在数据集中的下标，所以可以直接里利用该下标来取出对应的数据
        x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
        clf = tree.DecisionTreeClassifier(max_depth = 10)
        # clf = RandomForestClassifier(max_depth = 5)
        clf = clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        recall = 0
        for i in range(len(y_test)):
            if y_test[i] == 1 and y_predict[i] == 1:
                recall += 1
        print(y_predict)
        print("recall: %f" %(recall/np.sum(y_test)))
        print("positive ratio: ", np.sum(y_train)/len(y_train))
        print("percision: ", clf.score(x_test, y_test))
        # print(clf.predict(x_test))
        # print(y_test)
        # 
    path = file_util.get_figure_path("slot")
    with open(os.path.join(path,"test.dot"), 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data) 
    graph.write_pdf(os.path.join(path,"test.pdf"))




if __name__ == '__main__':
    logfile = file_util.get_path(config.log_base_dir, "slot", "after_read")
    # a = ActionSequence(logfile)
    # with open("E://log_tmp//stat//slot//sequence",'wb') as f:
    #     pickle.dump(a, f)
    # with open("E://log_tmp//stat//slot//sequence",'rb') as f:
    #     a = pickle.load(f)
    # purchase_odds,non_odds = get_data(a)
    

    # get_odds_data(logfile)
    
    path = file_util.get_path(config.log_tmp_dir, "slot")
    with open(os.path.join(path,"odds_purchase_scale"),'rb') as f:
        purchase_odds = pickle.load(f)
    with open(os.path.join(path,"odds_not_purchase_scale"),'rb') as f:
        non_odds = pickle.load(f)
    purchase_odds_diff = np.diff(purchase_odds)    
    non_odds_diff = np.diff(non_odds)

    purchase_odds_std = np.std(purchase_odds, axis=1)
    non_odds_diff_std = np.std(non_odds, axis=1)


    X = np.vstack((purchase_odds, non_odds))
    X_std = np.std(X, axis = 1).reshape(-1, 1)
    Y = np.array([1] * purchase_odds_diff.shape[0] + [0] * non_odds_diff.shape[0])
    print(X)
    print(X_std)
    X = np.hstack((X,X_std))
    print(X)

    select_feature(X,Y)

    # 画图
    # path = file_util.get_figure_path("slot")
    # plt.figure(1)
    # plt.plot([1] * len(purchase_odds[0]))
    # for i, odds in enumerate(purchase_odds):
    #     plt.plot(preprocessing.scale(odds))
    #     #plt.savefig(os.path.join(path, str(i)))
    #     #plt.cla()
    # plt.show()



    train(X,Y)
