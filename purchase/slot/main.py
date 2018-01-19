import os
import sys

head_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
# print(head_path)
sys.path.append(head_path)
sys.path.append(os.path.dirname(head_path))
import config

old_dir = os.getcwd()
os.chdir(os.path.join(config.base_dir, "purchase", "slot"))

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from slot_log_parser import SlotPurchaseLogParser
from utils import *
from MysqlConnection import MysqlConnection
from para_tuning import ParaTuner
import random
import pickle
import paras
from sklearn.metrics import mean_squared_error, make_scorer, f1_score, roc_auc_score
from imp import reload 

def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions)

MSE = make_scorer(mean_squared_error_, greater_is_better=False)
F1 = make_scorer(f1_score)
AUC = make_scorer(roc_auc_score)


# after_read_file = os.path.join(config.log_base_dir, "after_read")
# profile_file = os.path.join(os.path.dirname(__file__), "data", "user_profiles_tmp_test")
# parser = SlotPurchaseLogParser(after_read_file, outfile = profile_file)

# parser.parse()
# parser.output_to_file()

conn = conn = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
# features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time", "average_login_interval", "average_spin_interval", "average_bonus_win", "average_bet", "bonus_ratio", "spin_per_active_day", "bonus_per_active_day"]
# features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time", "average_login_interval", "average_spin_interval", "average_bonus_win"]
features = ["average_day_active_time","average_login_interval", "average_spin_interval", "average_bonus_win", "spin_per_active_day", "bonus_per_active_day","average_bet", "bonus_ratio", "free_spin_ratio", "coin"]
x = []
y = []
# sql = "select uid, level, coin, purchase_times, active_days, average_day_active_time, average_login_interval, average_spin_interval from slot_user_profile where purchase_times > 0"
sql = "select * from slot_purchase_profile where purchase_times > 0 and active_days > 1"
result_pay = conn.query(sql)
pay_num = len(result_pay)
for record in result_pay:
    d = []
    for feature in features:
        d.append(record[feature])
    x.append(d)
    y.append(1)

# sql = "select uid, level, coin, purchase_times, active_days, average_day_active_time, average_login_interval, average_spin_interval from slot_user_profile where purchase_times = 0"
sql = "select * from slot_purchase_profile where purchase_times = 0 and active_days > 1"
result_no_pay = conn.query(sql)
result_no_pay = random.sample(result_no_pay, 5 * pay_num)
no_pay_num = len(result_no_pay)
for record in result_no_pay:
    d = []
    for feature in features:
        d.append(record[feature])
    x.append(d)
    y.append(0)

x = np.array(x)
y = np.array(y)


if os.path.exists("model/slot_rf.model"):
    with open("model/slot_rf.model","rb") as f:
        model = pickle.load(f)
else:
    model = RandomForestClassifier(random_state=0)

p = ParaTuner(model, AUC)

while True:
    para_grid = paras.rf_paras
    model = p.tune(x,y,para_grid)
    str = input("Modify the parameter: ")
    if str ==  's':
        with open("slot_rf.model", 'wb') as f:
            pickle.dump(model, f)
        break
    else:
        reload(paras)

os.chdir(old_dir)