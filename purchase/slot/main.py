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
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from slot_log_parser import SlotPurchaseLogParser
from utils import *
from MysqlConnection import MysqlConnection
from para_tuning import ParaTuner
from data_reader import DataReader
import random
import pickle
import parameters
from sklearn.metrics import mean_squared_error, make_scorer, f1_score, roc_auc_score, recall_score, precision_score, accuracy_score
from imp import reload
import matplotlib.pyplot as plt

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

# features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time", "average_login_interval", "average_spin_interval", "average_bonus_win", "average_bet", "bonus_ratio", "spin_per_active_day", "bonus_per_active_day"]
# features = ["login_times", "spin_times", "bonus_times", "active_days", "average_day_active_time", "average_login_interval", "average_spin_interval", "average_bonus_win"]

 
reader = DataReader() 
features = ["average_day_active_time","average_login_interval", "average_spin_interval", "average_bonus_win", "spin_per_active_day", "bonus_per_active_day","average_bet", "bonus_ratio", "free_spin_ratio", "coin"]

x, y = reader.read("slot_purchase_profile_2017", features)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=43)


if os.path.exists("model/slot_gbdt_2017.model"):
    with open("model/slot_gbdt_2017.model","rb") as f:
        model = pickle.load(f)
else:
    print("Waring : model not found")
    exit()
    model = RandomForestClassifier(random_state = 0)

p = ParaTuner(model, AUC)

# while True:
#     para_grid = parameters.xgb_paras
#     model = p.tune(x_train,y_train,para_grid)
#     str = input("Modify the parameter: ")
#     if str ==  's':
#         with open("model/slot_xgb.model", 'wb') as f:
#             pickle.dump(model, f)
#         break
#     else:
#         print("reload parameters")
#         reload(parameters)

# Print the feature importances
print(model.feature_importances_)
tmp = zip(features, model.feature_importances_)
tmp = sorted(list(tmp), key = lambda x : x[1])
plt.barh(range(len(tmp)), [x[1] for x in tmp])
plt.yticks(range(len(tmp)), [x[0] for x in tmp])
plt.title("feature_importances")
# plt.show()

y_pre = model.predict(x_test)
print(y_test)
print(y_pre)
recall = recall_score(y_test, y_pre)
precision = precision_score(y_test, y_pre)
accuracy = accuracy_score(y_test, y_pre)
f1 = f1_score(y_test, y_pre)

print("accuracy : %f" %accuracy)
print("recall : %f" %recall)
print("precision : %f" %precision)
print("f1 score : %f" %f1)

os.chdir(old_dir)