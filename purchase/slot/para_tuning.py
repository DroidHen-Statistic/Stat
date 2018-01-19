import os
import sys
head_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.append(head_path)
sys.path.append(os.path.dirname(head_path))
import config

old_dir = os.getcwd()
os.chdir(os.path.join(config.base_dir, "purchase", "slot"))

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer, f1_score, roc_auc_score
import time

from sklearn.model_selection import GridSearchCV
from MysqlConnection import MysqlConnection
import random
from imp import reload


import paras
import pickle

def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions)

MSE = make_scorer(mean_squared_error_, greater_is_better=False)
F1 = make_scorer(f1_score)
AUC = make_scorer(roc_auc_score)

class ParaTuner(object):

    def __init__(self, model, scorer):
        self.model = model
        self.scorer = scorer
    

    def tune(self, X, Y, param_grid, cv_ = 10):
        ret = GridSearchCV(estimator = self.model, param_grid = param_grid, n_jobs = 1, cv=cv_, verbose=20, scoring = self.scorer)
        ret.fit(X, Y)
    
        # print('--- Grid Search Completed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
        print('Param grid:')
        print(param_grid)
        print('Best Params:')
        print(ret.best_params_)
        print('Best CV Score:')
        print(ret.best_score_)
        return ret.best_estimator_


if __name__ == "__main__":
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


    cf = GradientBoostingClassifier(random_state = 0)

    p = ParaTuner(cf, AUC)

    while True:
        para_grid = paras.GBDT_paras
        model = p.tune(x,y,para_grid)
        str = input("Modify the parameter: ")
        if str == 's':
            with open("model/slot_gbdt.model", 'wb') as f:
                pickle.dump(model, f)
            break
        else:
            reload(paras)

os.chdir(old_dir)

