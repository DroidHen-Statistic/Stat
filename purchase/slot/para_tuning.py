import os
import sys
head_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.append(head_path)
sys.path.append(os.path.dirname(head_path))
import config

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


import parameters
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
    old_dir = os.getcwd()
    os.chdir(os.path.join(config.base_dir, "purchase", "slot"))

    reader = DataReader() 
    features = ["average_day_active_time","average_login_interval", "average_spin_interval", "average_bonus_win", "spin_per_active_day", "bonus_per_active_day","average_bet", "bonus_ratio", "free_spin_ratio", "coin"]

    x, y = reader.read("slot_purchase_profile", features)


    cf = GradientBoostingClassifier(random_state = 0)

    p = ParaTuner(cf, AUC)

    while True:
        para_grid = parameters.GBDT_paras
        model = p.tune(x,y,para_grid)
        str = input("Modify the parameter: ")
        if str == 's':
            with open("model/slot_gbdt.model", 'wb') as f:
                pickle.dump(model, f)
            break
        else:
            reload(parameters)

    os.chdir(old_dir)

