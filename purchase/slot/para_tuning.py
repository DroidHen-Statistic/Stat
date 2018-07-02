import os
import sys
head_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.append(head_path)
sys.path.append(os.path.dirname(head_path))
import config

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer, f1_score, roc_auc_score
import time
from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV
from MysqlConnection import MysqlConnection
from data_reader import DataReader
import random
from imp import reload
from utils import *


import parameters
import pickle

def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions)

def auc_score_(y_true, y_pred_proba):
    return roc_auc_score(y_true, y_pred_proba[:,1])


MSE = make_scorer(mean_squared_error_, greater_is_better=False)
F1 = make_scorer(f1_score)
AUC = make_scorer(auc_score_, needs_proba = True)

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

    reader = DataReader(proportion=-1) 
    # ['average_bonus_win', 'spin_per_active_day', 'bonus_per_active_day', 'free_spin_ratio', 'coin', 'is_new', 'bonus_per_active_day+is_new', 'bonus_per_active_day-is_new', 'bonus_per_active_day/free_spin_ratio', 'bonus_per_active_day+bonus_per_active_day', 'coin-spin_per_active_day']
    # features = ["average_day_active_time","average_login_interval", "average_spin_interval", "average_bonus_win", "spin_per_active_day", "bonus_per_active_day","average_bet", "bonus_ratio", "free_spin_ratio", "coin"]
    feature_combine = ['average_bonus_win', 'spin_per_active_day', 'bonus_per_active_day', 'free_spin_ratio', 'coin', 'is_new', 'bonus_per_active_day+is_new', 'bonus_per_active_day-is_new', 'bonus_per_active_day/free_spin_ratio', 'bonus_per_active_day+bonus_per_active_day', 'coin-spin_per_active_day']
    df = reader.read("slot_purchase_profile_2017")
    df = other_util.dataCombine(df, feature_combine)
    df = df[feature_combine + ["purchase"]]
    

    # print(df.purchase)
    
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=43, stratify = df.purchase)
    df_train_y = df_train.pop("purchase")
    df_test_y = df_test.pop("purchase")
    # print(df_test)

    cf = GradientBoostingClassifier(random_state = 0)
    # cf = RandomForestClassifier(random_state=0)
    # cf = XGBClassifier(random_state=0, objective = "binary:logistic")
    # cf = LGBMClassifier(random_state=0, objective = "binary")

    p = ParaTuner(cf, AUC)

    while True:
        para_grid = parameters.gbdt_paras_2017
        model = p.tune(df_train,df_train_y,para_grid, cv_ = 5)
        str = input("Modify the parameter: ")
        if str == 's':
            with open("model/slot_gbdt_2017_combine.model", 'wb') as f:
                pickle.dump(model, f)
            break
        elif str == 'q':
            print("Parameter Tuning stopped")
            break
        else:
            reload(parameters)

    os.chdir(old_dir)
