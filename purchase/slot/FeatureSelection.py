import os
import sys

head_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
# print(head_path)
sys.path.append(head_path)
sys.path.append(os.path.dirname(head_path))
import config
from data_reader import DataReader
from para_tuning import ParaTuner
import parameters

import time
start_time = time.time()
import numpy as np
import pandas as pd 

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, make_scorer, f1_score, roc_auc_score, recall_score, precision_score, accuracy_score,roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier,StackingCVClassifier  
import matplotlib.pyplot as plt
import os
import pickle

import LRS_SA_RGSS as LSR



def modelscore(y_test, y_pred):
    """for setting up the evaluation score
    """
    return roc_auc_score(y_test, y_pred)

""" The cross methods """
def add(x,y):
    return x + y

def substract(x,y):
    return x - y

def times(x,y):
    return x * y

def divide(x,y):
    return (x + 0.001)/(y + 0.001)

def obtaincol(df, delete):
    """ for getting rid of the useless columns in the dataset
    """
    ColumnName = list(df.columns)
    for i in delete:
        if i in ColumnName:
            ColumnName.remove(i)
    return ColumnName

def prepareData():
    """prepare you dataset here"""
    # with open('all.data', 'rb') as f:
    #     df = pickle.load(f)
    # df = df[~pd.isnull(df.is_trade)]

    reader = DataReader() 
    # features = ["average_day_active_time","average_login_interval", "average_spin_interval", "average_bonus_win", "spin_per_active_day", "bonus_per_active_day","average_bet", "bonus_ratio", "free_spin_ratio", "coin"]

    df = reader.read("slot_purchase_profile_2017")
    
    return df
    
def main(temp, clf, CrossMethod, RecordFolder, test = False):
    # set up the data set first
    df = prepareData()
    # get the features for selection
    uselessfeatures = ["first_active_time", "last_active_time", "purchase_times", "uid", "locale"]
    ColumnName = obtaincol(df, uselessfeatures) #obtain columns withouth the useless features
    SearchColumns = ColumnName[:] # the search features library. if columnname == [] teh code will run the backward searching at the very beginning
    # start selecting
    a = LSR.LRS_SA_RGSS_combination(df = df,
                                    clf = clf,
                                    RecordFolder = RecordFolder,
                                    LossFunction = modelscore,
                                    label = 'purchase',
                                    columnname = SearchColumns[:],  
                                    start = temp,
                                    CrossMethod = CrossMethod, # your cross term method
                                    PotentialAdd = [] # potential feature for Simulated Annealing
                                    )
    try:
        a.run()
    finally:
        with open(RecordFolder, 'a') as f:
            f.write('\n{}\n%{}%\n'.format(type,'-'*60))

if __name__ == "__main__":
    # algorithm group, add any sklearn type algorithm inside as a based selection algorithm
    # change the validation function in file LRS_SA_RGSS.py for your own case
    # model = {'lgb6': lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=5000, max_depth=3, learning_rate = 0.05, n_jobs=8)}

    old_dir = os.getcwd()
    os.chdir(os.path.join(config.base_dir, "purchase", "slot"))

    model = {'lgb6': LGBMClassifier(objective='binary',
                                        # metric='binary_error',
                                        num_leaves=8,
                                        max_depth=5,
                                        learning_rate=0.05,
                                        random_state=0,
                                        colsample_bytree=0.8,
                                        min_child_samples=2,
                                        subsample=0.7,
                                        n_estimators=20000,
                                        )
                                        }

    CrossMethod = {'+':add,
                   '-':substract,
                   '*':times,
                   '/':divide,}

    RecordFolder = 'record.log' # result record file
    modelselect = 'lgb6' # selected algorithm

    temp = ["average_day_active_time","average_login_interval", "average_spin_interval", "average_bonus_win", "spin_per_active_day", 
             "bonus_per_active_day","average_bet", "bonus_ratio", "free_spin_ratio", "coin"] # start features combination
    main(temp,model[modelselect], CrossMethod, RecordFolder,test=False)

    os.chdir(old_dir)