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
from utils import *

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

MSE = make_scorer(mean_squared_error, greater_is_better=False)
F1 = make_scorer(f1_score)
AUC = make_scorer(roc_auc_score)


if __name__ == '__main__':

    old_dir = os.getcwd()
    os.chdir(os.path.join(config.base_dir, "purchase", "slot"))

    reader = DataReader(proportion=-1)
    features = ["average_day_active_time","average_login_interval", "average_spin_interval", "average_bonus_win", "spin_per_active_day", "bonus_per_active_day","average_bet", "bonus_ratio", "free_spin_ratio", "coin"]
    feature_combine = ['average_bonus_win', 'free_spin_ratio', 'average_bet', 'coin', 'average_bonus_win+coin', 'free_spin_ratio-average_bet', 'average_bet*average_bonus_win', 'coin*coin', 'coin+average_bet', 'coin-average_bet']
    df = reader.read("slot_purchase_profile_2017", features)
    df = other_util.dataCombine(df, feature_combine)
    

    # print(df.purchase)
    
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=43, stratify = df.purchase)
    df_train_y = df_train.pop("purchase")
    df_test_y = df_test.pop("purchase")

    print('--- Features Set: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Number of Features: ', df_train.shape[1])


    if os.path.exists("model/slot_rf_2017_combine.model"):
        with open("model/slot_rf_2017_combine.model","rb") as f:
            rf_model = pickle.load(f)
    else:
        print("Waring : RF model not found, default model used")
        exit()
        rf_model = RandomForestClassifier(random_state = 0, 
                            n_estimators=151, 
                            min_samples_split = 2,
                            min_samples_leaf = 2,
                            max_features = 0.79,
                            max_depth = 8
        )

    if os.path.exists("model/slot_xgb_2017_combine.model"):
        with open("model/slot_xgb_2017_combine.model","rb") as f:
            xgb_model = pickle.load(f)
    else:
        print("Waring : XGB model not found, default model used")
        exit()
        xgb_model = XGBRegressor(n_estimators = 66,
                        gamma = 0, 
                        learning_rate = 0.1,
                        subsample = 0.81,
                        colsample_bytree = 0.61,
                        max_depth = 3,
                        random_state=0
        )

    if os.path.exists("model/slot_gbdt_2017_combine.model"):
        with open("model/slot_gbdt_2017_combine.model","rb") as f:
            gbdt_model = pickle.load(f)
    else:
        print("Waring : GBDT model not found, default model used")
        exit()
        gbdt_model = GradientBoostingClassifier(learning_rate = 0.1, 
                                random_state = 0, 
                                n_estimators=30, 
                                min_samples_split = 2,
                                min_samples_leaf = 8,
                                max_features = 0.79,
                                subsample = 0.78,
                                max_depth = 5
        )

    if os.path.exists("model/slot_lgb_2017_combine.model"):
        with open("model/slot_lgb_2017_combine.model","rb") as f:
            lgb_model = pickle.load(f)
    else:
        print("Waring : lgb model not found, default model used")
        exit()
        lgb_model = LGBMClassifier(n_estimators = 110,
                            max_depth = 3,
                            num_leaves = 4,
                            subsample = 0.6,
                            colsample_bytree = 0.75,
                            learning_rate = 0.1,
                            min_child_samples = 2,
        )

    # if os.path.exists("model/slot_stacking.model"):
    #     with open("model/slot_stacking.model","rb") as f:
    #         stacker = pickle.load(f)
    # else:
    #     print("Waring : Stacking model not found, default model used")
    #     exit()
    #     stacker = XGBClassifier(
    #             n_estimators = 50,
    #             learning_rate = 0.05,
    #             subsample = 0.74,
    #             colsample_bytree = 0.67,
    #             max_depth = 3
    #     )

    base_models = {"RandomForest" : rf_model, "GBDT" : gbdt_model, "Xgboost" : xgb_model, "lgb": lgb_model}
    # base_models = {"GBDT" : gbdt_model, "Xgboost" : xgb_model}
    stacker = LogisticRegression(random_state=43)
    # stacker = XGBClassifier(random_state = 42)
    stacking_model = StackingClassifier(classifiers=list(base_models.values()),   
                          meta_classifier=stacker, use_probas=True, verbose = 1)  

    stacking_model.fit(df_train, df_train_y)

    base_models["stacking"] = stacking_model
    plt.figure()

    for name, clf in base_models.items():
        print("---------%s-----------" %name)
        y_pre = clf.predict(df_test)
        
        # print(y_test)
        # print(y_pre)
        recall = recall_score(df_test_y, y_pre)
        precision = precision_score(df_test_y, y_pre)
        accuracy = accuracy_score(df_test_y, y_pre)
        f1 = f1_score(df_test_y, y_pre)
    
        print("accuracy : %f" %accuracy)
        print("recall : %f" %recall)
        print("precision : %f" %precision)
        print("f1 score: %f" %f1)


        probas_ = clf.predict_proba(df_test)
        fpr, tpr, _ = roc_curve(df_test_y, probas_[:, 1])
        roc_auc = auc(fpr, tpr)  
        #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来  
        plt.plot(fpr, tpr, lw=1, label='ROC area of %s = %0.2f)' % (name, roc_auc))  
      
        #画对角线  
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])  
    plt.ylim([-0.05, 1.05])  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title('ROC')  
    plt.legend(loc="lower right")  
    plt.show()  

    print('--- Submission Generated: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

    os.chdir(old_dir)