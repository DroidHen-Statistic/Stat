from xgboost import XGBClassifier
import xgboost as xgb
import math
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer, f1_score
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import sys
head_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.append(head_path)

import config
def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions)

MSE = make_scorer(mean_squared_error_, greater_is_better=False)

def main(X, Y):
    start_time = time.time()

    print('--- Features Set: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Number of Features: ', len(X[0]))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 1024, test_size=0.1)
    # print(y_test)
    # exit()

    model = XGBClassifier(random_state = 0)

    # 调参
    param_grid = {
        'n_estimators' : [49],
        'gamma': [0], 
        'learning_rate' : [0.1],
        'subsample': [0.78],
        'colsample_bytree': [0.62],
        'max_depth': [3],
    }
    model = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = 1, cv=5, verbose=20, scoring = MSE)
    model.fit(X, Y)

    print('--- Grid Search Completed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Param grid:')
    print(param_grid)
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)



    #手动调参
    # from imp import reload 
    # while True:
    #     param_grid = config.Xgboost_category_config
    #     model = GridSearchCV(estimator = reg, param_grid = param_grid, n_jobs = 1, cv=10, verbose=20, scoring = MSE)
    #     model.fit(X_train, Y_train)
    
    #     # print('--- Grid Search Completed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    #     print('Param grid:')
    #     print(param_grid)
    #     print('Best Params:')
    #     print(model.best_params_)
    #     print('Best CV Score:')
    #     print(-model.best_score_)
        
    #     str = input("Modify the parameter: ")
    #     if str ==  's':
    #         break
    #     else:
    #         reload(config)

    
    # xgb接口
    # dtrain = xgb.DMatrix(x_train, y_train)
    # deval = xgb.DMatrix(x_test, y_test)
    # watchlist = [(deval, 'eval')]
    # params = {
    #     'num_boost_round ':500,
    #     'booster': 'gbtree',
    #     'objective': 'reg:linear',
    #     'subsample': 0.78,
    #     'colsample_bytree': 0.62,
    #     'eta': 0.1,
    #     'max_depth': 3,
    #     'seed': 0,
    #     'silent': 0,
    #     'eval_metric': 'rmse'
    # }
    # reg_xgb = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds = 50)
    # y_pred_xgb = reg_xgb.predict(xgb.DMatrix(X_test),ntree_limit = reg_xgb.best_ntree_limit)
    # pd.DataFrame({'id': test_index, 'y': y_pred_xgb}).to_csv(os.path.join(config.base_path,"answer", "xgb" + answer_file), index=False, header=False)

    print('--- Result Generated: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    return model


