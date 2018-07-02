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
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, make_scorer, f1_score, roc_auc_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import pickle


def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions)

MSE = make_scorer(mean_squared_error_, greater_is_better=False)
F1 = make_scorer(f1_score)
AUC = make_scorer(roc_auc_score)


class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))
        S_train = np.zeros((X.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):

            print('Fitting For Base Model #%d / %d ---', i+1, len(self.base_models))
            for j, (train_idx, test_idx) in enumerate(folds):

                print('--- Fitting For Fold %d / %d ---', j+1, self.n_folds)

                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred

                print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

            print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        print('--- Base Models Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        clf = self.stacker
        clf.fit(S_train, y)

        print('--- Stacker Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

    def preidct(self, X):
        X = np.array(X)
        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state = 1026)
        S_test = np.zeros((X.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((X.shape[0], folds.get_n_splits(X)))
            for j, (train_idx, test_idx) in enumerate(folds.split(X)):
                S_test_i[:, j] = clf.predict(X)[:]
            S_test[:, i] = S_test_i.mean(1)

        clf = self.stacker
        y_pred = clf.predict(S_test)[:]
        return y_pred

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=2016)

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):

            print('Fitting For Base Model #{0} / {1} ---'.format(i+1, len(self.base_models)))

            S_test_i = np.zeros((T.shape[0], folds.get_n_splits(X)))

            for j, (train_idx, test_idx) in enumerate(folds.split(X)):

                print('--- Fitting For Fold #{0} / {1} ---'.format(j+1, self.n_folds))

                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

                print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

            tmp = S_test_i.mean(1)
            for k in range(len(tmp)):
                if tmp[k] >= 0.5:
                    tmp[k] = 1
                else:
                    tmp[k] = 0
            # print(tmp)
            # exit()
            S_test[:, i] = tmp

            print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        print('--- Base Models Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        # if os.path.exists("model/slot_stacking.model"):
        #     with open("model/slot_stacking.model","rb") as f:
        #         stacking_model = pickle.load(f)
        # else:
        #     print("Waring : Stacking model not found, default model used")
        #     stacking_model = XGBClassifier(
        #                 n_estimators = 50,
        #                 learning_rate = 0.05,
        #                 subsample = 0.74,
        #                 colsample_bytree = 0.67,
        #                 max_depth = 3
        #     )


        # grid = GridSearchCV(estimator=self.stacker, param_grid=param_grid, n_jobs=1, cv=5, verbose=20, scoring=MSE)
        # grid.fit(S_train, y)

        # # a little memo
        # message = 'to determine local CV score of #28'

        # try:
        #     print('Param grid:')
        #     print(param_grid)
        #     print('Best Params:')
        #     print(grid.best_params_)
        #     print('Best CV Score:')
        #     print(-grid.best_score_)
        #     print('Best estimator:')
        #     print(grid.best_estimator_)
        #     print(message)
        # except:
        #     pass



        # for i in range(len(y)):
        #     print(S_train[i], y[i])
        # exit()

        # 手动调参
        from imp import reload
        
        p = ParaTuner(self.stacker, F1)
        while True:
            param_grid = parameters.stacking_paras
            stacking_model = p.tune(S_train, y, param_grid)
            str = input("Modify the parameter: ")
            if str == 's':
                with open("model/slot_stacking.model", 'wb') as f:
                    pickle.dump(stacking_model, f)
                break
            else:
                reload(parameters)
    

        print('--- Stacker Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        y_pred = stacking_model.predict(S_test)[:]

        return y_pred


if __name__ == '__main__':

    old_dir = os.getcwd()
    os.chdir(os.path.join(config.base_dir, "purchase", "slot"))
    reader = DataReader() 
    features = ["average_day_active_time","average_login_interval", "average_spin_interval", "average_bonus_win", "spin_per_active_day", "bonus_per_active_day","average_bet", "bonus_ratio", "free_spin_ratio", "coin"]

    x, y = reader.read("slot_purchase_profile", features)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=46)

    print('--- Features Set: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Number of Features: ', len(x_train[0]))


    if os.path.exists("model/slot_rf.model"):
        with open("model/slot_rf.model","rb") as f:
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

    if os.path.exists("model/slot_xgb.model"):
        with open("model/slot_xgb.model","rb") as f:
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

    if os.path.exists("model/slot_gbdt.model"):
        with open("model/slot_gbdt.model","rb") as f:
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

    base_models = [rf_model, gbdt_model, xgb_model]
    # stacker = LogisticRegression(random_state=43)
    stacker = XGBClassifier(random_state = 42)

    ensemble = Ensemble(
        n_folds = 5,
        stacker = stacker,
        base_models = base_models
    )

    y_pre = ensemble.fit_predict(X=x_train, y=y_train, T=x_test)

    print(y_test)
    print(y_pre)
    recall = recall_score(y_test, y_pre)
    precision = precision_score(y_test, y_pre)
    accuracy = accuracy_score(y_test, y_pre)
    f1 = f1_score(y_test, y_pre)

    print("accuracy : %f" %accuracy)
    print("recall : %f" %recall)
    print("precision : %f" %precision)
    print("f1 score: %f" %f1)

    print('--- Submission Generated: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

    os.chdir(old_dir)