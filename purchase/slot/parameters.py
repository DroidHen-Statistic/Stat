xgb_paras = {
        'n_estimators' : [73],
        'gamma': [0], 
        'learning_rate' : [0.1],
        'subsample': [0.81],
        'colsample_bytree': [0.7],
        'max_depth': [4],
    }


rf_paras = {
        'n_estimators': [99],
        'min_samples_split':[2],
        'min_samples_leaf':[3],
        'max_features':[0.80],
        'max_depth': [6],
}

GBDT_paras = {
        'learning_rate': [0.1],
        'n_estimators': [44],
        'min_samples_split':[2],
        'min_samples_leaf':[6],
        'max_features':[0.80],
        'subsample': [0.79],
        'max_depth': [2,3,4],
}