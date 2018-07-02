xgb_paras_2017 = {
        'n_estimators' : [55],
        'gamma': [0], 
        'learning_rate' : [0.1],
        'subsample': [0.6],
        'colsample_bytree': [0.75],
        'max_depth': [4],
        'scale_pos_weight' : [7],
        'max_delta_step' : [1]
    }


rf_paras_2017 = {
        'n_estimators': [104],
        'min_samples_split':[2],
        'min_samples_leaf':[2],
        'max_features':[0.69],
        'max_depth': [4],
        'class_weight' : ['balanced']
}

gbdt_paras_2017 = {
        'learning_rate': [0.1],
        'n_estimators': [44, 45, 46],
        'min_samples_split':[2],
        'min_samples_leaf':[7],
        'max_features':[0.63],
        'subsample': [0.65],
        'max_depth': [4],
}

gbdt_paras_2018 = {
        'learning_rate': [0.1],
        'n_estimators': [45, 55, 60],
        'min_samples_split':[2],
        'min_samples_leaf':[7],
        'max_features':[0.7],
        'subsample': [0.7],
        'max_depth': [2],
}

lgb_paras_2017 = {
        'n_estimators' : [110],
        'max_depth' : [3],
        'num_leaves' : [4],
        'subsample' : [0.6],
        'colsample_bytree' : [0.75],
        'learning_rate': [0.1],
        'min_child_samples' : [2],
}

stacking_paras = {
        'n_estimators': [60,70,80],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'subsample': [0.7,0.75,0.8],
        'colsample_bytree' : [0.65, 0.7, 0.75,0.8],
        'max_depth' : [2, 3, 4]
}

SVC_paras = {
        'C' : [0.1,1,10],
        'kernel' : ['rbf'],
        'tol' : [1e-3, 1e-2],
        'max_iter' : [-1],
        # 'degree ' : [2,3,4,5],  #Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
        'gamma' : ['auto']  #Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.
}

logistic_paras = {
        'tol' : [1e-4, 1e-3, 1e-2],
        'C' : [0.1,1,10]
}