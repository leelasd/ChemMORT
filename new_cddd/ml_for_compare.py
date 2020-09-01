# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 15:21:25 2020

@Author: Zhi-Jiang Yang, Dong-Sheng Cao
@Institution: CBDD Group, Xiangya School of Pharmaceutical Science, CSU, China
@Homepage: http://www.scbdd.com
@Mail: yzjkid9@gmail.com; oriental-cds@163.com
@Blog: https://blog.iamkotori.com

♥I love Princess Zelda forever♥
"""

import re
import os
import json

import pandas as pd 
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error# regression
from sklearn.metrics import accuracy_score, roc_auc_score  #classfication
from sklearn.model_selection import train_test_split
# from imblearn.metrics import specificity_score, sensitivity_score # regression 

# from mol_utils import train_test_split

from tqdm import tqdm

END_POINT = 'tox'
SMILES_FILED = 'SMILES'
LABEL_FILED = 'Tox'

TASK = 'reg' # a member of ['reg', 'clf']
VEC = 'Newcdddwt' # a member of ['Morgan', 'Maccs', 'PubChem', 'Cddd2', 'Emb']
METHOD = 'XGB' # a member of ['RF', 'XGB']

MAX_LEN = 100
SEEDS = (71, 14, 76, 68, 42, 67, 46, 73, 56, 92,)

N_JOBS = 20

"""
RF
"""
rf_hyper = {'n_estimators': 800,
#            'class_weight': {0: 0.9, 1: 0.1} ,
#            'min_samples_split': 5,
            # 'min_samples_leaf': 2,
            # 'max_features': 'sqrt',
            # 'max_depth': 8,
            'n_jobs': N_JOBS, }


"""
XGB
"""
xgb_hyper= {'n_estimators': 1200,
            'max_depth': 6,
            'learning_rate': 0.02,
            'subsample': 0.4,
            # 'scale_pos_weight': 0.35,
            'n_jobs': N_JOBS,
            }

# xgb_hyper= {'n_estimators': 800,
#             'max_depth': 8,
#             'learning_rate': 0.02,
#             'subsample': 0.4,
#             # 'scale_pos_weight': 0.35,
#             'n_jobs': N_JOBS,
#             }

def build_rf_model(seed):
    """
    """
    
    if TASK == 'reg':
        model = RandomForestRegressor(random_state = seed, **rf_hyper)
    elif TASK == 'clf':
        model = RandomForestClassifier(random_state = seed, **rf_hyper)
    
    return model



def build_xgb_model(seed):
    """
    """
    if TASK == 'reg':
        model = XGBRegressor(random_state = seed, **xgb_hyper)
    elif TASK == 'clf':
        model = XGBClassifier(random_state = seed, **xgb_hyper)
        
    return model
    

def evaluate(model, X_test, y_test):
    """
    """
    y_pred = model.predict(X_test)
    
    if TASK == 'reg':
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return {'MSE':mse, 'MAE':mae, 'R2': r2, }
    
    else:
        y_porba = model.predict_proba(X_test)[:,1]
        
        acc = accuracy_score(y_test, y_pred)
        # sp = specificity_score(y_test, y_pred)
        # se = sensitivity_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_porba)
        
        return {'ACC': acc, 'AUC': auc}
    


def save_res(res, key):
    out = pd.DataFrame(res[key]).T
    out.loc['Mean'], out.loc['Std'] = out.mean(axis=0), out.std(axis=0)
    
    return out



if '__main__' == __name__:
    # assert TASK in ['reg', 'clf']
    # assert VEC in ['Morgan', 'Maccs', 'PubChem', 'Newcddd', 'Newcdddwt']
    # assert METHOD in ['RF', 'XGB']
    
    # try:
    #     os.makedirs('./data/{0}/{1}/res/{2}'.format(TASK, END_POINT, VEC.lower()))    
    # except FileExistsError:
    #     pass
    
    #Init Result
    res = {'Train':{}, 'Test':{}}
    #Load data
    data = pd.read_csv('./out.csv', sep='\t')
    X = data.drop('smiles', axis=1)
    y = X.pop('Value').to_numpy()
    X = X.to_numpy()
    
    for seed in tqdm(SEEDS):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # y_train = X_train.pop(LABEL_FILED).to_numpy()
        # y_test = X_test.pop(LABEL_FILED).to_numpy()
        
        # X_train = X_train.drop(SMILES_FILED, axis=1).to_numpy()
        # X_test = X_test.drop(SMILES_FILED, axis=1).to_numpy()
        
        if METHOD == 'XGB':
            m = build_xgb_model(seed)
            # with open('./data/{0}/{1}/res/{3}/{2}_hyper.json'.format(
            #         TASK, END_POINT, METHOD.lower(), VEC.lower()), 'w') as f_obj:
            #     json.dump(xgb_hyper, f_obj)
            # f_obj.close()
            
        elif METHOD == 'RF':
            m = build_rf_model(seed)
            # with open('./data/{0}/{1}/res/{3}/{2}_hyper.json'.format(
            #         TASK, END_POINT, METHOD.lower(), VEC.lower()), 'w') as f_obj:
            #     json.dump(rf_hyper, f_obj)
            # f_obj.close()
            
        
        # print(y_train.sum()/len(y_train))
    #     m.fit(X_train, y_train)
        
    #     train_metrics = evaluate(m, X_train, y_train)
    #     test_metrics = evaluate(m, X_test, y_test)
    #     print('\n{:5s}: {}'.format('Test', test_metrics), end='\n\n')
        
    #     res['Train'].update({seed: train_metrics})
    #     res['Test'].update({seed: test_metrics})
        
    # train = save_res(res, 'Train')
    # test = save_res(res, 'Test')
    
    # if TASK == 'reg':
    #     print('{}: {}'.format(METHOD, test.loc['Mean', 'R2']))
    # else:
    #     print('{}: {}'.format(METHOD, test.loc['Mean', 'AUC']))
    
    # train.to_csv('./data/{0}/{1}/res/{3}/{1}_{2}_train_{3}.csv'.format(TASK, END_POINT, METHOD.lower(), VEC.lower()))
    # test.to_csv('./data/{0}/{1}/res/{3}/{1}_{2}_test_{3}.csv'.format(TASK, END_POINT, METHOD.lower(), VEC.lower()))


    
    
    
    
    
    