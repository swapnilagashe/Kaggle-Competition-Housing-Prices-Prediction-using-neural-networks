#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:51:59 2020

@author: swapnillagashe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:42:37 2020

@author: swapnillagashe
"""

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn import  model_selection

import xgboost
import csv as csv
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from scipy.stats import skew
from collections import OrderedDict

import keras
import pickle
os.chdir('/Users/DATA/Coding /Kaggle /house-prices-advanced-regression-techniques/')

traindata=pd.read_pickle("./traindata.pkl")
testdata=pd.read_pickle("./testdata.pkl")
y=pd.read_pickle("./y.pkl")

original_data_train= pd.read_csv('train.csv')
original_data_test= pd.read_csv('test.csv')

target=pd.DataFrame(original_data_train['SalePrice'])


#Xgboost

model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 
model.fit(traindata,y)
oredered_dict=OrderedDict(sorted(model.get_booster().get_fscore().items(), key=lambda t: t[1], reverse=True))

most_relevant_features= list(dict((k, v) for k, v in model.get_booster().get_fscore().items() if v >= 10).keys())
most_relevant_features = [int(i) for i in most_relevant_features]
print(most_relevant_features)
traindata=traindata[most_relevant_features]
testdata=testdata[most_relevant_features]
#for tuning parameters
#parameters_for_testing = {
#    'colsample_bytree':[0.4,0.6,0.8],
#    'gamma':[0,0.03,0.1,0.3],
#    'min_child_weight':[1.5,6,10],
#    'learning_rate':[0.1,0.07],
#    'max_depth':[3,5],
#    'n_estimators':[10000],
#    'reg_alpha':[1e-5, 1e-2,  0.75],
#    'reg_lambda':[1e-5, 1e-2, 0.45],
#    'subsample':[0.6,0.95]  
#}

                    
#xgb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
#     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)

#gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6,iid=False, verbose=10,scoring='neg_mean_squared_error')
#gsearch1.fit(train_x,train_y)
#print (gsearch1.grid_scores_)
#print('best params')
#print (gsearch1.best_params_)
#print('best score')
#print (gsearch1.best_score_)

best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)
best_xgb_model.fit(traindata.to_numpy(),y.to_numpy())


#Let's remove the less important ones 

##removing outliers
#train_dataset = train_dataset[train_dataset.GrLivArea < 8.25]
#train_dataset = train_dataset[train_dataset.LotArea < 11.5]
#train_dataset = train_dataset[train_dataset.SalePrice<13]
#train_dataset = train_dataset[train_dataset.SalePrice>10.75]
#train_dataset.drop("Id", axis=1, inplace=True)



#for tuning parameters
#parameters_for_testing = {
#    'colsample_bytree':[0.4,0.6,0.8],
#    'gamma':[0,0.03,0.1,0.3],
#    'min_child_weight':[1.5,6,10],
#    'learning_rate':[0.1,0.07],
#    'max_depth':[3,5],
#    'n_estimators':[10000],
#    'reg_alpha':[1e-5, 1e-2,  0.75],
#    'reg_lambda':[1e-5, 1e-2, 0.45],
#    'subsample':[0.6,0.95]  
#}

                    
#xgb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
#     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)

#gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6,iid=False, verbose=10,scoring='neg_mean_squared_error')
#gsearch1.fit(train_x,train_y)
#print (gsearch1.grid_scores_)
#print('best params')
#print (gsearch1.best_params_)
#print('best score')
#print (gsearch1.best_score_)




predictions=best_xgb_model.predict(testdata.to_numpy())
sc=StandardScaler()
sc=sc.fit(target)
predictions_scaled=pd.DataFrame(sc.inverse_transform(predictions))
predictions_scaled.reset_index(inplace=True)
original_data_test.reset_index(inplace=True)
final_csv=original_data_test.merge(predictions_scaled, on='index')
final_csv=final_csv[['Id',0]]
final_csv = final_csv.rename(columns={0: 'SalePrice'})

final_csv.to_csv('Xgboost_with_feature_engg.csv',index=False)




