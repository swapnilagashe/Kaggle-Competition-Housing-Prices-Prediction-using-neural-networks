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
import pickle

import keras

os.chdir('/Users/DATA/Coding /Kaggle /house-prices-advanced-regression-techniques/')


original_data_train= pd.read_csv('train.csv')
original_data_test=pd.read_csv('test.csv')


train=original_data_train.copy()
test=original_data_test.copy()

target=pd.DataFrame(train['SalePrice'])
#-----adding sale price per sq ft feature

#create a feature price per sq ft
train['price_per_sqft']= train['SalePrice']/train['LotArea']
#we can also assign the mean value according to the neighbourhood
train['price_per_sqft'] = train.groupby('Neighborhood')['price_per_sqft'].transform(lambda x: x.median())

#Then we make a dictionary with the prices for each neighborhood: because we will have to add this feature to test data as well
d = {}
for indice_fila, x_train in train.iterrows():
    d.update({x_train['Neighborhood']:x_train['price_per_sqft']})
#And finally, we created the feature in the test dataset
test['price_per_sqft'] = 0.00
for indice, x_test in test.iterrows():
    test.loc[test.index == indice ,'price_per_sqft'] = d[x_test['Neighborhood']]
    
cat_vars_list=['MSSubClass', 'MSZoning','Street', 'MasVnrType','Alley','LotShape','LandContour','Utilities','LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond','RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinType2', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional','FireplaceQu', 'GarageType',
       'GarageFinish','GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC',
       'Fence', 'MiscFeature','SaleType',
       'SaleCondition','MoSold']
non_num_vars=cat_vars_list #we will need this later

f_to_remove= ['Alley', 'PoolQC', 'MiscFeature', 'Fence','FireplaceQu']

cat_vars_list= [e for e in cat_vars_list if e not in f_to_remove]

df_cat= train[cat_vars_list]
df_cat.info()

df_num = train.drop(['Id']+non_num_vars, axis=1)
df_num.info()
num_vars_list=df_num.columns.to_list()


####drop more cols
cat_drop_cols = ['Alley','FireplaceQu','PoolQC','MiscFeature','Fence'] #very less data points 
num_drop_cols = ['1stFlrSF','GarageArea','GarageYrBlt','MiscVal','3SsnPorch','PoolArea'] 
train.drop(cat_drop_cols, axis=1, inplace=True)
train.drop(num_drop_cols, axis=1, inplace=True)
#df_cat.drop(cat_drop_cols, axis=1, inplace=True) #already did this step above
df_num.drop(num_drop_cols, axis=1, inplace=True)

#---------------------------outliers handling--------------------
features_with_outliers_to_handle =['ScreenPorch','EnclosedPorch', 'OpenPorchSF','WoodDeckSF', 'GrLivArea','LowQualFinSF', 'TotalBsmtSF','BsmtFinSF1','LotArea']

for x in range(len(features_with_outliers_to_handle)):
    
    mean = df_num[features_with_outliers_to_handle[x]].mean()
    sd = df_num[features_with_outliers_to_handle[x]].std()
    lower_limit = mean - 3*sd
    upper_limit = mean + 3*sd
    df_num[features_with_outliers_to_handle[x]+'_no_outlier'] = pd.Series([min(max(a,lower_limit), upper_limit) for a in df_num[features_with_outliers_to_handle[x]]])

df_num.drop(features_with_outliers_to_handle, axis=1, inplace=True)
df_num.drop(['SalePrice'],axis=1,inplace=True)

#---------------fill missing numerical values
for column in df_num.columns:
    df_num[column].fillna(df_num[column].mean(), inplace=True)
    
#------------------scale numerical data
sc=StandardScaler()
df_num_scaled=sc.fit_transform(df_num)
df_num_scaled=pd.DataFrame(df_num_scaled).reset_index()
y=target.values
y=sc.fit_transform(y)
    
#---------------categorical data ----------
df_cat=df_cat.astype(str)
df_cat_encoded = pd.get_dummies(df_cat).reset_index(drop=True)
df_cat_encoded=pd.DataFrame(df_cat_encoded).reset_index()

#-----------combine num ans cat data
traindata=df_num_scaled.merge(df_cat_encoded,on='index')
traindata.drop(['index'], inplace=True, axis=1)

traindata=traindata.to_numpy()





#----------------------Training the model---------------------------

trainX, testX, trainY, testY = train_test_split(traindata, y, test_size=0.001, random_state=42)

#parameters
numFeatures=trainX.shape[1]
numLabels = trainY.shape[1]
n_hidden1 = 256
n_hidden2=128
n_hidden3=56
print(numFeatures,numLabels)
num_folds = 10
# Define per-fold score containers <-- these are new
acc_per_fold = []
loss_per_fold = []

# Merge inputs and targets
inputs = np.concatenate((trainX, testX), axis=0)
targets = np.concatenate((trainY, testY), axis=0)


# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation

fold_no = 1
batch_size=50
no_epochs=500
verbosity=1
validation_split=0.2


#optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)

for train, test in kfold.split(trainX, trainY):
    model =keras.Sequential([
        keras.layers.Dense(numFeatures,activation='relu'),
        keras.layers.Dense(256,activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(56, activation='relu'),
        keras.layers.Dense(numLabels, activation='linear'),
        ])
#    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    model.compile(optimizer='adam', loss='mse',metrics=['mae'])
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    
    history=model.fit(inputs[train],targets[train],batch_size=batch_size, epochs=no_epochs,verbose=verbosity,                                         validation_split=validation_split)
    
    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    # Increase fold number
    fold_no = fold_no + 1



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')




#-------------predictions---------------


to_predict=original_data_test
to_predict['price_per_sqft'] = 0.00
for indice, x_test in to_predict.iterrows():
    to_predict.loc[to_predict.index == indice ,'price_per_sqft'] = d[x_test['Neighborhood']]


df_cat_test= original_data_test[cat_vars_list]

df_num_test = original_data_test.drop(['Id']+non_num_vars, axis=1)

####drop more cols
to_predict.drop(cat_drop_cols, axis=1, inplace=True)
to_predict.drop(num_drop_cols, axis=1, inplace=True)
df_cat_test.drop(cat_drop_cols, axis=1, inplace=True) #already did this step above
df_num_test.drop(num_drop_cols, axis=1, inplace=True)



for x in range(len(features_with_outliers_to_handle)):
    
    mean = df_num_test[features_with_outliers_to_handle[x]].mean()
    sd = df_num_test[features_with_outliers_to_handle[x]].std()
    lower_limit = mean - 3*sd
    upper_limit = mean + 3*sd
    df_num_test[features_with_outliers_to_handle[x]+'_no_outlier'] = pd.Series([min(max(a,lower_limit), upper_limit) for a in df_num_test[features_with_outliers_to_handle[x]]])

df_num_test.drop(features_with_outliers_to_handle, axis=1, inplace=True)

#---------------fill missing numerical values
for column in df_num_test.columns:
    df_num_test[column].fillna(df_num_test[column].mean(), inplace=True)

sc=StandardScaler()
df_num_test_scaled=sc.fit_transform(df_num_test)
df_num_test_scaled=pd.DataFrame(df_num_test_scaled).reset_index()
    
#---------------categorical data ----------
df_cat_test=df_cat_test.astype(str)
df_cat_test_encoded = pd.get_dummies(df_cat_test).reset_index(drop=True)

df = pd.DataFrame({'cat':['a','b','c','d'],'val':[1,2,5,10]})
df1 = pd.get_dummies(pd.DataFrame(df_cat_test))
dummies_frame = pd.get_dummies(df_cat)
df_cat_test_encoded=df1.reindex(columns = dummies_frame.columns, fill_value=0)
df_cat_test_encoded=pd.DataFrame(df_cat_test_encoded).reset_index()



#-----------combine num ans cat data
testdata=df_num_test_scaled.merge(df_cat_test_encoded,on='index')
testdata.drop(['index'], inplace=True, axis=1)

testdata=testdata.to_numpy()




predictions=model.predict(testdata)
sc=StandardScaler()
sc=sc.fit(target)
predictions_scaled=pd.DataFrame(sc.inverse_transform(predictions))
predictions_scaled.reset_index(inplace=True)
original_data_test.reset_index(inplace=True)
final_csv=original_data_test.merge(predictions_scaled, on='index')
final_csv=final_csv[['Id',0]]
final_csv = final_csv.rename(columns={0: 'SalePrice'})

pd.DataFrame(testdata).to_pickle("./testdata.pkl")   #save testing data as a pickle file
pd.DataFrame(traindata).to_pickle("./traindata.pkl")   #save testing data as a pickle file
pd.DataFrame(y).to_pickle("./y.pkl")   #save testing data as a pickle file




final_csv.to_csv('model4_with_feature_engg_test_size0.001.csv',index=False)


