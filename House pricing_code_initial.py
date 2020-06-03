#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 18:17:07 2020

@author: swapnillagashe
"""
""" 1.This is a regression problem and not classification problem
    2.RMSE will be used as the evaluation metric
"""
#House price detection using neural nets
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

import keras

os.chdir('/Users/DATA/Coding /Kaggle /house-prices-advanced-regression-techniques/')

original_data_train= pd.read_csv('train.csv')
original_data_test=pd.read_csv('test.csv')

train=original_data_train.copy()


target=pd.DataFrame(train['SalePrice'])
y=target.values #returns a numpy array
train=train.drop(['SalePrice'], axis=1)

all_features=train.columns.to_list()



################################Exploratory data Analyis and Encodings if any############################
#
##plot target variable against each feature and think about how each feature affects the salesprice
#
#for x in range(len(all_features)):
#    plt.figure()
#    plt.plot(train[all_features[x]],target.SalePrice)
#    plt.show()
#
#"""Handling categorical data options 
#1. No processing
#2.Frequency encoding
#3.OneHotEncoding
#4.LabelEncoding
#
#Handling rare categorical data(values in a row which occur very less number of time) - 
#group = df.groupby('feature')
#group.filter(lambda x: len(x) >= 100) 
#df.loc[df[col].value_counts()[df[col]].values < 10, col] = "RARE_VALUE"
#    """
#
#####################feature 1 - MSSubClass: Identifies the type of dwelling involved in the sale.
#train.MSSubClass.value_counts()
#
#sns.FacetGrid(train,size=5).map(sns.distplot,'MSSubClass').add_legend() #hue='SurvStat'
#
##We know that this is a categorical variable, so lets use label encoder 
#labelencoder=LabelEncoder()
#train['dwelling_type'] = labelencoder.fit_transform(data['MSSubClass'])
#features.append('dwelling_type')
#"""Label encoding has a con that higher numerical values might create a bias against lower numerical values while training the model, so we can use one hot encoding. But One hot encoding will add n columns to the data if there are n categories of a feature which might create a lot of columns in training set. For now lets use Label Encoder and we will check One hot encoder later"""
### creating instance of one-hot-encoder
##enc = OneHotEncoder(handle_unknown='ignore')
### passing bridge-types-cat column (label encoded values of bridge_types)
##enc_df = pd.DataFrame(enc.fit_transform(data[['MSSubClass']]).toarray())
### merge with main df bridge_df on key values
##data = data.join(enc_df)
##data.head()
#data.iloc[:,80]
#
#######feature 2 - MSZoning: Identifies the general zoning classification of the sale. - Categorical
#
#data['Zone'] = labelencoder.fit_transform(data['MSZoning'])
#sns.FacetGrid(data,size=5).map(sns.distplot,'Zone').add_legend() #hue='SurvStat'
#features.append('Zone')
#########Lets encode all the categorical variables



#-------------------decide which features are categorical and which are numerical

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
       'SaleCondition']
non_num_vars=cat_vars_list #we will need this later


df_cat=train[cat_vars_list]


df_cat.info()


"""We can see that some columns like Alley, PoolQc, MiscFeature, Fence,FireplaceQu have lot of empty values, so lets drop them for now, Later we can see if we should include them in the model..
for rows with some missing values,we can replace with the most occuring value for now, If we find a better approach we will implement if"""


f_to_remove= ['Alley', 'PoolQC', 'MiscFeature', 'Fence','FireplaceQu']

cat_vars_list= [e for e in cat_vars_list if e not in f_to_remove]

df_cat= df_cat[cat_vars_list]
df_cat['BsmtQual'].value_counts() #run this now and after replacing with mode to check how the values are affected, is it fair? if now think of some other approac later
df_cat.info()

#check how should we handle the missing values
for column in df_cat.columns:
    df_cat[column].fillna(df_cat[column].mode()[0], inplace=True)


df_cat.info() #No missing values in the dataset now :)


####################### Encoding cat vars with label encoder
df_cat_encoded=df_cat.apply(LabelEncoder().fit_transform) 

    
######numerical features
df_num = train.drop(['Id']+non_num_vars, axis=1)
df_num.info()
num_vars_list=df_num.columns.to_list() #can use this command as well, will do it while refining the code
#lot frontage has few missing values, lets fill them with the average values
for column in df_num.columns:
    df_num[column].fillna(df_num[column].mean(), inplace=True)

#features1 = sparse.hstack((num_features.astype(float), df_encoded)) #this was present in general workflow of abhishek thakur, I think it is necessary while doing one hot encoding


###########scaling the features
""" Scaling should be done when using Neural Nets, Linear Reg, Logistic Reg etc because they use concept of euclidean or manhattan distance
Scaling not required while using XGboost, LightGBM"""

sc=StandardScaler()
df_num_scaled=sc.fit_transform(df_num)
y=sc.fit_transform(y)

#-------------------combining categorical and numerical features
numerical_data=pd.DataFrame(df_num_scaled)
numerical_data=numerical_data.reset_index()

categorical_data=df_cat_encoded.reset_index()

complete_train_data= numerical_data.merge(categorical_data, on='index')
complete_train_data=complete_train_data.drop(['index'],axis=1)
train_array=complete_train_data.to_numpy()
num_features_used=df_num.columns.to_list()
cat_features_used=categorical_data.columns.to_list()
cat_features_used.remove('index')

num_features_used[0]
features_used=[num_features_used,cat_features_used]
features_used = [item for sublist in features_used for item in sublist]
complete_train_data.info()

""" Cross Validation Technique to be used = K fold cross validation, lets use 5 folds, we can vary this, maybe we will try with leave one out as well"""

RANDOM_STATE=42




#----------------------Training the model---------------------------

trainX, testX, trainY, testY = train_test_split(train_array, y, test_size=0.33, random_state=RANDOM_STATE)

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
no_epochs=1000
verbosity=1
validation_split=0.2


#optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)

for train, test in kfold.split(trainX, trainY):
    model =keras.Sequential([
        keras.layers.Dense(numFeatures),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
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

#"""We will check multiple classifier and use the one that works best for us"""
#
##combine df_num_scaled and df_cat_encoded
#
##features1 = sparse.vstack((df_num_scaled, df_cat_encoded)) 
#X = df_num_scaled
#X_test = X.values.astype(float)
# 
#
#
#clfs=[]
#folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)
#
#oof_preds = np.zeros((len(train), 1))
#test_preds = np.zeros((len(test), 1))
#no_epochs=10
#verbosity=1
#num_outputs=1
#num_features= len(all_features)
#validation_split=0.2
#
#for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
#    print("Current Fold: {}".format(fold_))
#    trn_x, trn_y = X[trn_, :], y[trn_]
#    val_x, val_y = X[val_, :], y[val_]
#
#
#    #clf = DEFINE MODEL HERE (Inside the for loop)
#    clf= keras.Sequential([
#            keras.layes.Dense(num_features, activation='sigmoid'),
#            keras.layers.Dropout(0.5),
#            keras.layers.Dense(256, activation='sigmoid'),
#            keras.layers.Dropout(0.5),
#            keras.layers.Dense(128, activation='sigmoid'),
#            keras.layers.Dropout(0.5),
#            keras.layers.Dense(56, activation='sigmoid'),
#            keras.layers.Dropout(0.5),
#            keras.layers.Dense(num_outputs, activation='sigmoid')
#            ])
#    
#    # FIT MODEL HERE (inside the for loop)
#    clf.fit(trn_x,trn_y, epochs=no_epochs, verbosity = verbosity, validation_split=validation_split)
#    
#    
#    #val_pred = GENERATE PREDICTIONS FOR VALIDATION DATA 
#    #test_fold_pred = GENERATE PREDICTIONS FOR TEST DATA
#    print("AUC = {}".format(metrics.roc_auc_score(val_y, val_pred))) 
#    oof_preds[val_, :] = val_pred.reshape((-1, 1))
#    test_preds += test_fold_pred.reshape((-1, 1))
#    clfs.append(clf)
#    
#test_preds /= NFOLDS
#
#
#
#roc_score = metrics.roc_auc_score(y, oof_preds.ravel()) 
#print("Overall AUC = {}".format(roc_score))
#
#print("Saving OOF predictions")
#oof_preds = pd.DataFrame(np.column_stack((train_ids, oof_preds.ravel())), columns=['id', 'target']) 
#oof_preds.to_csv('../kfolds/{}__{}.csv'.format(MODEL_NAME, str(roc_score)), index=False)
#
#print("Saving code to reproduce")
#shutil.copyfile(os.path.basename(__file__), '../model_source/{}__{}.py'.format(MODEL_NAME, str(roc_score)))
# 
#
#""" We can also do feature engineering from numerical features by 
#1. Transformation - converting them into log or exp  etc
##data['new_feature'] = data['feature'].apply(np.log)
##data['new_feature'] = data['feature'].apply(np.exp)
#2. Binning -
## for in num_cols:
#    dx= pd.DataFrame(data[c],columns=[c])
#    data[c+'_bin'] =pd.cut(data[c], bins=100, labels=False)
#3.Interactions - 
#pf=PolynomialFeatures(degree=2,interaction_only=False, include_bias=False)
#pf.fit(training_matrix)
#transformed_training = pf.transform(training_matrix)
#"""
#
#
#""" Selecting the best features out of all the features that were present and that we created 
#➢ Recursively eliminating the features
#➢ Based on model
#➢ Select top N features: SelectKBest
#➢ Selecting a percentile: SelectPercentile
#➢ Mutual information based
#➢ Chi2 based
#
#Note- This are some methods but in the end we should select the features that are improving our accuracy scores
#"""
#


#-----------------------------------performing predictions on new data------------------------------------------------

to_predict= original_data_test.copy()
to_predict_num= to_predict[num_features_used]
to_predict_cat= to_predict[cat_features_used]

#handle missing values
for column in to_predict_num.columns:
    to_predict_num[column].fillna(to_predict_num[column].mean(), inplace=True)
to_predict_num_scaled=sc.fit_transform(to_predict_num)


for column in to_predict_cat.columns:
    to_predict_cat[column].fillna(to_predict_cat[column].mode()[0], inplace=True)


to_predict_cat.info() #No missing values in the dataset now :)


####################### Encoding cat vars with label encoder
to_predict_cat_encoded=to_predict_cat.apply(LabelEncoder().fit_transform) 

to_predict_num_scaled=pd.DataFrame(to_predict_num_scaled).reset_index()

to_predict_cat_encoded=to_predict_cat_encoded.reset_index()

complete_pred_data= to_predict_num_scaled.merge(to_predict_cat_encoded, on='index')

complete_pred_data.drop('index',inplace=True,axis=1)
prediction_input=complete_pred_data.to_numpy()

predictions=model.predict(prediction_input)
sc=StandardScaler()
sc=sc.fit(target)
predictions_scaled=pd.DataFrame(sc.inverse_transform(predictions))
predictions_scaled.reset_index(inplace=True)
original_data_test.reset_index(inplace=True)
final_csv=original_data_test.merge(predictions_scaled, on='index')
final_csv=final_csv[['Id',0]]
final_csv = final_csv.rename(columns={0: 'SalePrice'})

final_csv.to_csv('Model1.csv',index=False)


