#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:50:16 2020

@author: swapnillagashe
"""

"""lets do some feature engineering on housing data set"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from keras import Sequential
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
import keras
from helper_functions import RMSE_diff # I have made a seperate file with helper functions
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


os.chdir('/Users/DATA/Coding /Kaggle /house-prices-advanced-regression-techniques/')
original_data_train= pd.read_csv('train.csv')

original_data_test=pd.read_csv('test.csv')
X=original_data_train.drop(['SalePrice'], axis=1)
train=original_data_train
test=original_data_test.copy()

target=pd.DataFrame(train['SalePrice'])
y=target.values #returns a numpy array


all_features=train.columns.to_list()

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

f_to_remove= ['Alley', 'PoolQC', 'MiscFeature', 'Fence','FireplaceQu']

cat_vars_list= [e for e in cat_vars_list if e not in f_to_remove]

df_cat= df_cat[cat_vars_list]
df_cat.info()

##check how should we handle the missing values
#for column in df_cat.columns:
#    df_cat[column].fillna(df_cat[column].mode()[0], inplace=True)
#


df_num = train.drop(['Id']+non_num_vars, axis=1)
df_num.info()
num_vars_list=df_num.columns.to_list() #can use this command as well, will do it while refining the code
##lot frontage has few missing values, lets fill them with the average values
#for column in df_num.columns:
#    df_num[column].fillna(df_num[column].mean(), inplace=True)



"""so now we have 2 dfs, one for cat data and one for num data"""


"""Feature engg steps
combine both train and test data first for features other than target var
1. Convert sale price to log scale- RMSE decreases on applying any model - can check by experimenting
2. Check corellation betw features and target var
3. Box plots are good way to check for outliers
(If the box is pushed to one side and some values are far away from the box then it’s a clear indication of outliers)"""
#feature1='GrLivArea'
#feature2='LotArea'
#feature2=feature1
#x1=train[feature1]
#x2=x1
#x2=train[feature2]
#y1=target
#y2=np.log(y1)
#y_transform=False
#print(RMSE_diff(x1,y1,x2,y2,y_transform)) # 
## sample model
## Generate generalization metrics
#scores = model.evaluate(inputs[test], targets[test], verbose=0)

#correlation, for  now we will perform only on train data

data_corr = train.select_dtypes(np.number) #data with numerical values
data_corr.head()
corr=data_corr.corr()
corr_high=corr[corr>0.8]
corr.sort_values(['SalePrice'], ascending=False, inplace=True)
corr['SalePrice']
 #check features that are highly correlated
#features that have very less correlation with target variable may decrease the accuracy of the model
#Lets look at some features that are highly correlated with target variable

#----------------------------GrLivArea--------------------------------------------------
#1. GrLivArea -Above grade (ground) living area square feet
#lets check for any outliers by making a boxplot
train.GrLivArea.describe()
sns.boxplot(train['GrLivArea']) #we can see that the box is pushed to one side and some values are far away --> outliers 
plt.boxplot(train['GrLivArea']) #we can see that the box is pushed to one side and some values are far away --> outliers 
#we will check rmse diff after eliminating outliers > 4500
feature='GrLivArea'
target_var='SalePrice'
x1=train[feature]
x2=train[train[feature1]<4500][feature] #only 2 values are there
y1=train[target_var]
y2=train[train[feature1]<4500][target_var]
y_transform=False
print(RMSE_diff(x1,x2,y1,y2,y_transform)) #we saw that rmse decrease 


"""Observations
1. RMSE increases on changing y to log scale but an online article shows reverse results --> will try this with all the features combined in the final model
2. RMSE decreased after removing outliers(greater than 4500) from GrLivArea feature, we can also remove values greater than 4000 but that might generate overfitting --> lets try this later
3. Check the correlation matrix for features that have high correlation with each other- we should remove one of them as including both will inflate variance of other features([GarageYrBuilt, YearBuilt], [1stFlrSF and TotalBsmtSF 0.819] [TotRmsAbvGrd, GrLivArea], [GarageCars, GarageArea]
4. We can create features from target var like price per sq ft area
5. I think we should convert pool area feature to pool present/absent feature as pool area doesnt matter much but having pool or not matters)


"""
#-----------------------null values
plt.figure(figsize=(12, 6))
sns.heatmap(train.isnull())
plt.show()
train.info()

cat_drop_cols = ['Alley','FireplaceQu','PoolQC','MiscFeature','Fence'] #very less data points 
num_drop_cols = ['1stFlrSF','GarageArea','GarageYrBlt','MiscVal','3SsnPorch','PoolArea'] #less data points or having high corr with some other feature (remove one of the feature)

train.drop(cat_drop_cols, axis=1, inplace=True)
train.drop(num_drop_cols, axis=1, inplace=True)
df_cat.drop(cat_drop_cols, axis=1, inplace=True)
df_num.drop(num_drop_cols, axis=1, inplace=True)

#create a feature price per sq ft
train['price_per_sqft']= train['SalePrice']/train['LotArea']
#we can also assign the mean value according to the neighbourhood
train['price_per_sqft'] = train.groupby('Neighborhood')['price_per_sqft'].transform(lambda x: x.median())
df_num['price_per_sqft']=train['price_per_sqft']

#Then we make a dictionary with the prices for each neighborhood: because we will have to add this feature to test data as well
d = {}
for indice_fila, x_train in train.iterrows():
    d.update({x_train['Neighborhood']:x_train['price_per_sqft']})
#And finally, we created the feature in the test dataset
test['price_per_sqft'] = 0.00
for indice, x_test in test.iterrows():
    test.loc[test.index == indice ,'price_per_sqft'] = d[x_test['Neighborhood']]
    
#---------------handling Outliers
for x in range(df_num.shape[1]):
    plt.figure(figsize=(12, 6))
    plt.boxplot(df_num.iloc[:,x]) #we can see that the box is pushed to one side and some values are far away --> outliers 
    plt.title(str(df_num.columns[x]))
    plt.show()
    
features_with_outliers_to_handle =['ScreenPorch','EnclosedPorch', 'OpenPorchSF','WoodDeckSF', 'GrLivArea','LowQualFinSF', 'TotalBsmtSF','BsmtFinSF1','LotArea']

for x in range(len(features_with_outliers_to_handle)):
    
    mean = df_num[features_with_outliers_to_handle[x]].mean()
    sd = df_num[features_with_outliers_to_handle[x]].std()
    lower_limit = mean - 3*sd
    upper_limit = mean + 3*sd
    df_num[features_with_outliers_to_handle[x]+'_no_outlier'] = pd.Series([min(max(a,lower_limit), upper_limit) for a in df_num[features_with_outliers_to_handle[x]]])

df_num.drop(features_with_outliers_to_handle, axis=1, inplace=True)


#check whether outliers have minimised
plt.boxplot(df_num['LotArea']) #we can see that the box is pushed to one side and some values are far away --> outliers 
plt.boxplot(df_num['LotArea_no_outlier']) #better than before, too much processing might lead to overfitting
df_num.info() #MasVnrArea and LotFrontage have some null values, we wll fill them with average values
for column in df_num.columns:
    df_num[column].fillna(df_num[column].mean(), inplace=True)
    
    
#----------------checking categorical vars
df_cat['MoSold'] = df_num['MoSold']
df_num[['OverallQual','OverallCond']]=df_cat[['OverallQual','OverallCond']]
df_cat.drop(['OverallQual','OverallCond'],inplace=True, axis=1)
df_num.drop(['MoSold'], inplace=True,axis=1)
df_cat['MoSold'] = df_cat['MoSold'].apply(str)
df_cat['MSSubClass']=df_cat['MSSubClass'].astype(str)
df_cat_encoded = pd.get_dummies(df_cat).reset_index(drop=True)

###removing columns which have more than 99.94 percent zeros present
#overfit = []
#for i in final_features.columns:
#    counts = final_features[i].value_counts()
#    zeros = counts.iloc[0]
#    if zeros / len(X) * 100 > 99.94:
#       overfit.append(i)
#overfit = list(overfit)
#overfit.append('MSZoning_C (all)')
#overfit
#X = X.drop(overfit, axis=1).copy()
df_cat_encoded.reset_index(inplace=True)
df_num_scaled
final_train_data=df_cat_encoded.merge(df_num_scaled, on='index')
