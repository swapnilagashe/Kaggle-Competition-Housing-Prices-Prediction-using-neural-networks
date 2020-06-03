#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:48:35 2020

@author: swapnillagashe
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics

import pandas as pd


def RMSE_diff(x1,x2,y1,y2,y_transform):
    trainX1,testX1, trainY1, testY1 = train_test_split(x1.to_numpy().reshape(-1,1), y1.to_numpy(), test_size=0.2, random_state=42)
    trainX2,testX2, trainY2, testY2= train_test_split(x2.to_numpy().reshape(-1,1), y2.to_numpy(), test_size=0.2, random_state=42)
#    trainX1=trainX1.reshape(-1,1)
#    trainX2=trainX2.reshape(-1,1)
#    testX1=testX1.reshape(-1,1)
#    testX2=testX2.reshape(-1,1)

    
    model1=LinearRegression()
    model2=LinearRegression()
    model1.fit(trainX1,trainY1)
    model2.fit(trainX2,trainY2)
    y_pred1=model1.predict(testX1)
    rmse1=metrics.mean_squared_error(testY1,y_pred1)
    y_pred2=model2.predict(testX2)
    if y_transform==True:
        y_pred2=np.exp(y_pred2)
        testY2=np.exp(testY2)
    rmse2=metrics.mean_squared_error(testY2,y_pred2)
    diff=rmse2-rmse1
    return(rmse1,rmse2,diff)


    