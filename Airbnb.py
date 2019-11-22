#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:09:13 2019

@author: nageshsinghchauhan
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/nageshsinghchauhan/Downloads/ML/kaggle/airbnb/AB_NYC_2019.csv")

##Data cleaning
#remove duplicates if any
data.duplicated().sum()
data.drop_duplicates(inplace=True)

#Drop unneceassry columns
data.drop(['name','id', 'host_id','last_review'], axis = 1, inplace = True)

#Drop NaN values
#Since reviews_per_month column has many NaN values so lets replace them with 0 instead of removing
data.fillna({'reviews_per_month' : 0}, inplace = True)
#remove Nan from rest of the column
data.isnull().sum() #to check for null values in each column
data.dropna(how = 'any', inplace = True)

#Get correlation between different variables
corr = data.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
data.columns

#Regression analysis to predict the price
#drop unneceassy columns
data.drop(['host_name','latitude','longitude','neighbourhood','number_of_reviews','reviews_per_month'], axis = 1, inplace = True)
X = data.iloc[:,[0,1,3,4,5]]
y = data['price']


#Label encoding
X = pd.get_dummies(X, prefix=['neighbourhood_group', 'room_type'], drop_first=True)

#splitting the dataset into test and training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.2, random_state= 0)

#Prepare a Linear Regression Model
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
#R2 score
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred)*100)
#Error
error_diff = pd.DataFrame({'Actual Values': np.array(y_test).flatten(), 'Predicted Values': y_pred.flatten()})
#Visualize the error
df1 = error_diff.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


#Prepairng a Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.1,random_state=105)
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(X_train,y_train)
y_pred=DTree.predict(X_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred)*100)
#Error
error_diff = pd.DataFrame({'Actual Values': np.array(y_test).flatten(), 'Predicted Values': y_pred.flatten()})
#Visualize the error
df1 = error_diff.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

