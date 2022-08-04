# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:03:23 2018

@author: Sai Siddhanth
"""

#Multiple Linear Regression

#Importing libraries

#numpy used in making mathematical models
import numpy as np
# matplotlib.pyplot is used in graphs and plotting 
import matplotlib.pyplot as plt
#pandas is used for importing datasets
import pandas as pd


#importing the dataset
dataset = pd.read_csv('50_Startups.csv')

#creating matrix of features....independent variables
X = dataset.iloc[:, :-1].values
#creating matrix of dependent variables
y = dataset.iloc[:,4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X = X[:, 1:]


#Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
train_X,test_X,train_y,test_y = train_test_split(X,y , test_size = 0.2, random_state = 0)

"""#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.fit_transform(test_X)"""

#Fitting Mulitple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_X,train_y)

#Predicting the test set results
y_pred = regressor.predict(test_X)

#Building the optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis = 1)
X_opt = X[: , [0,1,2,3,4,5]]
regressor_OLS = sm.OLS( endog = y , exog = X_opt).fit()
regressor_OLS.summary()
#Removing 3
X_opt = X[: , [0,1,2,4,5]]
regressor_OLS = sm.OLS( endog = y , exog = X_opt).fit()
regressor_OLS.summary()
#Removing 2
X_opt = X[: , [0,1,4,5]]
regressor_OLS = sm.OLS( endog = y , exog = X_opt).fit()
regressor_OLS.summary()
#Removing 5
X_opt = X[: , [0,1,4]]
regressor_OLS = sm.OLS( endog = y , exog = X_opt).fit()
regressor_OLS.summary()








