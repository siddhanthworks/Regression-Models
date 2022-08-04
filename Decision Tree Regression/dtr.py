# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 23:08:37 2018

@author: Sai Siddhanth
"""


#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

#Creating matrix of features and an dependent vector
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2:3].values

#Splitting the dataset into training set and test set
"""from sklearn.cross_validation import train_test_split
train_X,test_X,train_y,test_y = train_test_split(X,y , test_size = 0.2, random_state = 0)"""


#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.fit_transform(test_X)"""

#Fitting Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

#Predicting a new result with decision tree regression
regressor.predict(6.5)

#Visualzing the Regression results
X_grid = np.arange(min(X),max(X),0.1)
X_grid= X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict((X_grid)), color = 'blue')
plt.title('Level vs Salary (Decision Tree Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
