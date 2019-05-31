# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 09:16:28 2019

@author: ajseshad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

### If we have limited data here, complex models are prone to overfitting
### Out of sample errors would be high but in sample errors will be low
### TODO: try k-fold cross validation
# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

linReg = None
linReg2 = None
level = 6.5

from sklearn.preprocessing import StandardScaler
class FeatureScaling:
    sc_X = None
    sc_Y = None
    
    def __init__(self):
        self.sc_X = StandardScaler()
        self.sc_y = StandardScaler()

    def scale(self, X, y):
        # Feature Scaling
        X = self.sc_X.fit_transform(X)
        y = self.sc_y.fit_transform(y)
        return (X, y)

def linearRegression(X, y):
    # Fitting Linear Regression to the dataset
    linReg = LinearRegression()
    linReg.fit(X, y)
    # Visualising the Linear Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, linReg.predict(X), color = 'blue')
    plt.title('Linear Regression')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

    # Predicting a new result with Linear Regression
    print("Linear Regression prediction for level %f is %d" %(linReg.predict(level), level))

def polyLinearRegression(X, y):
    # Fitting Polynomial Regression to the dataset
    from sklearn.preprocessing import PolynomialFeatures
    polyReg = PolynomialFeatures(degree = 4)
    X_poly = polyReg.fit_transform(X)
    polyReg.fit(X_poly, y)
    #This is analogous to working in Z-space where Z is the polynomial transform of X
    linReg2 = LinearRegression()
    linReg2.fit(X_poly, y)
    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, linReg2.predict(X_poly), color = 'blue')
    plt.title('Polynomial Regression')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

    # Visualising the Polynomial Regression results (for higher resolution and smoother curve)
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, linReg2.predict(polyReg.fit_transform(X_grid)), color = 'blue')
    plt.title('Truth or Bluff (Polynomial Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

    # Predicting a new result with Polynomial Regression
    print("Polynomial Linear Regression prediction for level %f is %d" %(linReg2.predict(polyReg.fit_transform(level)), level))

#Radial basis function    
def SVR(X,y):
    # Fitting the SVR Model to the dataset
    # Create your regressor here
    fs = FeatureScaling()
    X, y = fs.scale(X, y)
    from sklearn.svm import SVR
    regressor = SVR(kernel='rbf')
    regressor.fit(X,y)
    plot(X, y, regressor.predict(X), 'SVR', 'Position Level', 'Salary')
    
    # Predicting a new result with SVR
    # Predicting a new result
    y_pred = regressor.predict(fs.sc_X.transform(np.array([[level]])))
    #y_pred2 = regressor.predict(6.5)
    y_final = fs.sc_y.inverse_transform(y_pred)
    print("SVR prediction for level %f is %d" %(level, y_final))

def DecisionTree(X,y):
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X, y)
    # Model values are NOT continuous here. So need to visualize differently
    plot(X, y, regressor.predict(X), 'SVR', 'Position Level', 'Salary')
    plotHighRes(X, y, 'Decision Tree', 'Position Level', 'Salary', regressor)
    
    y_pred = regressor.predict(np.array([[level]]))
    print("Decision tree prediction for level %f is %d" %(level, y_pred))
 
def RandomForestRegression(X,y):
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
    regressor.fit(X,y)
    plotHighRes(X, y, 'Random Forest Regressor', 'Position Level', 'Salary', regressor)
    
    y_pred = regressor.predict(np.array([[level]]))
    print("Random forest prediction for level %f is %d" %(level, y_pred))

#TODO Model comparison - Adjusted R-Squared Value

def plot(X, y, yPred, title, xlabel, ylabel):
    plt.scatter(X, y, color = 'red')
    plt.plot(X, yPred, color = 'blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plotHighRes(X, y, title, xlabel, ylabel, regressor):
    # Visualising the Polynomial Regression results (for higher resolution and smoother curve)
    X_grid = np.arange(min(X), max(X), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
    plt.title('Truth or Bluff (Polynomial Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    
if __name__ == "__main__":
    # Importing the dataset
    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2].values
    y = np.reshape(y, (-1, 1))
    #linearRegression(X,y)
    #polyLinearRegression(X, y)
    #SVR(X,y)
    #DecisionTree(X, y)
    RandomForestRegression(X, y)