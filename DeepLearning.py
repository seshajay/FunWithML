# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 08:35:21 2019

@author: ajseshad
"""

# 1: Data preprocessing
# -*- coding: utf-8 -*-
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Making the confusion matrix
from sklearn.metrics import confusion_matrix

def ImportData(filename):
    # Importing the dataset
    dataset = pd.read_csv(filename)
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values
    return X,y

#Encode categorical independent variable
def EncodeDataSet(X)    :
    # Encoding categorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
    onehotencoder = OneHotEncoder(categorical_features = [1])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]
    return X


def SplitDataSet(X,y):
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size = 0.2, random_state = 0) 

# Simple Linear Regression does not need Feature Scaling, library will take care of it for us
# Feature Scaling
from sklearn.preprocessing import StandardScaler
class FeatureScaling:
    sc_X = None
    #sc_Y = None
    
    def __init__(self):
        self.sc_X = StandardScaler()
        #self.sc_y = StandardScaler()

    def scale(self, Xtrain, ytrain, Xtest, ytest):
        # Feature Scaling
        self.Xtrain = self.sc_X.fit_transform(Xtrain)
        self.ytrain = ytrain
        self.Xtest = self.sc_X.transform(Xtest)
        self.ytest = ytest

def LogReg(fs):
    # Fit simple Linear Regression to training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(fs.Xtrain, fs.ytrain)
    ypred = classifier.predict(fs.Xtest)
    cm = confusion_matrix(fs.ytest, ypred)
    return cm, fs, classifier

#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
def ANN(fs):
    #initialize ANN as a sequence of layers
    classifier = Sequential()
    #choose rectifier activation function in the hidden layers and 
    #sigmoid activation function in the outer layer
    # tip 1: No of nodes in hidden layer = avg (num of input nodes, no of output nodes)
    # tip 2: Or hidden layer nodes can be chosen based on parameter tuning (k-fold cross validation)
    #First hidden layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    #Second hidden layer --> input from previous layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    #output layer
    #if there are more than 2 categories: change output_dim and activation to softmax 
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    # Compile ANN
    # Optimizer: algorithm used to find the optimial set of weights. We use stochastic gradient descent. 
    #           adam is an efficient implementation for stochastic gradient descent
    # Loss: loss function within the stochastic gradient descent algorithm
    #       The loss is logarithmic. For binary outcome: binary_crossentropy. For more than 2 outcomes: categorical_crossentropy
    # metrics: model uses this to update the weights every iteration
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # batch size = 1 -- reinforcement learning else batch learning
    classifier.fit(fs.Xtrain, fs.ytrain, batch_size = 10, nb_epoch = 100)
    ypred = classifier.predict(fs.Xtest)
    ypred = (ypred > 0.5)
    cm = confusion_matrix(fs.ytest, ypred)
    return cm, fs, classifier




X, y = ImportData('Churn_Modelling.csv')
X = EncodeDataSet(X)
X_train, X_test, y_train, y_test = SplitDataSet(X,y)
fs = FeatureScaling()
fs.scale(X_train, np.reshape(y_train, (-1,1)), X_test, np.reshape(y_test, (-1,1)))
#fs.scale(X_train, y_train, X_test, y_test)
cm1, fs1, classifier1 = LogReg(fs)
cm2, fs2, classifier2 = ANN(fs)