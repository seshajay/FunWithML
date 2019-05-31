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
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.iloc[:, [2,3]].values
    y = dataset.iloc[:, 4].values
    return X,y

def SplitDataSet(X,y):
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    return train_test_split(X, y, test_size = 0.25, random_state = 0) 

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

def PlotClassify(X_train, y_train, classifier, title):
    # Visualising the Training set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

def LogReg(fs):
    # Fit simple Linear Regression to training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(fs.Xtrain, fs.ytrain)
    ypred = classifier.predict(fs.Xtest)
    cm = confusion_matrix(fs.ytest, ypred)
    return cm, fs, classifier

def KNN(fs):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(fs.Xtrain, fs.ytrain)
    ypred = classifier.predict(fs.Xtest)
    cm = confusion_matrix(fs.ytest, ypred)
    return cm, fs, classifier

def SVM(fs):
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(fs.Xtrain, fs.ytrain)
    ypred = classifier.predict(fs.Xtest)
    cm = confusion_matrix(fs.ytest, ypred)
    return cm, fs, classifier

def KernelSVM(fs):
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0) #rbf is the default
    classifier.fit(fs.Xtrain, fs.ytrain)
    ypred = classifier.predict(fs.Xtest)
    cm = confusion_matrix(fs.ytest, ypred)
    return cm, fs, classifier

def NaiveBayes(fs):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(fs.Xtrain, fs.ytrain)
    ypred = classifier.predict(fs.Xtest)
    cm = confusion_matrix(fs.ytest, ypred)
    return cm, fs, classifier

# This algorithm is not based on euclidean distance
# hence feature scaling is not needed. However we increased
# the resolution and hence FS will make computations faster
# We use the criterion as entropy - more homogenity after a split 
# less is the entropy.
def DecisionTree(fs):
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='entropy', random_state = 0)
    classifier.fit(fs.Xtrain, fs.ytrain)
    ypred = classifier.predict(fs.Xtest)
    # Making the confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(fs.ytest, ypred)
    return cm, fs, classifier

def RandomForest(fs):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
    classifier.fit(fs.Xtrain, fs.ytrain)
    ypred = classifier.predict(fs.Xtest)
    # Making the confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(fs.ytest, ypred)
    return cm, fs, classifier

# Accuracy paradox - can't rely on error rate alone
# TODO: model comparison - CAP(Cumulative accuracy profile)

#regressor = LinearRegression()
#regressor.fit(X_train, y_train)

# Predict the test results
#y_pred = regressor.predict(X_test)

X, y = ImportData('Social_Network_Ads.csv')
X_train, X_test, y_train, y_test = SplitDataSet(X,y)
fs = FeatureScaling()
#fs.scale(X_train, np.reshape(y_train, (-1,1)), X_test, np.reshape(y_test, (-1,1)))
fs.scale(X_train, y_train, X_test, y_test)
cm1, fs1, classifier1 = LogReg(fs)
PlotClassify(fs1.Xtrain, fs1.ytrain, classifier1, 'Logistic Regression (Training set)')

cm2, fs2, classifier2 = KNN(fs)
PlotClassify(fs2.Xtrain, fs2.ytrain, classifier2, 'KNN (Training set)')

cm3, fs3, classifier3 = SVM(fs)
PlotClassify(fs3.Xtrain, fs3.ytrain, classifier3, 'SVM (Training set)')

cm4, fs4, classifier4 = KernelSVM(fs)
PlotClassify(fs4.Xtrain, fs4.ytrain, classifier4, 'Kernel SVM (Training set)')

cm5, fs5, classifier5 = NaiveBayes(fs)
PlotClassify(fs5.Xtrain, fs5.ytrain, classifier5, 'Naive Bayes (Training set)')

cm6, fs6, classifier6 = DecisionTree(fs)
PlotClassify(fs6.Xtrain, fs6.ytrain, classifier6, 'Decision Tree (Training set)')

cm7, fs7, classifier7 = RandomForest(fs)
PlotClassify(fs7.Xtrain, fs7.ytrain, classifier7, 'Random Forest (Training set)')