# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:55:58 2019

@author: ajseshad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ClassificationLib import FeatureScaling, LogReg, KNN, SVM, KernelSVM, NaiveBayes, DecisionTree, RandomForest

# dataset text can inherently contain commas and quotes
# Hence we use 'tab' as the delimiter - tsv file

def ImportData(filename):
    return pd.read_csv(filename, delimiter = '\t', quoting = 3)

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
def InitClean():
    nltk.download('stopwords')    

def CleanText(reviewText):
    import re
    #remove special characters and digits
    cleanReview = re.sub('[^a-zA-Z]', ' ', reviewText)
    #lower case
    cleanReview = cleanReview.lower()
    #remove preposition and other filler words that do not influence review type    
    reviewWords = cleanReview.split()
    #reviewWords = [word for word in reviewWords if word not in set(stopwords.words('english'))]
    #
    #Stemming - Convert words into their root form, i.e. loved => love to avoid a huge sparse matrix
    ps = PorterStemmer()
    reviewWords = [ps.stem(word) for word in reviewWords if word not in set(stopwords.words('english'))]
    #print(reviewWords)
    return ' '.join(reviewWords)

def CleanReview(dataset):
    InitClean()
    for i in range(0,len(dataset)):
        dataset['Review'][i] = CleanText(dataset['Review'][i])
    return dataset

# Create a bag of words model
def Vectorize(cleanDataset):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=1500)
    return cv.fit_transform(cleanDataset).toarray()

# -- Classification - Naive Bayes
def SplitDataSet(X,y):
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    return train_test_split(X, y, test_size = 0.2, random_state = 0) 

def ConfusionEval(cm):
    accuracy = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])
    precision = cm[1,1] / (cm[0,1] + cm[1,1])
    recall = cm[1,1] / (cm[1,0] + cm[1,1])
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1

dataset = ImportData('Restaurant_Reviews.tsv')
#review = CleanText(dataset['Review'][0])
cleanDataset = CleanReview(dataset)
bagOfWords = Vectorize(dataset['Review'])
X = bagOfWords # Independent variable
y = dataset.iloc[:,1].values

X_train, X_test, y_train, y_test = SplitDataSet(X,y)
fs = FeatureScaling()
#fs.scale(X_train, np.reshape(y_train, (-1,1)), X_test, np.reshape(y_test, (-1,1)))
fs.scale(X_train, y_train, X_test, y_test)
cm1, fs1, _ = NaiveBayes(fs)
accuracy1, precision1, recall1, f11 = ConfusionEval(cm1)

cm2, fs2, _ = LogReg(fs)
accuracy2, precision2, recall2, f12 = ConfusionEval(cm2)

cm3, fs3, _ = KNN(fs)
accuracy3, precision3, recall3, f13 = ConfusionEval(cm3)

cm4, fs4, _ = SVM(fs)
accuracy4, precision4, recall4, f14 = ConfusionEval(cm4)

cm5, fs5, _ = KernelSVM(fs)
accuracy5, precision5, recall5, f15 = ConfusionEval(cm5)

cm6, fs6, _ = DecisionTree(fs)
accuracy6, precision6, recall6, f16 = ConfusionEval(cm6)

cm7, fs7, _ = RandomForest(fs)
accuracy7, precision7, recall7, f17 = ConfusionEval(cm7)