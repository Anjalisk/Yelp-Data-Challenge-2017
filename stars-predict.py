#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 20:53:16 2019

@author: Anjalikhushalani
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#importing the dataset
dataset = pd.read_csv('bizreviewsclt.csv', delimiter = ',', dtype='object')

dataset.shape[0]

#Cleaning the text

corpus=[]
useless_words=nltk.corpus.stopwords.words("english")+list(string.punctuation)

for i in range(0,1000):
#When you run this for the entire sample
#for i in range(0,dataset.shape[0]):
    review = re.sub('[^a-zA-Z]',' ', dataset['text'][i])
    #convert to lower case
    review = review.lower()
    #separate the review into words
    review = review.split()
    #use stemming and remove english stop words
    ps = PorterStemmer()
    #review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = [ps.stem(word) for word in review if not word in useless_words]
    review = ' '.join(review)
    corpus.append(review)
    
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
#When you run this for the entire sample
#y = dataset['stars_x']
y = dataset['stars_x'][:1000]

# Splitting the dataset into the Training set and Test set
a=LabelEncoder()
y = a.fit_transform(y.astype('str'))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .80, test_size = .20)

# Fitting Logistic Regression to the Training set
model = LogisticRegression(multi_class='multinomial', solver = 'newton-cg', random_state = 0)
model.fit(X_train, y_train)
model_predict = model.predict(X_test)

print('5-fold Cross Validation Score = ' + str(cross_val_score(model, X, y, cv=5)))
print('Classification Report: ',)
print(classification_report(y_test, model_predict, target_names=['0', '1', '2', '3', '4']))
print('Confusion Matix: ',)
print(confusion_matrix(y_test, model_predict))