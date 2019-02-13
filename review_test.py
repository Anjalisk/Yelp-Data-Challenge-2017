#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 19:44:41 2019

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


df_elite_users = pd.read_csv('elite_users.csv', sep=';')

df_elite_users.drop('Unnamed: 0', axis = 1, inplace = True)

df_elite_users.columns


df_elite_users.shape[0]


top_users = 1000
srted_top_elite_users = df_elite_users.sort_values('review_count', ascending = False).reset_index()

#srted_top_elite_users.head(10)

list_top_elite_users = srted_top_elite_users['user_id'][:top_users].tolist()

#len(list_top_elite_users)

#list_top_elite_users[:10]

#importing the charlotte restaurant reviews
dataset = pd.read_csv('restreviewsclt.csv', delimiter = ',', dtype='object')

#dataset.shape[0]

#dataset.columns


#Cleaning the text

#dataset.head(10)

corpus=[]
useless_words=nltk.corpus.stopwords.words("english")+list(string.punctuation)

for i in range(0,dataset.shape[0]):
    review = re.sub('[^a-zA-Z]',' ', dataset['text'][i])
    #convert to lower case
    review = review.lower()
    #separate the review into words
    review = review.split()
    #use stemming and remove english stop words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in useless_words]
    review = ' '.join(review)
    corpus.append(review)
    #if i % 100 == 0:
    #    print(i)

#corpus
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset['stars_x']

#X.shape

# Splitting the dataset into the Training set and Test set
a= LabelEncoder()
y= a.fit_transform(y.astype('str'))

#y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .80, test_size = .20)

# Fitting Logistic Regression to the Training set
model = LogisticRegression(multi_class='multinomial', solver = 'newton-cg', random_state = 0)
model.fit(X_train, y_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='multinomial',
          n_jobs=1, penalty='l2', random_state=0, solver='newton-cg',
          tol=0.0001, verbose=0, warm_start=False)

model_predict = model.predict(X_test)

print('5-fold Cross Validation Score = ' + str(cross_val_score(model, X, y, cv=5)))
print('Classification Report: ',)
print(classification_report(y_test, model_predict, target_names=['0', '1', '2', '3', '4']))
print('Confusion Matix: ',)
print(confusion_matrix(y_test, model_predict))
print('Accuracy Score: ',)
print(str(accuracy_score(y_test,model_predict)))

