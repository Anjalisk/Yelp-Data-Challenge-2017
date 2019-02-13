#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 20:50:29 2019

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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#importing the dataset
dataset = pd.read_csv('bizreviewsclt.csv', delimiter = ',', dtype='object')

posneg = []

for i in range(0,dataset.shape[0]):
    if dataset['stars_x'][i] in (['1', '2', '3']):
        posneg.append(0)
    elif dataset['stars_x'][i] in (['4', '5']):
        posneg.append(1)
    else:
        #unexpected data
        posneg.append(-99)
dataset['pos_neg'] = posneg

#Create a dataset with an equal number of positive and negative records
dataset_pos = dataset[dataset['pos_neg']== 1]
dataset_neg = dataset[dataset['pos_neg']== 0]
num_pos = dataset_pos.shape[0]
num_neg = dataset_neg.shape[0]

dataset_pos = dataset_pos[:min(num_pos,num_neg)]
dataset_neg = dataset_neg[:min(num_pos,num_neg)]
#When you run this for the entire sample
#dataset_eql = pd.concat([dataset_pos, dataset_neg], ignore_index = True)
#This is my dataset with equal positive and negative reviews
dataset_eql = pd.concat([dataset_pos[:500], dataset_neg[:500]], ignore_index = True)

dataset_eql.shape[0]

corpus=[]
useless_words=nltk.corpus.stopwords.words("english")+list(string.punctuation)

#for i in range(0,1000):
#When you run this for the entire sample
for i in range(0,dataset_eql.shape[0]):
    review = re.sub('[^a-zA-Z]',' ', dataset_eql['text'][i])
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
y = dataset_eql['pos_neg']


# Splitting the dataset into the Training set and Test set
a=LabelEncoder()
y = a.fit_transform(y.astype('str'))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .80, test_size = .20)

# Fitting Gaussian Naive Bayes to the Training set

model = GaussianNB()
model.fit(X_train, y_train)
model_predict = model.predict(X_test)

print('Accuracy = ' + str(accuracy_score(y_test, model_predict)))
print('5-fold Cross Validation Score = ' + str(cross_val_score(model, X, y, cv=5)))
print('Classification Report: ',)
print(classification_report(y_test, model_predict, target_names=['0', '1']))
print('Confusion Matix: ',)
print(confusion_matrix(y_test, model_predict))