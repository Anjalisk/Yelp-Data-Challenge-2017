#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 20:16:01 2019

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
from nltk.tokenize import word_tokenize
nltk.download("punkt")
from collections import Counter
from nltk.classify import NaiveBayesClassifier

#importing the dataset
dataset = pd.read_csv('restreviewsclt.csv', delimiter = ',', dtype='object')


#Create a new dataset feature pos_neg which indicates whether a review is positive or negative based
#on the star rating.  If a review is 1, 2, or 3 stars, it is negative.  If a review is 4 or 5 stars, it
#is positive.

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

#Create a dataset with an equal number of positive and negative records to train the model

dataset_pos = dataset[dataset['pos_neg'] == 1]
dataset_neg = dataset[dataset['pos_neg'] == 0]
num_pos = dataset_pos.shape[0]
num_neg = dataset_neg.shape[0]

print(str(num_pos)+ ' '+ str(num_neg))

dataset_pos = dataset_pos.sample(n=min(num_pos,num_neg))
dataset_neg = dataset_neg.sample(n=min(num_pos,num_neg))    


#This is my dataset with equal positive and negative reviews
#dataset_eql = pd.concat([dataset_pos[:500], dataset_neg[:500]], ignore_index = True)
dataset_eql = pd.concat([dataset_pos, dataset_neg], ignore_index = True)

dataset_eql.shape[0]

dataset_eql[:5]

dataset_eql[-5:]

dataset_eql['text'][0]

useless_words=nltk.corpus.stopwords.words("english") + list(string.punctuation)

def bag_of_word_features_filtered(words):
    ps = PorterStemmer()
    return{
        ps.stem(word):1 for word in words if not word in useless_words}

corpus = []
negative_features = []
positive_features = []

#for i in range(0,1000):
#When you run this for the entire sample
for i in range(0,dataset_eql.shape[0]):
    #review_text = dataset_eql['text'][0]
    review_text = dataset_eql['text'][i]
    #remove everything not in a-z, A-Z, ., or \n
    review_text  = re.sub('[^a-zA-Z]',' ', review_text).lower()
    review_words = nltk.word_tokenize(review_text)

    dict_of_word_features_filtered = bag_of_word_features_filtered(review_words)
    if dataset_eql['pos_neg'][i] == 0:
        negative_features_element = []
        negative_features_element.append(dict_of_word_features_filtered)
        negative_features_element.append('neg')
        negative_features.append(negative_features_element) 
    elif dataset_eql['pos_neg'][i] == 1:
        positive_features_element = []
        positive_features_element.append(dict_of_word_features_filtered)
        positive_features_element.append('pos')
        positive_features.append(positive_features_element)
    #list_of_filtered_words = list(dict_of_word_features_filtered.keys())
    corpus = corpus + list(dict_of_word_features_filtered.keys())
    if i % 25000 == 0:
        print('i = ' + str(i))
        
print(positive_features[6])

print(negative_features[33])
#len(negative_features)

#list_of_filtered_words
corpus

from collections import Counter
word_counter = Counter(corpus)

len(word_counter)

len(corpus)

most_common_words = word_counter.most_common()[:5]

most_common_words

import matplotlib.pyplot as plt

sorted_word_counts = sorted(list(word_counter.values()),reverse=True)

plt.loglog(sorted_word_counts)
plt.ylabel("Frequency")
plt.xlabel("Word Rank")
plt.show()

plt.hist(sorted_word_counts, bins=50, log=True)
plt.show()

split = round(len(positive_features) * 0.8)

split

sentiment_classifier = NaiveBayesClassifier.train(positive_features[:split]+negative_features[:split])

nltk.classify.util.accuracy(sentiment_classifier,positive_features[:split]+negative_features[:split])*100

nltk.classify.util.accuracy(sentiment_classifier,positive_features[split:]+negative_features[split:])*100

len(positive_features)

sentiment_classifier.show_most_informative_features()

#PorterStemmer seems to create some odd stems such as fairli.  I suppose as long as it's consistent
#the word should still classify accurately. Lemmitization is supposed to maintain the same word.