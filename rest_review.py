#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 20:23:51 2019

@author: Anjalikhushalani
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


yelp = pd.read_csv('restreviewsclt.csv', delimiter = ',', dtype='object')

yelp.info()

yelp.drop(['business_id','user_id','date','date','address','hours','categories',
             'is_open','latitude','longitude','name','neighborhood','postal_code'], axis=1, inplace=True)

yelp['text length'] = yelp['text'].apply(len)
yelp.head()

import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

g = sns.FacetGrid(data=yelp, col='stars_x')
g.map(plt.hist, 'text length', bins=50)



sns.boxplot(x='stars_x', y='text length', data=yelp)

yelp['cool'].astype(str).astype(int)
#yelp['stars_y'].astype(str).astype(int)

yelp['funny'].astype(str).astype(int)


yelp['useful'].astype(str).astype(int)

yelp['stars_x'].astype(str).astype(int)
