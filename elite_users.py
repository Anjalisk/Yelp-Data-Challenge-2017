#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 19:56:25 2019

@author: Anjalikhushalani
"""

#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


elite_users_df = pd.read_csv('elite_users.csv', sep=';')

elite_users_df.columns

elite_users_df.drop('Unnamed: 0', axis = 1, inplace = True)

nc_rest_rvws_df = pd.read_csv('restreviewsclt.csv')

nc_rest_rvws_df.columns

nc_rest_rvws_df.shape[0]


#Find Restaurant reviews from elite users

#List of elite user ids
list_elite_users = elite_users_df['user_id'].unique().tolist()

nc_rest_elite_rvws_df = nc_rest_rvws_df.loc[nc_rest_rvws_df['user_id'].isin(list_elite_users)].reset_index()

nc_rest_elite_rvws_df.shape[0]


nc_rest_elite_rvws_df.sort_values('review_count', ascending = False, inplace = True)

documents = nc_rest_elite_rvws_df['text'][0:3000]

df = nc_rest_elite_rvws_df[:3000]

len(documents)


documents_train, documents_test= train_test_split(documents, test_size=0.2)

vectorizer = TfidfVectorizer(analyzer = 'word', stop_words = 'english', 
                             lowercase = True, max_features = 5000
                            )
# Train the model with my training data
documents_train_vec = vectorizer.fit_transform(documents_train).toarray()
# Get the vocab of my tfidf
words = vectorizer.get_feature_names()
# Use the trained model to transform all the reviews
documents_vec = vectorizer.transform(documents).toarray()

wcss = []
for i in range(1,11):         #Testing up to 10 clusters
    kmeans = KMeans(n_clusters = i, init='k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(documents_vec)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

km_clf = KMeans(verbose = 0,n_clusters = 3)
km_clf.fit(documents_train_vec)

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=3, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)

cluster = km_clf.predict(documents_vec)

km_clf.cluster_centers_

cluster_top_features = list()

for i in range(km_clf.n_clusters):
    cluster_top_features.append(np.argsort(km_clf.cluster_centers_[i])[::-1][:20])
for num, centroid in enumerate(cluster_top_features):
    print ('%d: %s' % (num, ", ".join(words[i] for i in centroid)))


for i in range(km_clf.n_clusters):
    sub_cluster = np.arange(0, cluster.shape[0])[cluster == i]
    sample = np.random.choice(sub_cluster, 1)
    print("The cluster is %d." % (i+1))
    print("The star is: %s stars." % df['stars_x'].iloc[sample[0]])
    print("The review is:\n%s.\n" % df['text'].iloc[sample[0]])
    
    
useless_words=nltk.corpus.stopwords.words("english")+list(string.punctuation)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

list_k_3_c1= []
list_k_3_c2= []
list_k_3_c3= []

for i in range(km_clf.n_clusters):
    sub_cluster = np.arange(0, cluster.shape[0])[cluster == i]
    for j in range(0,len(sub_cluster)):
        if i == 0:
            list_k_3_c1.append(df['text'].iloc[sub_cluster[j]])
        elif i == 1:
            list_k_3_c2.append(df['text'].iloc[sub_cluster[j]])
        elif i == 2:
            list_k_3_c3.append(df['text'].iloc[sub_cluster[j]])

list_k_3_c1

list_k_3_c2

list_k_3_c3

#clean cluster 1 k = 3
#One sub-cluster
corpus=[]
for i in range(0,len(list_k_3_c1)):
    review = re.sub('[^a-zA-Z]',' ', list_k_3_c1[i])
    #convert to lower case
    review = review.lower()
    #separate the review into words
    review = review.split()
    #use stemming and remove english stop words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in useless_words]
    review = ' '.join(review)
    corpus.append(review)
    
no_features = 1000
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', 
                                learning_offset=50.,random_state=0).fit(tf)
no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)


#clean cluster 2 k = 3
#One sub-cluster
corpus=[]
for i in range(0,len(list_k_3_c2)):
    review = re.sub('[^a-zA-Z]',' ', list_k_3_c2[i])
    #convert to lower case
    review = review.lower()
    #separate the review into words
    review = review.split()
    #use stemming and remove english stop words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in useless_words]
    review = ' '.join(review)
    corpus.append(review)
    
no_features = 1000
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', 
                                learning_offset=50.,random_state=0).fit(tf)
no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)


#clean cluster 3 k = 3
#One sub-cluster
corpus=[]
for i in range(0,len(list_k_3_c3)):
    review = re.sub('[^a-zA-Z]',' ', list_k_3_c3[i])
    #convert to lower case
    review = review.lower()
    #separate the review into words
    review = review.split()
    #use stemming and remove english stop words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in useless_words]
    review = ' '.join(review)
    corpus.append(review)
    
no_features = 1000
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', 
                                learning_offset=50.,random_state=0).fit(tf)
no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)



km_4_clf = KMeans(verbose = 0,n_clusters = 4)
km_4_clf.fit(documents_train_vec)

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=4, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)

cluster = km_4_clf.predict(documents_vec)

cluster_top_features = list()
for i in range(km_4_clf.n_clusters):
    cluster_top_features.append(np.argsort(km_4_clf.cluster_centers_[i])[::-1][:20])
for num, centroid in enumerate(cluster_top_features):
    print ('%d: %s' % (num, ", ".join(words[i] for i in centroid)))

km_4_clf.cluster_centers_


for i in range(km_4_clf.n_clusters):
    sub_cluster = np.arange(0, cluster.shape[0])[cluster == i]
    sample = np.random.choice(sub_cluster, 1)
    print("The cluster is %d." % (i+1))
    print("The star is: %s stars." % df['stars_x'].iloc[sample[0]])
    print("The review is:\n%s.\n" % df['text'].iloc[sample[0]])
    
    
list_k_4_c1= []
list_k_4_c2= []
list_k_4_c3= []
list_k_4_c4= []

for i in range(km_4_clf.n_clusters):
    sub_cluster = np.arange(0, cluster.shape[0])[cluster == i]
    for j in range(0,len(sub_cluster)):
        if i == 0:
            list_k_4_c1.append(df['text'].iloc[sub_cluster[j]])
        elif i == 1:
            list_k_4_c2.append(df['text'].iloc[sub_cluster[j]])
        elif i == 2:
            list_k_4_c3.append(df['text'].iloc[sub_cluster[j]])
        elif i == 3:
            list_k_4_c4.append(df['text'].iloc[sub_cluster[j]])

#clean cluster 1 k = 4
#One sub-cluster
corpus=[]
for i in range(0,len(list_k_4_c1)):
    review = re.sub('[^a-zA-Z]',' ', list_k_4_c1[i])
    #convert to lower case
    review = review.lower()
    #separate the review into words
    review = review.split()
    #use stemming and remove english stop words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in useless_words]
    review = ' '.join(review)
    corpus.append(review)
    
no_features = 1000
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', 
                                learning_offset=50.,random_state=0).fit(tf)
no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)


#clean cluster 2 k = 4
#One sub-cluster
corpus=[]
for i in range(0,len(list_k_4_c2)):
    review = re.sub('[^a-zA-Z]',' ', list_k_4_c2[i])
    #convert to lower case
    review = review.lower()
    #separate the review into words
    review = review.split()
    #use stemming and remove english stop words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in useless_words]
    review = ' '.join(review)
    corpus.append(review)
    
no_features = 1000
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', 
                                learning_offset=50.,random_state=0).fit(tf)
no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)


#clean cluster 3 k = 4
#One sub-cluster
corpus=[]
for i in range(0,len(list_k_4_c3)):
    review = re.sub('[^a-zA-Z]',' ', list_k_4_c3[i])
    #convert to lower case
    review = review.lower()
    #separate the review into words
    review = review.split()
    #use stemming and remove english stop words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in useless_words]
    review = ' '.join(review)
    corpus.append(review)
    
no_features = 1000
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', 
                                learning_offset=50.,random_state=0).fit(tf)
no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)


#clean cluster 4 k = 4
#One sub-cluster
corpus=[]
for i in range(0,len(list_k_4_c4)):
    review = re.sub('[^a-zA-Z]',' ', list_k_4_c4[i])
    #convert to lower case
    review = review.lower()
    #separate the review into words
    review = review.split()
    #use stemming and remove english stop words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in useless_words]
    review = ' '.join(review)
    corpus.append(review)
    
no_features = 1000
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', 
                                learning_offset=50.,random_state=0).fit(tf)
no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)


km_5_clf = KMeans(verbose = 0,n_clusters = 5)
km_5_clf.fit(documents_train_vec)

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=5, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)

cluster = km_5_clf.predict(documents_vec)

km_5_clf.cluster_centers_


cluster_top_features = list()
for i in range(km_5_clf.n_clusters):
    cluster_top_features.append(np.argsort(km_5_clf.cluster_centers_[i])[::-1][:20])
for num, centroid in enumerate(cluster_top_features):
    print ('%d: %s' % (num, ", ".join(words[i] for i in centroid)))
    
    
for i in range(km_5_clf.n_clusters):
    sub_cluster = np.arange(0, cluster.shape[0])[cluster == i]
    sample = np.random.choice(sub_cluster, 1)
    print("The cluster is %d." % (i+1))
    print("The star is: %s stars." % df['stars_x'].iloc[sample[0]])
    print("The review is:\n%s.\n" % df['text'].iloc[sample[0]])
    
    
list_k_5_c1= []
list_k_5_c2= []
list_k_5_c3= []
list_k_5_c4= []
list_k_5_c5= []

for i in range(km_5_clf.n_clusters):
    sub_cluster = np.arange(0, cluster.shape[0])[cluster == i]
    for j in range(0,len(sub_cluster)):
        if i == 0:
            list_k_5_c1.append(df['text'].iloc[sub_cluster[j]])
        elif i == 1:
            list_k_5_c2.append(df['text'].iloc[sub_cluster[j]])
        elif i == 2:
            list_k_5_c3.append(df['text'].iloc[sub_cluster[j]])
        elif i == 3:
            list_k_5_c4.append(df['text'].iloc[sub_cluster[j]])
        elif i == 4:
            list_k_5_c5.append(df['text'].iloc[sub_cluster[j]])
            
#clean cluster 1 k = 5
#One sub-cluster
corpus=[]
for i in range(0,len(list_k_5_c1)):
    review = re.sub('[^a-zA-Z]',' ', list_k_5_c1[i])
    #convert to lower case
    review = review.lower()
    #separate the review into words
    review = review.split()
    #use stemming and remove english stop words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in useless_words]
    review = ' '.join(review)
    corpus.append(review)
    
no_features = 1000
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', 
                                learning_offset=50.,random_state=0).fit(tf)
no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)


#clean cluster 2 k = 5
#One sub-cluster
corpus=[]
for i in range(0,len(list_k_5_c2)):
    review = re.sub('[^a-zA-Z]',' ', list_k_5_c2[i])
    #convert to lower case
    review = review.lower()
    #separate the review into words
    review = review.split()
    #use stemming and remove english stop words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in useless_words]
    review = ' '.join(review)
    corpus.append(review)
    
no_features = 1000
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', 
                                learning_offset=50.,random_state=0).fit(tf)
no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)


#clean cluster 3 k = 5
#One sub-cluster
corpus=[]
for i in range(0,len(list_k_5_c3)):
    review = re.sub('[^a-zA-Z]',' ', list_k_5_c3[i])
    #convert to lower case
    review = review.lower()
    #separate the review into words
    review = review.split()
    #use stemming and remove english stop words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in useless_words]
    review = ' '.join(review)
    corpus.append(review)
    
no_features = 1000
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', 
                                learning_offset=50.,random_state=0).fit(tf)
no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)


#clean cluster 4 k = 5
#One sub-cluster
corpus=[]
for i in range(0,len(list_k_5_c4)):
    review = re.sub('[^a-zA-Z]',' ', list_k_5_c4[i])
    #convert to lower case
    review = review.lower()
    #separate the review into words
    review = review.split()
    #use stemming and remove english stop words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in useless_words]
    review = ' '.join(review)
    corpus.append(review)
    
no_features = 1000
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', 
                                learning_offset=50.,random_state=0).fit(tf)
no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)


#clean cluster 5 k = 5
#One sub-cluster
corpus=[]
for i in range(0,len(list_k_5_c5)):
    review = re.sub('[^a-zA-Z]',' ', list_k_5_c5[i])
    #convert to lower case
    review = review.lower()
    #separate the review into words
    review = review.split()
    #use stemming and remove english stop words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in useless_words]
    review = ' '.join(review)
    corpus.append(review)
    
no_features = 1000
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', 
                                learning_offset=50.,random_state=0).fit(tf)
no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)


km_7_clf = KMeans(verbose = 0,n_clusters = 7)
km_7_clf.fit(documents_train_vec)

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=7, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)

cluster = km_7_clf.predict(documents_vec)

km_7_clf.cluster_centers_


cluster_top_features = list()
for i in range(km_7_clf.n_clusters):
    cluster_top_features.append(np.argsort(km_7_clf.cluster_centers_[i])[::-1][:20])
for num, centroid in enumerate(cluster_top_features):
    print ('%d: %s' % (num, ", ".join(words[i] for i in centroid)))
    
    
