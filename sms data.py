# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:58:26 2019

@author: Adarsh
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set

sms_data = pd.read_csv("E:/ADM/Excelr solutions/DS assignments/naive bayes/sms_raw_NB.csv",encoding = "ISO-8859-1")


# cleaning data 
import re
stop_words = []
with open("E:/ADM/Excelr solutions/DS assignments/naive bayes/stop.txt") as f:
    stop_words = f.read()


# splitting the entire string by giving separator as "\n" to get list of 
# all stop words
stop_words = stop_words.split("\n")


def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))


# testing above function with sample text => removes punctuations, numbers

sms_data.text = sms_data.text.apply(cleaning_text)

# removing empty rows 
sms_data.shape
sms_data = sms_data.loc[sms_data.text != "",:]

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# TfidfTransformer
# Transform a count matrix to a normalized tf or tf-idf representation

# creating a matrix of token counts for the entire text document 

def split_into_words(i):
    return [word for word in i.split(" ")]


# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

sms_train,sms_test = train_test_split(sms_data,test_size=0.3)


# Preparing sms texts into word count matrix format 
sms_bow = CountVectorizer(analyzer=split_into_words).fit(sms_data.text)

# For all messages
all_sms_matrix = sms_bow.transform(sms_data.text)
all_sms_matrix.shape # (5559,6661)
# For training messages
train_sms_matrix = sms_bow.transform(sms_train.text)
train_sms_matrix.shape # (3891,6661)

# For testing messages
test_sms_matrix = sms_bow.transform(sms_test.text)
test_sms_matrix.shape # (1668,6661)

####### Without TFIDF matrices ########################
# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_sms_matrix,sms_train.type)
train_pred_m = classifier_mb.predict(train_sms_matrix)
accuracy_train_m = np.mean(train_pred_m==sms_train.type) # 98%

test_pred_m = classifier_mb.predict(test_sms_matrix)
accuracy_test_m = np.mean(test_pred_m==sms_test.type) # 96%
print(accuracy_train_m,accuracy_test_m)

#########################################################3

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_sms_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_sms_matrix)

train_tfidf.shape # (3891, 6661)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_sms_matrix)

test_tfidf.shape #  (1668, 6661)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf,sms_train.type)
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m==sms_train.type) # 96%

test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m==sms_test.type) # 95%
print(accuracy_train_m,accuracy_test_m)

# inplace of tfidf we can also use train_sms_matrix and test_sms_matrix instead of term inverse document frequency matrix 

'''
train accuracy=98%
test accuracy=96 using without TFIDF matrices.

'''
