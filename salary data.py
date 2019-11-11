# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:41:35 2019

@author: Adarsh
"""

import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Reading the Salary Data 
salary_train = pd.read_csv("E:/ADM/Excelr solutions/DS assignments/naive bayes/SalaryData_Train.csv")
salary_test = pd.read_csv("E:/ADM/Excelr solutions/DS assignments/naive bayes/SalaryData_Test.csv")
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])
	
colnames = salary_train.columns
len(colnames[0:13])
trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]
testX  = salary_test[colnames[0:13]]
testY  = salary_test[colnames[13]]

# use gaussian matrices because this dataset has numerical variables so it will give better accuracy,
sgnb = GaussianNB()

spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_gnb)
print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) # 80%

'''
# models accuracy = 80%

'''