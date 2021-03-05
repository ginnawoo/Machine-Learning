#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier #library that's already implemented for us in library called sklearn. No need to program it.
import joblib #importing module that contains loading and saving model that is trained already no need to start from scratch. 

# boba = pd.read_csv(r'C:\Users\ginna\OneDrive - University of South Florida\Desktop\OrlandoBobaShops.csv')
# X = boba.drop(columns=['boba_store_name']) #input will be capitalize
# y = boba['boba_store_name'] #output will be lowercase

# model = DecisionTreeClassifier() #creating object and set to decision tree.
# model.fit(X, y) #training the model to learn patterns by calling fit()- if model fits our definition for X and y.

model = joblib.load('boba-prediction.joblib')
prediction = model.predict([ [2020, 310, 4.0], [2012, 52, 4.0], [2013, 94, 4.0] ]) #model needs to make prediction using three data set.
prediction

