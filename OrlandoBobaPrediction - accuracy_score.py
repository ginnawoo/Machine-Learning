#!/usr/bin/env python
# coding: utf-8

# In[158]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier #library that's already implemented for us in library called sklearn. No need to program it.
from sklearn.model_selection import train_test_split #80/20: use 20% as test and rest for actual training
from sklearn.metrics import accuracy_score

boba = pd.read_csv(r'C:\Users\ginna\OneDrive - University of South Florida\Desktop\OrlandoBobaShops.csv')
X = boba.drop(columns=['boba_store_name']) #input will be capitalize
y = boba['boba_store_name'] #output will be lowercase
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier() #creating object and set to decision tree.
model.fit(X_train, y_train) #training the model to learn patterns by calling fit()- if model fits our definition for X and y.

prediction = model.predict(X_test) #model needs to make prediction using three data set.

score = accuracy_score(y_test, prediction)
score


# In[109]:


import joblib #importing module that contains loading and saving model that is trained already no need to start from scratch. 


# In[ ]:




