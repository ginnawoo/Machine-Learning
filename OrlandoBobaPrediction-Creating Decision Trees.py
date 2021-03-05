#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier #library that's already implemented for us in library called sklearn. No need to program it.
from sklearn import tree

boba = pd.read_csv(r'C:\Users\ginna\OneDrive - University of South Florida\Desktop\OrlandoBobaShops.csv')
X = boba.drop(columns=['boba_store_name']) #input will be capitalize
y = boba['boba_store_name'] #output will be lowercase

model = DecisionTreeClassifier() #creating object and set to decision tree.
model.fit(X, y) #training the model to learn patterns by calling fit()- if model fits our definition for X and y.

#adding method to export decision tree in a graph format
tree.export_graphviz(model, out_file = 'boba-prediction.dot', # .dot file name
                            feature_names = ['year_launched', 'No_Reviews', 'Ratings'], #display the rules
                            class_names = sorted(y.unique()), #display class
                             label = 'all', #nodes has label
                             rounded = True, #rounded boxes
                             filled = True) #filled with color


# In[ ]:




