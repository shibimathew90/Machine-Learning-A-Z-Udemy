#!/usr/bin/env python
# coding: utf-8

# In[4]:


import xgboost


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


dataset = pd.read_csv('C:/Udemy/Machine Learning A-Z Template Folder/Part 10 - Model Selection & Boosting/Section 49 - XGBoost/Churn_Modelling.csv')


# In[7]:


dataset.head()


# In[8]:



X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,-1].values


# In[9]:


X, y


# In[10]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()


# In[11]:


X = X[:,1:] 


# In[12]:


X.shape


# In[13]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 0)


# In[14]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[15]:


# Fitting XGboost to the training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


# In[16]:


y_pred = classifier.predict(X_test)


# In[17]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm


# In[20]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)


# In[21]:


accuracies.mean()


# In[22]:


accuracies.std()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




