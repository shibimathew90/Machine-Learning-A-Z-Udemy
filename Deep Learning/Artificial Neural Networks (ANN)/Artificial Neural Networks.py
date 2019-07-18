#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


dataset = pd.read_csv('C:/Udemy/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Churn_Modelling.csv')


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


# In[13]:


X = X[:,1:]                  # removing the first column created after OneHotEncoding to avoid the Dummy variable trap


# In[14]:


X.shape


# In[ ]:





# In[15]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 0)


# In[16]:


X_train.shape, X_test.shape


# In[17]:


y_train.shape, y_test.shape


# In[18]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[19]:


X_train.shape, X_test.shape


# In[20]:


X_train, X_test


# In[21]:


# Part 2 - Importing libraries and packages


# In[22]:


import keras


# In[24]:


from keras.models import Sequential    # Sequential is required to initialize our neural network
from keras.layers import Dense           # Dense is required to build the layers of our ANN


# In[26]:


#Initializing the ANN classifier- i.e. defining the sequence of layers
classifier = Sequential()


# In[27]:


# Adding the input layer and the first hidden layer


# In[35]:


# a general rule to select the number of nodes in the hidden layer is to take average of input and output parameters.
# Hence units = (11+1)/2  = 6....Here the units is the hidden layer 
classifier.add(Dense(units = 6, kernel_initializer= 'uniform', activation = 'relu', input_dim = 11))     


# In[37]:


# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))


# In[38]:


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer= 'uniform', activation = 'sigmoid'))


# If we have dependent variables with more than 2 categories say 3, then units = 3 and activation = 'softmax'
# Softmax activation function is a Sigmoid function but applied to a dependent variable with more than 2 categories.


# In[39]:


# If the dependent variable has 2 categories then the loss function used is binary_crossentropy and if it has more than
# 2 categoris then the loss function used in categorical_crossentropy

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[40]:


# Fitting the ANN to training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch = 100)


# In[41]:


y_pred = classifier.predict(X_test)


# In[45]:


y_pred = (y_pred > 0.5)    # Setting a threshold i.e. if value of y_pred is >0.5 or 50% then set it as True else False


# In[46]:


y_pred


# In[47]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)


# In[48]:


cm


# In[49]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy*100)


# In[59]:


from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred)
print('Precision: %f' % precision)


# recall: tp / (tp + fn)
recall = recall_score(y_test, y_pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, y_pred)
print('F1 score: %f' % f1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




