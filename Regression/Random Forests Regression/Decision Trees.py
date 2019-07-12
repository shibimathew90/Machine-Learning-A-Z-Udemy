#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv('C:/Udemy/Projects/Decision Tree Regression/Position Salaries.csv')


# In[5]:


df.head()


# In[8]:


X = df.iloc[:, 1:2].values
Y = df.iloc[:,-1].values


# In[9]:


X, Y


# In[10]:


from sklearn.tree import DecisionTreeRegressor


# In[11]:


regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, Y)


# In[12]:


y_pred = regressor.predict(6.5)


# In[13]:


y_pred


# In[16]:


plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X))
plt.title('Experience and Salary Training set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:


# This doesnt look like the real shape of DT regression. Considering the entropy in the informaion gain, its splitting the independent
# variables into several intervals. According to the DT its calculating the average of dependent variables salaries and therefore
# for all the levels contained in an interval the value of the prediction should be constant equal to the average of the dependent
# variable in the interval. But as per our graph, the value of prediction is not constant for any 2 intervals. Currently we are
# making predictions for each of the 10 levels incremented by 1 and joining the predictions by a straight line as it had no 
# predictions to plot in this interval.
# The DT regression model is not continuous model, and the best way to visualize in higher resolution as below.


# In[21]:


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[ ]:




