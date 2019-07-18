#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Grid search provides an answer to what should be the optimal values of hyper-parameters we need for a model. 

# Which model would be the best one.

# first of course what you need to do is know if your problem is a regression problem or a classification problem or a clustering problem.
# So that's easy you just need to look at your dependent variable.If you don't have a dependent variable then it's a clustering problem.
# And if you have a dependent variable you see if it's a continuous outcome or a categorical outcome if, it's a continuous 
# outcome then your problem is a regression problem & if it's a categorical outcome then your problem is a classification problem.
# And then the second step is to ask yourself is my problem a linear problem or nonlinear problem. 
# And that is not an obvious question especially when you have a large data set you cannot figure out if your data is linearly 
# separable or if you would rather choose a linear model like SVM. If you're doing classification or a nonlinear model 
# like kernel SVM and this question can be answered by Grid search
# Grid search will tell us if we should rather choose a linear model like as SVM or a non-linear model like kernelSVM.


# In[24]:


dataset = pd.read_csv('C:/Udemy/Machine Learning A-Z Template Folder/Part 10 - Model Selection & Boosting/Section 48 - Model Selection/Social_Network_Ads.csv')


# In[25]:


dataset.head()


# In[26]:


X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values


# In[27]:


X, y


# In[28]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state= 0)


# In[29]:


X_train.shape, X_test.shape


# In[30]:


y_train.shape, y_test.shape


# In[31]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[32]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


# In[33]:


y_pred = classifier.predict(X_test)


# In[34]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[35]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy*100)


# In[36]:


# Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)


# In[37]:


accuracies


# In[38]:


accuracies.mean()     # mean of the 10 accuracies


# In[39]:


accuracies.std()          # calcualte the sd of 10 accuracies


# In[53]:


# Applying Grid Search to find the best model and best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C' : [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma' : [0.5, 0.1, 0.6, 0.4, 0.3, 0.2, 0.7, 0.8]}
             ]

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
gird_search = grid_search.fit(X_train, y_train)


# In[54]:


best_accuracy = grid_search.best_score_
best_accuracy


# In[50]:


best_parameters = grid_search.best_params_
best_parameters


# In[ ]:





# In[ ]:





# In[ ]:


#Visualizing Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step=0.01),
                     np.arange(start=X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red','green'))(i), label = j)
plt.title('Kernel SVM (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[ ]:


# Visualizing test set results
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step=0.01),
                     np.arange(start=X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red','green'))(i), label = j)
plt.title('Kernel SVM (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




