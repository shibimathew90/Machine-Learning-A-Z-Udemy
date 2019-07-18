#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[78]:


dataset = pd.read_csv('C:/Udemy/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 9 - Dimensionality Reduction\Section 44 - Linear Discriminant Analysis (LDA)/Wine.csv')


# In[79]:


dataset.head()


# In[80]:


X = dataset.iloc[:,0:13].values
y = dataset.iloc[:,-1].values


# In[81]:


X.shape, y.shape


# In[82]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 0)


# In[83]:


X_train.shape, X_test.shape


# In[84]:


y_train.shape, y_test.shape


    


# In[85]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[86]:


X_train.shape, X_test.shape


# In[87]:


X_train, X_test


# In[88]:


# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[89]:


lda = LDA(n_components = 2)

X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


# In[90]:


X_train, X_test


# In[ ]:





# In[91]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[92]:


y_pred = classifier.predict(X_test)


y_pred


# In[93]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm


# In[94]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy*100)



# In[97]:


#Visualizing Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step=0.01),
                     np.arange(start=X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red','green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training Set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()


# In[98]:


# Visualizing test set results
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step=0.01),
                     np.arange(start=X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red','green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training Set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
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




