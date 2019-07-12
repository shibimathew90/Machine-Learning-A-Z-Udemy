#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[43]:


dataset = pd.read_csv('C:/Udemy/Machine Learning A-Z Template Folder/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# quoting = 3 ignores the " (double quotes) from our data. Suppose there is any " in the review column, it will just ignore any action on that double quote and move ahead


# In[44]:


dataset.head()


# In[45]:


dataset.shape


# In[46]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[47]:


from nltk.stem.porter import PorterStemmer


# In[48]:


corpus = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    
    corpus.append(review)


# In[49]:


corpus


# In[51]:


from sklearn.feature_extraction.text import CountVectorizer


# In[55]:


cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()


# In[56]:


X


# In[57]:


X.shape


# In[62]:


y = dataset.iloc[:,1].values


# In[64]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[65]:


cm


# In[ ]:




