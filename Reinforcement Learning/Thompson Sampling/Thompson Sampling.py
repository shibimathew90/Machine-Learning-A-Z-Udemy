#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# In[12]:


dataset.head()


# In[13]:


dataset.shape


# In[6]:


#Implementing UCB


# In[16]:


import random
d = dataset.shape[1]
ads_selected = []
numbers_of_rewards_1 = [0]  * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, dataset.shape[0]):     # dataset.shape[0] = 10000
    ad = 0
    max_random = 0
    for i in range(0, d):   # We need to compute for each version of ad the average and the confidence interval. So creating another loop
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)  
        if(random_beta > max_random):
            max_random = random_beta
            ad = i       # storing the index of the ad which is having the max upper bound.
    ads_selected.append(ad)
    rewards = dataset.values[n,ad]
    if rewards == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
    total_reward += rewards
                                        


# In[17]:


print(total_reward)


# In[18]:


plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('number of times each ad was selected')
plt.show()


# In[ ]:




