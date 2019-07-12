#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# In[4]:


dataset.head()


# In[5]:


dataset.shape


# In[6]:


#Implementing UCB


# In[7]:


import math
d = dataset.shape[1]
ads_selected = []
numbers_of_selections = [0] * d    # creating a vector of size d containing 0. This is done because none of the ads are selected and hence no reward given at any round
sum_of_rewards = [0] * d    # same as above, sum of rewards for each round at starting is 0.
total_reward = 0
for n in range(0, dataset.shape[0]):     # dataset.shape[0] = 10000
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):   # We need to compute for each version of ad the average and the confidence interval. So creating another loop
        if(numbers_of_selections[i] > 0):
            average_reward = sum_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1)/numbers_of_selections[i])    # Since n starts at 0, so need to add + 1 to the log value so that it doest take log(0)
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400   
        if(upper_bound > max_upper_bound):
            max_upper_bound = upper_bound    # here we are taking the max upper bound from our ads.
            ad = i       # storing the index of the ad which is having the max upper bound.
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    rewards = dataset.values[n,ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + rewards
    total_reward += rewards
                                        


# In[8]:


print(total_reward)


# In[9]:


plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('number of times each ad was selected')
plt.show()


# In[ ]:




