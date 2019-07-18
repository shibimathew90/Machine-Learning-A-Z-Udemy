#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Part 1 - Building CNN model


# In[12]:


from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Conv2D

# From the new keras library, convolution2d has been updated to Conv2D, Conv2D(32, 3, 3) becomes Conv2D(32, (3, 3))


# In[4]:


# initializing the classifier


# In[5]:


classifier = Sequential()


# In[13]:


# step 1 - adding Convolution layer
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# 32 here is the number of feature detectors therefore 32 feature layers will be created 
# and 3,3 is the no. of rows and columns of each feature detector.
# Since we have colored images, hence in input shape we use 3 as R,G,B columns. If we had black n white images, then it will be 1
# 64, 64 are the dimension of the 2d array into which we want to convert all our images since our images are all of different sizes.


# In[14]:


# Step 2 - Pooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:


classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:





# In[ ]:





# In[15]:


# Step 3 - Flattenig
classifier.add(Flatten())


# In[ ]:





# In[17]:


# Full connection - building a classic ANN 
classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))


# In[18]:


# Compiling the CNN
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[19]:


# Part 2 - Fitting CNN to images

from keras.preprocessing.image import ImageDataGenerator


# In[20]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


# In[21]:


test_datagen = ImageDataGenerator(rescale=1./255)


# In[22]:


training_set = train_datagen.flow_from_directory(
                                                'C:/Udemy/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/training_set',
                                                target_size=(64, 64),   # 64,64 since out image size was set as 64,64 in step 1 of adding convolution layer
                                                batch_size=32,
                                                class_mode='binary')


# In[23]:


test_set = test_datagen.flow_from_directory(
                                            'C:/Udemy/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')


# In[ ]:


classifier.fit_generator(training_set,
                         steps_per_epoch=8000,    # since we have 8000 images in training set
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)


# In[ ]:




