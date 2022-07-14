#!/usr/bin/env python
# coding: utf-8

# ## Importing Library & Data

# In[4]:


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import math
import os
import seaborn
import matplotlib.pyplot as plt
from keras.datasets import mnist


# In[5]:


(X_train, Y_train) , (X_test, Y_test) = mnist.load_data()
X_train = X_train / 255
X_test = X_test / 255
X_train_1d = np.array([a.reshape(1,784) for a in X_train]).reshape(60000,-1)
X_test_1d = np.array([a.reshape(1,784) for a in X_test]).reshape(10000,-1)


# ## Display of Handwritten Digits

# In[6]:


def display(data_1d,idx):
    image = data_1d[idx].reshape(28,28)
    plt.figure
    plt.imshow(image, cmap='gray_r')


# In[7]:


test_data_index = 0
plt.figure(figsize=(14,14))

for i in range(10):
    plt.subplot(4,5,i+1)
    display(X_train_1d,i)
   
plt.show()


# ## ANN Model Implementation (Without Inner Layers)

# In[8]:


model = keras.Sequential([keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_1d, Y_train, epochs=6)


# ## Model Evaluation (without hidden layers)

# In[9]:


result = model.evaluate(X_test_1d, Y_test) 


# In[10]:


Y_predict = [np.argmax(i) for i in model.predict(X_test_1d)]


# ## Confusion Matrix (Without Hidden Layers)

# In[11]:


conf_mat = tf.math.confusion_matrix(labels=Y_test,predictions=Y_predict)
ax = seaborn.heatmap(conf_mat, annot=True, fmt='d', cmap="YlGnBu")


# ## ANN Model Implementation (With Inner Layers)

# In[12]:


model = keras.Sequential([keras.layers.Dense(100, input_shape=(784,), activation='relu'),
                          keras.layers.Dense(100, input_shape=(784,), activation='relu'),
                         keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_1d, Y_train, epochs=6)


# ## Model Evaluation (with 2 hidden layers)

# In[13]:


result = model.evaluate(X_test_1d, Y_test) 


# In[14]:


Y_predict = [np.argmax(i) for i in model.predict(X_test_1d)]


# ## Checking the output

# In[15]:


display(X_test_1d,2)


# In[16]:


Y_predict[2]


# In[17]:


Y_test[2]


# ## Confusion Matrix (With Hidden Layers) 

# In[18]:


conf_mat = tf.math.confusion_matrix(labels=Y_test,predictions=Y_predict)
ax = seaborn.heatmap(conf_mat, annot=True, fmt='d', cmap="YlGnBu")


# In[ ]:




