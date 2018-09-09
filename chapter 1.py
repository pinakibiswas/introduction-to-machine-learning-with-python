#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import sklearn
import sys
import pandas as pd
import matplotlib.pyplot as plt
import mglearn


# In[20]:


#import sklearn as skl
#print (skl.__version__)
#print (skl.__file__)


# In[22]:


from sklearn.datasets import load_iris
iris_dataset = load_iris()


# In[23]:


print(iris_dataset.keys())


# In[42]:


#print(iris_dataset['DESCR'])
print("type of dataset: {}".format(type(iris_dataset['data'])))
print("data shape: \n {}".format(iris_dataset['data'].shape))
print("1st two rows of data: \n{}".format(iris_dataset['data'][:2]))


# In[43]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)


# In[46]:


print("x_train_shape: {}".format(x_train.shape))
print("y_train_shape: {}".format(y_train.shape))
print("x_test_shape: {}".format(x_test.shape))
print("y_test_shape: {}".format(y_test.shape))


# In[64]:


#print(x_train.keys())
print("first two rows of x_train: \n{}".format(x_train[:2]))
x_train_dataframe = pd.DataFrame(x_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(x_train_dataframe, 
                           c=y_train, 
                           figsize=(15,15), 
                           marker='o', 
                           hist_kwds={'bins': 20},
                           s=60, 
                           #alpha=.8, 
                           cmap=mglearn.cm3)


# In[65]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)


# In[71]:


x_new = np.array([[5, 2.9, 1, .2]])
prediction = knn.predict(x_new)
print("predicted class for x_new is {}".format(iris_dataset['target_names'][prediction]))


# In[73]:


y_pred = knn.predict(x_test)
print(np.mean(y_pred == y_test))


# In[ ]:




