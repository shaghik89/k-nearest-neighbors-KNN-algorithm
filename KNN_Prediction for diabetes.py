#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[5]:


dataset = pd.read_csv('desktop/KNN_Dataset.csv')


# In[7]:


dataset.head()


# In[8]:


#Replace Zeroes since the person with blooad pressure of 0 will die. 
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']


# In[9]:


for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)


# In[10]:


print(dataset['Glucose'])


# In[11]:


#Split dataset to train and test set
X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)


# In[12]:


#Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[16]:


#select K which should be an odd number here is 12 - 1=11.
import math
math.sqrt(len(y_test))


# In[17]:


# Define the model: Init K-NN
classifier = KNeighborsClassifier(n_neighbors=11, p=2,metric='euclidean')


# In[18]:


# Fit Model
classifier.fit(X_train, y_train)


# In[19]:


# Predict the test set results
y_pred = classifier.predict(X_test)
y_pred


# In[20]:


# Evaluate Model
cm = confusion_matrix(y_test, y_pred)
print (cm)
print(f1_score(y_test, y_pred))


# In[21]:


print(accuracy_score(y_test, y_pred))


# In[ ]:




