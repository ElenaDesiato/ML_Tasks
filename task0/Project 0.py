#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error

training_filename= 'task0_sl19d1/train.csv'
test_filename = 'task0_sl19d1/test.csv'

df=pd.read_csv(training_filename, dtype=str)
df=df.map(np.float64)

#just testing
print(df.info())   #gives information
print(df.head(10)) #gives 12 data points


# In[3]:


#verifying y is actually the mean of vector x as claimed
y = df['y']
id = df['Id']
x = df.drop(labels=['Id','y'],axis=1)
y_pred = x.mean(axis=1)
print(np.column_stack((id,y,y_pred)))
error = mean_squared_error(y, y_pred)**0.5
print(error) #small error but acceptable


# In[4]:


#Making predictions on the test set (computing the mean)
df=pd.read_csv(test_filename, dtype=str)
df=df.map(np.float64)
# print(df)
id = df['Id']
x = df.drop(labels=['Id'],axis=1)
y_pred = x.mean(axis=1)

res = pd.DataFrame()
res['Id'] = id
res['y'] = y_pred
print(res)
res.to_csv('task0_sl19d1/submission.csv', sep=',',index=False)


# In[ ]:




