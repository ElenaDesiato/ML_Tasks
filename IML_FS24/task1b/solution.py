#!/usr/bin/env python
# coding: utf-8

# #### General guidance
# 
# This serves as a template which will guide you through the implementation of this task. It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# This is the jupyter notebook version of the template. For the python file version, please refer to the file `template_solution.py`.
# 
# First, we import necessary libraries:

# In[10]:


import numpy as np
import pandas as pd

# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge


#  #### Loading data

# In[11]:


data = pd.read_csv("train.csv")
y = data["y"].to_numpy()
data = data.drop(columns=["Id", "y"])
# print a few data samples
print(data.head())
X = data.to_numpy()


# #### Transform data

# In[12]:


"""
Transform the 5 input features of matrix X (x_i denoting the i-th component of X) 
into 21 new features phi(X) in the following manner:
5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
1 constant feature: phi_21(X)=1

Parameters
----------
X: matrix of floats, dim = (700,5), inputs with 5 features

Compute
----------
X_transformed: array of floats: dim = (700,21), transformed input with 21 features
"""
X_transformed = np.zeros((700, 21))
# TODO: Enter your code here

X_transformed[:,0:5] = X[:,0:5]
X_transformed[:,5:10] = np.power(X[:,0:5],2)
X_transformed[:,10:15] = np.exp(X[:,0:5])
X_transformed[:,15:20] = np.cos(X[:,0:5])
X_transformed[:,20] = np.ones((700,))

assert X_transformed.shape == (700, 21)


# #### Fit data

# In[29]:


"""
Use the transformed data points X_transformed and fit the linear regression on this 
transformed data. Finally, compute the weights of the fitted linear regression. 

Parameters
----------
X_transformed: array of floats: dim = (700,21), transformed input with 21 features
y: array of floats, dim = (700,), input labels)

Compute
----------
w: array of floats: dim = (21,), optimal parameters of linear regression
"""
w = np.zeros((21,))
# TODO: Enter your code here

#X_train = X_transformed[:600,:]
#y_train = y[:600]
#X_test = X_transformed[600:,:]
#y_test= y[600:]
#reg1 = LinearRegression(fit_intercept=False).fit(X_train,y_train) #because we only pass weights we need it to be centered?
#w_test1 = reg1.coef_
#y_pred1 = np.dot(X_test,w_test1)
#RMSE_1 = mean_squared_error(y_test,y_pred1, squared=False) 
#print("w/o intercept: Weights from test set:", w_test1)
#print("RMSE 1:", RMSE_1)
#reg = LinearRegression(fit_intercept = False).fit(X_transformed,y)
#w = reg.coef_

lambdas = [45,46,47,48,49,50]
n_folds = 10
RMSE_mat = np.zeros((n_folds, len(lambdas)))

kf  = KFold(n_splits = n_folds,random_state = None)
for i,lam in enumerate(lambdas): 
    for j, (train_index, test_index) in enumerate (kf.split(X_transformed)): 
        X_train, X_test = X_transformed[train_index], X_transformed[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        ridge_reg = Ridge(alpha=lam,fit_intercept = False) #as we only pass the coeffs/weights, we need the bias b to be 0 
        w = ridge_reg.fit(X_train,y_train).coef_

        y_pred = np.dot(X_test,w)
        RMSE_mat[j][i] = mean_squared_error(y_test,y_pred, squared=False)

avg_RMSE = np.mean(RMSE_mat, axis=0)
#print(avg_RMSE)

#Use entire test set to compute weights
reg = Ridge(alpha=45,fit_intercept = False).fit(X_transformed,y)
w = reg.coef_
#print("Weights from entire test set:",w)

assert w.shape == (21,)


# In[31]:


# Save results in the required format
np.savetxt("./results.csv", w, fmt="%.12f")


# In[ ]:




