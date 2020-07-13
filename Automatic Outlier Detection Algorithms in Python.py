#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 6 Dimensionality Reduction Algorithms With Python
https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python/


# In[ ]:


# 4 Automatic Outlier Detection Algorithms in Python
# https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/
# how to use automatic outlier detection and removal to improve machine learning predictive modeling performance.


# In[ ]:


# Automatic outlier detection models provide an alternative to statistical techniques with a larger number of input variables with complex and unknown inter-relationships.
# How to correctly apply automatic outlier detection and removal to the training dataset only to avoid data leakage.
# How to evaluate and compare predictive modeling pipelines with outliers removed from the training dataset.


# In[ ]:


Data Source Link : https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv
Data Source Link : https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.names


# In[1]:


# load and summarize the dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
# load the dataset
url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df=read_csv(url,header=None)
# retrieve the array
data=df.values
# split into input and output elements
X,y= data[:,:-1],data[:,-1]
# summarize the shape of the dataset
print(X.shape,y.shape)
# split into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=1)
# summarize the shape of the train and test sets
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[2]:


# Baseline Model Performance
# evaluate model on the raw dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# load the dataset
url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(url, header=None)
# retrieve the array
data=df.values
# split into input and output elements
X,y=data[:,:-1],data[:,-1]
# split into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=1)
# fit the model
model=LinearRegression()
model.fit(X_train,y_train)
# evaluate the model
yhat=model.predict(X_test)
# evaluate predictions
mae=mean_absolute_error(y_test,yhat)
print('MAE: %.3f' % mae)


# In[ ]:


# Automatic Outlier Detection


# # identify outliers in the training dataset
# iso=IsolationForest(contamination=0.1)
# yhat=iso.fit_predict(X_train)
# # select all rows that are not outliers
# mask=yhat != -1
# X_train,y_train=X_train[mask,:],y_train[mask]

# In[3]:


# evaluate model performance with outliers removed using isolation forest
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
# load the dataset
url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df=read_csv(url,header=None)
# retrieve the array
data=df.values
# split into input and output elements
X,y=data[:,:-1],data[:,-1]
# split into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=1)
# summarize the shape of the training dataset
print(X_train.shape,y_train.shape)
# identify outliers in the training dataset
iso=IsolationForest(contamination=0.1)
yhat=iso.fit_predict(X_train)
# select all rows that are not outliers
mask=yhat != -1
X_train,y_train=X_train[mask,:],y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape,y_train.shape)
# fit the model
model=LinearRegression()
model.fit(X_train,y_train)
# evaluate the model
yhat=model.predict(X_test)
# evaluate predictions
mae=mean_absolute_error(y_test,yhat)
print('MAE: %.3f' % mae)


# Minimum Covariance Determinant and 
# identify outliers in the training dataset

# ee=EllipticEnvelope(contamination=0.01)
# yhat=ee.fit_predict(X_train)

#  Evaluate model performance with outliers removed using elliptical envelope

# In[6]:


from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import mean_absolute_error
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(url, header=None)
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
ee = EllipticEnvelope(contamination=0.01)
yhat = ee.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)


# Local Outlier Factor

# In[7]:


# identify outliers in the training dataset
lof=LocalOutlierFactor()
yhat=lof.fit_predict(X_train)


# In[8]:


# evaluate model performance with outliers removed using local outlier factor
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error
# load the dataset
url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df=read_csv(url,header=None)
# retrieve the array
data=df.values
# split into input and output elements
X,y=data[:,:-1],data[:,-1]
# split into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=1)
# summarize the shape of the training dataset
print(X_train.shape,y_train.shape)
# identify outliers in the training dataset
lof=LocalOutlierFactor()
yhat=lof.fit_predict(X_train)
# select all rows that are not outliers
mask=yhat != -1
X_train,y_train=X_train[mask,:],y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape,y_train.shape)
# fit the model
model=LinearRegression()
model.fit(X_train,y_train)
# evaluate the model
yhat=model.predict(X_test)
# evaluate predictions
mae=mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)


# One-Class SVM
# # identify outliers in the training dataset

# In[9]:


ee=OneClassSVM(nu=0.01)
yhat=ee.fit_predict(X_train)


# In[10]:


# evaluate model performance with outliers removed using one class SVM
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_absolute_error
# load the dataset
url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df=read_csv(url,header=None)
# retrieve the array
data=df.values
# split into input and output elements
X,y=data[:,:-1],data[:,-1]
# split into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=1)
# summarize the shape of the training dataset
print(X_train.shape,y_train.shape)
# identify outliers in the training dataset
ee=OneClassSVM(nu=0.01)
yhat=ee.fit_predict(X_train)
# select all rows that are not outliers
mask= yhat != -1
X_train,y_train=X_train[mask,:],y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape,y_train.shape)
# fit the model
model=LinearRegression()
model.fit(X_train,y_train)
# evaluate the model
yhat=model.predict(X_test)
# evaluate predictions
mae=mean_absolute_error(y_test,yhat)
print('MAE: %.3f' % mae)


# In[ ]:




