#!/usr/bin/env python
# coding: utf-8

# In[26]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings 
#We do not want to see warnings
warnings.filterwarnings("ignore") 


# In[2]:


#import data
data = pd.read_csv("uber.csv")


# In[3]:


#Create a data copy
df = data.copy()


# In[4]:


#Print data
df.head


# In[8]:


#Get Info
df.info()


# In[5]:


#pickup_datetime is not in required data format
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])


# In[6]:


df.info()


# In[11]:


#Statistics of data
df.describe()


# In[7]:


#Number of missing values
df.isnull().sum()


# In[22]:


#Correlation
df.corr()


# In[8]:


#Drop the rows with missing values
df.dropna(inplace=True)


# In[27]:


plt.boxplot(df['fare_amount'])


# In[10]:


#Remove Outliers
q_low = df["fare_amount"].quantile(0.01)
q_hi  = df["fare_amount"].quantile(0.99)

df = df[(df["fare_amount"] < q_hi) & (df["fare_amount"] > q_low)]


# In[11]:


#Check the missing values now
df.isnull().sum()


# In[12]:


#Time to apply learning models
from sklearn.model_selection import train_test_split


# In[13]:


#Take x as predictor variable
x = df.drop("fare_amount", axis = 1)
#And y as target variable
y = df['fare_amount']


# In[14]:


#Necessary to apply model
x['pickup_datetime'] = pd.to_numeric(pd.to_datetime(x['pickup_datetime']))
x = x.loc[:, x.columns.str.contains('^Unnamed')]


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


lrmodel = LinearRegression()
lrmodel.fit(x_train, y_train)


# In[18]:


#Prediction
predict = lrmodel.predict(x_test)


# In[19]:


#Check Error
from sklearn.metrics import mean_squared_error
lrmodelrmse = np.sqrt(mean_squared_error(predict, y_test))
print("RMSE error for the model is ", lrmodelrmse)


# In[20]:


#Let's Apply Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rfrmodel = RandomForestRegressor(n_estimators = 100, random_state = 101)


# In[21]:


#Fit the Forest
rfrmodel.fit(x_train, y_train)
rfrmodel_pred = rfrmodel.predict(x_test)


# In[23]:


#Errors for the forest
rfrmodel_rmse = np.sqrt(mean_squared_error(rfrmodel_pred, y_test))
print("RMSE value for Random Forest is:",rfrmodel_rmse)


# In[ ]:




