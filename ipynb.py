#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES AND DATASET

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


# read the csv file 
admission_df=pd.read_csv('Admission_Predict.csv')


# In[5]:


admission_df.head()


# In[6]:


# Let's drop the serial no.
admission_df.drop('Serial No.',axis=1,inplace=True)
admission_df


# # PERFORM EXPLORATORY DATA ANALYSIS

# In[7]:


# checking the null values
admission_df.isnull().sum()


# In[8]:


# Check the dataframe information
admission_df.info()


# In[9]:


# Statistical summary of the dataframe
admission_df.describe()


# In[11]:


# Grouping by University ranking 
df_university=admission_df.groupby(by='University Rating').mean()
df_university


# In[12]:


admission_df.hist(bins=30,figsize=(20,20),color='red')


# In[13]:


sns.pairplot(admission_df)


# In[14]:


corr_matrix=admission_df.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr_matrix,annot=True)
plt.show()
  


# # CREATE TRAINING AND TESTING DATASET

# In[15]:


admission_df.columns


# In[16]:


X=admission_df.drop(columns=['Chance of Admit'])


# In[17]:


y=admission_df['Chance of Admit']


# In[18]:


X.shape


# In[19]:


y.shape


# In[20]:


y


# In[21]:


X=np.array(X)
y=np.array(y)


# In[22]:


y=y.reshape(-1,1)
y.shape


# In[23]:


# scaling the data before training the model
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler_x=StandardScaler()
X=scaler_x.fit_transform(X)


# In[24]:


scaler_y=StandardScaler()
y=scaler_y.fit_transform(y)


# In[25]:


# spliting the data in to test and train sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15)


# In[26]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,accuracy_score


# In[27]:


LinearRegression_model=LinearRegression()
LinearRegression_model.fit(X_train,y_train)


# In[28]:


accuracy_LinearRegression=LinearRegression_model.score(X_test,y_test)
accuracy_LinearRegression

