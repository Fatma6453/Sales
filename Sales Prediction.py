#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=UserWarning)



df= pd.read_csv('Salary Data.csv')




df.head()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


# to know about how may data's are null in the dataframe
df.isnull().sum()


# In[10]:


df.describe()


# In[11]:


# to choose the columns which one will be our Feature and Target
df.columns


# In[12]:


# dropping all the null values from the dataset
df1 = df.dropna()
df1.isnull().sum()


# In[13]:


# Choosing only the features that is relevant for the prediction
# You can also keep 'Job Title' in the dataframe but i haven't kept it
df2= df1.drop( ['Age', 'Gender','Job Title'], axis=1)
df2.head(10)


# In[14]:


# Converting the Categorial data into numbers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df2['Education Level'] = label_encoder.fit_transform(df2['Education Level'])
df2.head()


# In[15]:


# Splitting the data into x and y for training and testing purpose
x=df2[['Education Level', 'Years of Experience']]
x.shape


# In[16]:


x.head(5)


# In[17]:


y=df2[['Salary']]
y.head(5)


# In[18]:


# Splitting the above data into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2,random_state = 9598)


# In[19]:


# to know about the shape of traing and testing data

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[21]:


# As it is the Regrssion Problem lets try all the Regression Algorith to get the best accuracy
from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[23]:


model.fit(x_train,y_train)


# In[24]:


# to predict the data
y_pred = model.predict(x_test)
y_pred


# In[25]:


# to know what the orignal prediction was for the testing data
y_test


# In[26]:


# to Calculate the Accuracy of the Model
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,mean_squared_error


# In[27]:


# to print the mean absolute error
mean_absolute_error(y_test,y_pred)


# In[28]:


# to print the mean absolute percentage
per_e = mean_absolute_percentage_error(y_test,y_pred)
per_e


# In[29]:


# to print the accuracy in percentage
acc1 = (1-per_e)*100
acc1


# In[31]:


#Trying other Algorhithms to check which gives the best accuracy

# KNeighbors Regressor

from sklearn.neighbors import KNeighborsRegressor
model2 = KNeighborsRegressor() 
model2.fit(x_train,y_train)
y_pred2 = model2.predict(x_test)
error2 = mean_absolute_percentage_error(y_test,y_pred2)
acc2 = (1-error2)*100
acc2


# In[32]:


# DecisionTree Regressor

from sklearn.tree import DecisionTreeRegressor
model3 = DecisionTreeRegressor()
model3.fit(x_train,y_train)
y_pred3 = model3.predict(x_test)
error3 = mean_absolute_percentage_error(y_test,y_pred3)
acc3 = (1-error3)*100
acc3


# In[36]:


# RandomForest Regressor

from sklearn.ensemble import RandomForestRegressor
model4 = RandomForestRegressor()
model4.fit(x_train,y_train)
y_pred4 = model4.predict(x_test)
error4 = mean_absolute_percentage_error(y_test,y_pred4)
acc4 = (1-error4)*100
acc4


# In[34]:


print("The Acuuracy of Linear Regressor is ",acc1)
print("The Acuuracy of KNeighbors Regressor is ",acc2)
print("The Acuuracy of DecisionTree Regressor is ",acc3)
print("The Acuuracy of RandomForest Regressor is ",acc4)


# In[ ]:




