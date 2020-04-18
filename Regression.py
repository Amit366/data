#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as ply
from sklearn.datasets import load_boston


# In[54]:


#understanding dataset
boston=load_boston()
print(boston.DESCR)


# In[10]:


#access data attributes
dataset = boston.data
for name,index in enumerate(boston.feature_names):
    print(index,name)


# In[18]:


#reshaping data
data =dataset[:,12].reshape(-1,1)


# In[19]:


#shape of data
np.shape(data)


# In[16]:


#target values
target=boston.target.reshape(-1,1)


# In[17]:


#shape of target
np.shape(target)


# In[26]:


#ensuring that matplotlib is working inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
ply.scatter(data,target,color='green')
ply.xlabel('Lower income population')
ply.ylabel('Cost of House')
ply.show()


# In[44]:


#regression
#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

#creating a regression model
reg=Ridge()

#fit model
reg.fit(data,target)


# In[45]:


#prediction
pred=reg.predict(data)


# In[46]:


#ensuring that matplotlib is working inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
ply.scatter(data,target,color='red')
ply.plot(data,pred,color='green')
ply.xlabel('Lower income population')
ply.ylabel('Cost of House')
ply.show()


# In[47]:


#circumventing curve using polynomial model
from sklearn.preprocessing import PolynomialFeatures

#to allow merging of models
from sklearn.pipeline import make_pipeline


# In[48]:


model=make_pipeline(PolynomialFeatures(7),reg)


# In[49]:


model.fit(data,target)


# In[50]:


pred=model.predict(data)


# In[51]:


#ensuring that matplotlib is working inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
ply.scatter(data,target,color='red')
ply.plot(data,pred,color='green')
ply.xlabel('Lower income population')
ply.ylabel('Cost of House')
ply.show()


# In[52]:


# r_2 metric
from sklearn.metrics import r2_score


# In[53]:


#prediction
r2_score(pred,target)

