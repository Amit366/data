#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as ply
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


data=pd.read_csv('emnist.csv')
data.head()


# In[11]:


data.loc[3]


# In[24]:


#extracting the data
d=data.iloc[2,1:].values


# In[32]:


#reshaping the extracted data
d=d.reshape(28,28).astype('uint8')
ply.imshow(d)


# In[34]:


#separating label and pixels
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]


# In[36]:


#train and test the datas
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)


# In[38]:


#check data
x_train.head()


# In[40]:


y_train.head()


# In[42]:


#calling rf classifier
rf=RandomForestClassifier(n_estimators=100)


# In[45]:


#fit the model
rf.fit(x_train,y_train)


# In[47]:


#prediction test
pred=rf.predict(x_test)


# In[49]:


pred


# In[51]:


#check prediction accuracy
s=y_test.values

#calculate no of correct predictions
count=0
for i in range(len(pred)):
    if pred[i]==s[i]:
        count=count+1


# In[53]:


count


# In[55]:


#total values which was predicted
len(pred)


# In[59]:


len(pred)


# In[60]:


#accuracy
2785/3760

