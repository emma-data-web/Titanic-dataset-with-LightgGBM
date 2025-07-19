#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# In[7]:


df = pd.read_csv('titanic_train.csv')
pd.DataFrame(df)


# In[8]:


df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)


# In[12]:


df.head(3)


# In[13]:


sns.heatmap(df.isnull())


# In[14]:


cat_col = ['Sex','Embarked']


# In[15]:


for col in cat_col:
    df[col] = df[col].astype('category')


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(df.drop('Survived',axis=1), df['Survived'], test_size=0.3,
                                                    random_state=101)


# In[17]:


x_train


# In[18]:


model = lgb.LGBMClassifier()


# In[19]:


model


# In[20]:


model.fit(x_train,y_train, categorical_feature=cat_col)


# In[29]:


y_pred = model.predict(x_test)


# In[22]:


importances = model.feature_importances_


# In[24]:


feature_columns = x_train.columns


# In[23]:


importances


# In[27]:


pd.DataFrame({
    'features': feature_columns,
    'importances': importances
}).sort_values(by='importances', ascending=False)


# In[30]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))


# In[ ]:




