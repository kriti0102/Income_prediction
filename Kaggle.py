
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import sklearn.metrics as metrics
import warnings
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
warnings.filterwarnings('ignore')


# In[3]:


df1 = pd.read_csv('../input/tcd ml 2019-20 income prediction training (with labels).csv')
df2 = pd.read_csv('../input/tcd ml 2019-20 income prediction test (without labels).csv')
df1.rename(columns={'Income':'Income in EUR'},inplace=True)
data = pd.concat([df1, df2])


# In[4]:


data = data.fillna(method='ffill')
data = data.set_index('Instance')


# In[5]:


data[['Age','Year of Record']] = data[['Age','Year of Record']].fillna(0).astype(np.int64)


# In[6]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical = ['Age', 'Year of Record', 'Size of City', 'Body Height [cm]']

features = pd.DataFrame(data = data)
features[numerical] = scaler.fit_transform(data[numerical])

display(features.head(n = 5))


# In[7]:


features = features.drop(['Hair Color','Wears Glasses'], axis=1)
finaldf = features


# In[8]:


categorical_feature_mask = finaldf.dtypes==object
categorical_cols = finaldf.columns[categorical_feature_mask].tolist()


# In[9]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[10]:



finaldf[categorical_cols] = finaldf[categorical_cols].apply(lambda col: le.fit_transform(col))
finaldf[categorical_cols].head(10)


# In[11]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features = categorical_feature_mask, sparse=False ) 


# In[15]:


train_df = finaldf[:df1.shape[0]]
test_df = finaldf[df1.shape[0]:]


# In[16]:


import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
train_df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in train_df.columns.values]
test_df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in test_df.columns.values]
x_test = test_df.drop(['Income in EUR'],axis=1)


# In[17]:


X = train_df.drop('Income in EUR',axis=1)
y = train_df['Income in EUR']


# In[18]:


x_train, x_val, y_train, y_val = ms.train_test_split(X,y, test_size=0.3, random_state=777)


# In[19]:


from xgboost import XGBRegressor
model = XGBRegressor(objective ='reg:squarederror',n_estimators=3500,
                     learning_rate=0.05,
                     nthread= 1,
                     reg_alpha=7,
                     max_depth=6,
                     colsample_bytree =1,
                     subsample = 0.5,
                     num_round = 10)
model.fit(x_train, y_train, early_stopping_rounds=20, 
             eval_set=[(x_val, y_val)], verbose=True)


# In[20]:


pred = model.predict(x_val)


# In[21]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_val, pred))
print(rms)


# In[23]:


pred_sub = model.predict(x_test)


# In[24]:


test_df['Income'] = pred_sub
test_df.to_csv('KaggleSub.csv', columns = ['Income'])
test_df['Income']

