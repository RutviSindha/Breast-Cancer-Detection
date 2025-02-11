#!/usr/bin/env python
# coding: utf-8

# In[1]:

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[2]:

file_path = 'breast-cancer.csv'
df = pd.read_csv(file_path)


# In[3]:

df.head()

# In[4]:

df.info()


# In[5]:

df.isnull().sum()


# In[6]:

df.drop('id',axis=1,inplace=True)


# In[7]:

df.describe().T


# In[8]:

px.histogram(data_frame=df, x='diagnosis', color='diagnosis', color_discrete_sequence=['#05445E','#75E6DA'])


# In[9]:

px.histogram(data_frame=df, x='area_mean', color='diagnosis', color_discrete_sequence=['#05445E','#75E6DA'])


# In[10]:

px.scatter(data_frame=df, x='symmetry_worst', color='diagnosis', color_discrete_sequence=['#05445E','#75E6DA'])


# In[12]:

px.scatter(data_frame=df, x='concavity_worst', color='diagnosis', color_discrete_sequence=['#05445E','#75E6DA'])


# In[13]:

df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)


# In[14]:

corr = df.corr()


# In[15]:

df['diagnosis'].value_counts()


# In[16]:

plt.figure(figsize=(20,20))
sns.heatmap(corr, cmap='mako_r', annot=True)
plt.show()


# In[17]:

cor_target = abs(corr["diagnosis"])
relevant_features = cor_target[cor_target>0.2]
names = [index for index, value in relevant_features.items()]
names.remove('diagnosis')
print(names)


# In[18]:

X=df[names]
y=df['diagnosis']


# In[19]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[20]:

scaler = StandardScaler()
scaler.fit(X_train)


# In[21]:

x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)


# In[22]:

model = LogisticRegression()


# In[23]:

model.fit(X_train,  y_train)


# In[24]:

predictions = model.predict(X_test)


# In[25]:

accuracy = accuracy_score(y_test, predictions)


# In[26]:

print(f'the model accuracy: {accuracy}')


# In[27]:

total_samples = len(df)
print(f"Total number of samples: {total_samples}")


# In[28]:

model.predict(X_test.iloc[[25]])
