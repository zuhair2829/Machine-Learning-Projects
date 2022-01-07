#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://www.kaggle.com/hesh97/titanicdataset-traincsv
# get dataset


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[5]:


df = pd.read_csv("C:\\Users\\ROG STRIX\\Desktop\\DataSets\\titanic_data.csv")


# In[6]:


df.shape


# In[28]:


data = pd.read_csv("C:\\Users\\ROG STRIX\\Desktop\\DataSets\\titanic_data.csv")


# In[29]:


df['SibSp'].unique()


# In[30]:


df.head()


# In[31]:


df.describe(include = "all")


# In[32]:


#check for any other unusable values
print(pd.isnull(df).sum())


# In[33]:


#draw a bar plot of survival by sex
sns.barplot(x="Sex", y="Survived", data=df)

#print percentages of females vs. males that survive
print("Percentage of females who survived:", df["Survived"][df["Sex"] == 'female'].value_counts())

print("Percentage of males who survived:", df["Survived"][df["Sex"] == 'male'].value_counts())


# In[34]:


#draw a bar plot of survival by sex
sns.barplot(x="Sex", y="Survived", data=df)

#print percentages of females vs. males that survive
print("Percentage of females who survived:", df["Survived"][df["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", df["Survived"][df["Sex"] == 'male'].value_counts(normalize = True)[1]*100)


# In[35]:


#draw a bar plot of survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=df)

#print percentage of people by Pclass that survived
print("Percentage of Pclass = 1 who survived:", df["Survived"][df["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", df["Survived"][df["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", df["Survived"][df["Pclass"] == 3].value_counts(normalize = True)[1]*100)


# In[36]:


#draw a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=df)

#I won't be printing individual percent values for all of these.
print("Percentage of SibSp = 0 who survived:", df["Survived"][df["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:", df["Survived"][df["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", df["Survived"][df["SibSp"] == 2].value_counts(normalize = True)[1]*100)


# In[37]:


#draw a bar plot for Parch vs. survival
sns.barplot(x="Parch", y="Survived", data=df)
plt.show()


# In[38]:


plt.figure(figsize=(5,5))
df.AgeGroup.value_counts().plot(kind = 'pie')


# In[39]:


df["CabinBool"] = (df["Cabin"].notnull().astype('int'))

#calculate percentages of CabinBool vs. survived
print("Percentage of CabinBool = 1 who survived:", df["Survived"][df["CabinBool"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of CabinBool = 0 who survived:", df["Survived"][df["CabinBool"] == 0].value_counts(normalize = True)[1]*100)
#draw a bar plot of CabinBool vs. survival
sns.barplot(x="CabinBool", y="Survived", data=df)
plt.show()


# In[40]:


data.shape


# In[41]:


data = data.dropna()


# In[42]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# In[43]:


data = data.drop(['Embarked', 'Name', 'Cabin', 'Ticket'], axis = 1)


# In[44]:


data['Sex'] = data['Sex'].map({'male': 0,'female': 1})


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(data.drop(['Survived'],axis=1), 
                                                    data['Survived'], test_size=0.20, 
                                                    random_state=8)


# In[46]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[47]:


predictions = logmodel.predict(X_test)
X_test.head()


# In[48]:


accuracy = logmodel.score(X_test,y_test)
print(accuracy*100,'%')


# In[49]:


predictions


# In[50]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[51]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)


# In[ ]:





# In[ ]:





# In[ ]:




