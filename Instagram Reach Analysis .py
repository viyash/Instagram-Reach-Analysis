#!/usr/bin/env python
# coding: utf-8

# # INSTAGRAM REACH ANALYSIS - ML

# The Instagram Reach Analysis project aims to utilize machine learning techniques to analyze and understand the reach and engagement of Instagram posts. Instagram is a popular social media platform with millions of users, and understanding the factors that impact the reach of posts can be valuable for individuals, influencers, and businesses alike.

# 1.Import necessary modules

# In[38]:


import pandas as pd
import statsmodels
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import LinearRegression


# # Data Pre-processing

# 2.Read the Dataset

# In[4]:


data = pd.read_csv("Instagram data.csv", encoding='latin1')
print(data.head())


# 3.Check for null values

# In[5]:


data.isnull().sum()


# 4.Drop the null values

# In[6]:


data = data.dropna()


# In[7]:


data.info()


# # Analysis

# In[31]:


plt.figure(figsize=(10, 6))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.histplot(data['From Home'])
plt.show()


# In[33]:


plt.figure(figsize=(10, 6))
plt.title("Distribution of Impressions From Hashtags")
sns.histplot(data['From Hashtags'])
plt.show()


# In[34]:


home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home', 'From Hashtags', 'From Explore', 'Other']
values = [home, hashtags, explore, other]

plt.figure(figsize=(10, 6))
plt.style.use('fivethirtyeight')
plt.bar(labels, values)
plt.title('Impressions on Instagram Posts From Various Sources')
plt.xlabel('Sources')
plt.ylabel('Impressions')
plt.show()


# In[35]:


text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Instagram Captions")
plt.show()


# In[39]:


sns.set_style("whitegrid")
plt.figure(figsize=(10, 8))
plt.scatter(data["Impressions"], data["Likes"], s=data["Likes"], alpha=0.7)
plt.title("Relationship Between Likes and Impressions")
plt.xlabel("Impressions")
plt.ylabel("Likes")

# Fit and plot the trendline
regression = LinearRegression()
regression.fit(data[["Impressions"]], data["Likes"])
plt.plot(data["Impressions"], regression.predict(data[["Impressions"]]), color='red')

plt.show()


# In[40]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Assuming 'data' is the DataFrame containing the relevant data

sns.set_style("whitegrid")
plt.figure(figsize=(10, 8))
plt.scatter(data["Impressions"], data["Saves"], s=data["Saves"], alpha=0.7)
plt.title("Relationship Between Post Saves and Total Impressions")
plt.xlabel("Impressions")
plt.ylabel("Saves")

# Fit and plot the trendline
regression = LinearRegression()
regression.fit(data[["Impressions"]], data["Saves"])
plt.plot(data["Impressions"], regression.predict(data[["Impressions"]]), color='red')

plt.show()


# In[14]:


correlation = data.corr()
print(correlation["Impressions"].sort_values(ascending=False))


# In[15]:


conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)


# In[41]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Assuming 'data' is the DataFrame containing the relevant data

sns.set_style("whitegrid")
plt.figure(figsize=(10, 8))
plt.scatter(data["Profile Visits"], data["Follows"], s=data["Follows"], alpha=0.7)
plt.title("Relationship Between Profile Visits and Followers Gained")
plt.xlabel("Profile Visits")
plt.ylabel("Follows")

# Fit and plot the trendline
regression = LinearRegression()
regression.fit(data[["Profile Visits"]], data["Follows"])
plt.plot(data["Profile Visits"], regression.predict(data[["Profile Visits"]]), color='red')

plt.show()


# In[17]:


x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)


# In[18]:


model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)


# In[19]:


features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
model.predict(features)


# In[ ]:




