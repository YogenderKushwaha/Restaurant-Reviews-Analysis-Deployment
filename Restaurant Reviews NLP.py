#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ## Loading Dataset

# In[2]:


df = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t')  


# In[3]:


print(df.shape)
df.head()


# In[ ]:





# ### Checking for Null values 

# In[4]:


df.isnull().sum()


# In[5]:


messages=df.copy()


# In[6]:


messages.head(10)


# ## Cleaning Data

# In[7]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


# In[8]:


ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['Review'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[9]:


corpus[5]


# ## Preprocessing Data Using TF IDF Vectorizer

# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[11]:


tfidf_v=TfidfVectorizer(max_features=1500,ngram_range=(1,3))
X=tfidf_v.fit_transform(corpus).toarray()


# In[12]:


tfidf_v.get_params()


# In[13]:


X.shape


# In[14]:


y=messages['Liked']


# ##  Train and Test Split

# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[16]:


count_df = pd.DataFrame(X_train, columns=tfidf_v.get_feature_names())


# In[17]:


count_df.head()


# ## Training Model

# In[18]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import itertools


# ### Multinomial Classifier with Hyperparameter Tuning

# In[19]:


classifier=MultinomialNB(alpha=0.1)


# In[20]:


previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(alpha,score))


# ### Multinomial Classifier with Updaated Parameter

# In[21]:


classifier=MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)


# ## Classification Report

# In[22]:


from sklearn.metrics import classification_report

model_score= (classification_report(y_test, pred))
print(model_score)


# ## Confusion Matrix

# In[23]:


import matplotlib.pyplot as plt 
from sklearn.metrics import plot_confusion_matrix
 
plot_confusion_matrix(classifier , X_test, y_test) 
plt.show()


# ## AUC Score

# In[24]:


from sklearn.metrics import plot_roc_curve

disp=plot_roc_curve(classifier,X_test, y_test);


# In[ ]:





# In[25]:


import pickle 
pickle_out= open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()


# In[ ]:




