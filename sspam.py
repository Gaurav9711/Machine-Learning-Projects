# -*- coding: utf-8 -*-
"""
Created on Thu May 27 17:10:32 2021

@author: Gajender Kumar
"""

import pandas as pd
messages=pd.read_csv('SMSSpamCollection',sep='\t',names=['labels','message'])
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
wordnet=WordNetLemmatizer()
ps= PorterStemmer()
corpus=[]

for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]','', messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[wordnet.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review=''.join(review)
    corpus.append(review)
    
    
    
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
X=tfidf.fit_transform(corpus).toarray()    
y=pd.get_dummies(messages['labels'])
y=y.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=101)


from sklearn.naive_bayes  import MultinomialNB
multinomial=MultinomialNB().fit(X_train,y_train)
y_pred=multinomial.predict(X_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)

