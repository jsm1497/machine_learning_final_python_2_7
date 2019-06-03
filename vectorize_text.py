#!/usr/bin/env python
# coding: utf-8

# In[24]:


#!/usr/bin/python

import os
import pickle
import re
import sys
import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

from parse_out_email_text import parseOutText
from poi_email_addresses import poiEmails

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

from collections import defaultdict


# In[25]:


poi_emails = poiEmails()

from_emails_folder=r'C:\Users\jcsmi329\Documents\Homework\Udacity\Projects\Machine Learning\ud120-projects-master\final_project\emails_by_address'
base_folder=r'C:\Users\jcsmi329\Documents\Homework\Udacity\Projects\Machine Learning\ud120-projects-master'

word_data = defaultdict(list)
final_project_emails = []
email_names = set()


### Load the dictionary containing the dataset
with open(r"C:\Users\jcsmi329\Documents\Homework\Udacity\Projects\Machine Learning\final_project\final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

    
for x in data_dict.values():
    em = x.get('email_address')
    if em != "NaN":
        final_project_emails.append(em)


# In[26]:



for filename in os.listdir(from_emails_folder):

    file_email_address = filename[filename.find('_') + 1:].replace('.txt','')

    if file_email_address in final_project_emails and filename[:4] == 'from':
        email_list = open(os.path.join(from_emails_folder,filename),'r')

        for path in email_list:
            path = path.replace(r'enron_mail_20110402','').replace(r'/','',1).replace('.','_')

            path = os.path.join(base_folder, path[:-1])
            email = open(path, "r")

            if path not in email_names:
                email_names.add(path)
                t = parseOutText(email)
                word_data[file_email_address].append(t)
                email.close()

print("emails processed")



# In[30]:


output_file = r"C:\Users\jcsmi329\Documents\Homework\Udacity\Projects\Machine Learning\final_project\email_text.pkl"

pickle.dump(word_data,open(output_file,'wb'))


# In[4]:



features_train, features_test, labels_train, labels_test = model_selection.train_test_split(word_data, from_poi)



### text vectorization--go from strings to lists of numbers
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.85,
                             stop_words='english',max_features=10000)
features_train_transformed = vectorizer.fit_transform(features_train)
features_test_transformed  = vectorizer.transform(features_test)
terms = vectorizer.get_feature_names()


### feature selection, because text is super high dimensional and 
### can be really computationally chewy as a result
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train_transformed, labels_train)
features_train = selector.transform(features_train_transformed).toarray()
features_test  = selector.transform(features_test_transformed).toarray()


## https://stackoverflow.com/questions/41724432/ml-getting-feature-names-after-feature-selection-selectpercentile-python
## https://stackoverflow.com/questions/9296658/how-to-filter-a-numpy-array-using-another-arrays-values

support = np.asarray(selector.get_support(),'bool')

terms = np.asarray(terms)

selected_feature_names = terms[support]


##https://stackoverflow.com/questions/30653642/combining-bag-of-words-and-other-features-in-one-model-using-sklearn-and-pandas


# In[5]:


from sklearn.naive_bayes import GaussianNB

gb = GaussianNB()

gb.fit(features_train, labels_train)


# In[6]:


print(gb.score(features_test,labels_test))

y_pred = gb.predict(features_test)

from sklearn.metrics import classification_report

cr = classification_report(labels_test, y_pred)

print(cr)

