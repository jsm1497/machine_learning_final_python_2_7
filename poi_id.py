#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint

from pandas.api.types import is_numeric_dtype

from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn import preprocessing
from sklearn import utils

from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from mlxtend.plotting import plot_decision_regions

from tester import dump_classifier_and_data
from feature_format import featureFormat, targetFeatureSplit

from collections import defaultdict

sys.path.append("../tools/")

get_ipython().magic(u'matplotlib inline')


# In[2]:


## declare simple data types

adj_types = ["unscaled", "unscaled - custom", "scaled", "scaled - custom"]

text_shape = "The original dataset has {0} rows and {1} columns"
text_class_dist = "The original dataset has {0} POI records and {1} non-POI records"
text_nan = "The following table is the count of NaN values in the original dataset."
text_final_clf = "The best performing classifier was the {0} using {1} features with an F1-score of {2}."
text_other_clf = "{0} classifier had an F1-score of {1}"
text_dataset_size = ("The original dataset was resampled to add 50 POI records, to assist with the imbalanced classes. After being split into training and testing data"
                     " the training dataset had {0} rows and the testing dataset had {1} rows.")
text_comp_perf = ("The dataset with the highest performance used {2} features. With this dataset, my custom feature's best classifier had an F1-score of {0},"
                  " while the standard feature's best classifier had an F1-score of {1}.")

text_final_features = ("The best performing classifier used {0} features. "
                       "Below are the most important features and their relative importance according to the ExtraTreeClassifier. {1}"
                       )
text_gt_10_features = "Note: Because there are more than 10 final features, I am only including the first 10 below."

trn_set_size = 0
tst_set_size = 0


# In[3]:


## declare objects

datasets = []
clf_models = []
f1_score = []
cr_dict = {}

df_nan = None

desc_variables = {"text_shape": text_shape, "text_class_dist": text_class_dist, "df_nan": df_nan,
                  "text_nan": text_nan, "trn_set_size": trn_set_size, "tst_set_size": tst_set_size}


# In[4]:


# process data
# process_data function accepts a few different arguments, allowing me to scale, resample, and filter the features
# https://elitedatascience.com/imbalanced-classes


def process_data(data, label_column='poi', scale=0, rsmpl=0, feature_list=None):
    # Resample first, if needed
    # https://stackoverflow.com/questions/52735334/python-pandas-resample-dataset-to-have-balanced-classes

    if rsmpl:

        df_majority = data[data[label_column] == 0]
        df_minority = data[data[label_column] == 1]

        # Upsample minority class
        df_minority_upsampled = utils.resample(df_minority.copy(),
                                               n_samples=50,
                                               replace=True     # sample with replacement
                                               )

        # because we are resampling, and we need to export the data to a dictionary, we need to add some
        # randomness to the resampled names
        new_index_names = []
        for i, j in enumerate(df_minority_upsampled.index):
            new_index_names.append(j + '_' + str(np.random.random()))

        df_minority_upsampled.index = new_index_names

        # Combine majority class with upsampled minority class
        data = pd.concat([df_majority, df_minority_upsampled, df_minority])

    # separate features from labels

    features = data.drop(columns=[label_column])
    labels = data[label_column]

    if feature_list:
        features = features[feature_list]

# adjusted scaler to use StandardScaler, as found in this post on sklearn
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    if scale:
        for x in features.columns:
            if is_numeric_dtype(features[x]):
                features[x] = preprocessing.StandardScaler().fit_transform(
                    features[x].values.reshape(-1, 1))

    features = features.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, stratify=labels)

    desc_variables["trn_set_size"] = len(X_train)
    desc_variables["tst_set_size"] = len(X_test)

    return X_train, X_test, y_train, y_test, data


# In[5]:


# https://machinelearningmastery.com/feature-selection-machine-learning-python/

# Feature Importance with Extra Trees Classifier


def get_best_features(X_train, X_test, y_train, percentile):
    et_clf = ExtraTreesClassifier(n_estimators=100)
    et_clf.fit(X_train, y_train)

    df_feature_importance = pd.DataFrame(
        et_clf.feature_importances_, index=X_train.columns)
    df_feature_importance = df_feature_importance.sort_values(
        by=0, ascending=False)
    df_feature_importance = df_feature_importance[df_feature_importance[0] >= percentile].reset_index(
    )

    feature_list = df_feature_importance['index']

    X_train = X_train[feature_list]
    X_test = X_test[feature_list]

    return X_train, X_test, df_feature_importance


# In[6]:


def get_best_features_and_append_datasets(adj_type):
    if 'custom' in adj_type:
        perc = .0
    else:
        perc = .06

    if 'scale' in adj_type:
        __X_train, __X_test, best_features = get_best_features(
            X_train_scale, X_test_scale, y_train_scale, perc)

        datasets.append((__X_train, __X_test, y_train_scale, y_test_scale,
                         best_features, adj_type, df_scale))
    else:
        __X_train, __X_test, best_features = get_best_features(
            X_train_std, X_test_std, y_train_std, perc)

        datasets.append((__X_train, __X_test, y_train_std, y_test_std,
                         best_features, adj_type, df_std))


# In[7]:


def run_all_datasets():
    for X_train, X_test, y_train, y_test, feature_list, adj_type, df_loop in datasets:
        for clf, params in clf_models:

            if params:
                Grid_CV = GridSearchCV(clf, params, cv=5, iid=True)
                gv = Grid_CV.fit(X_train, y_train)
                clf.set_params(**gv.best_params_)

            clf.fit(X_train, y_train)
            clf.score(X_test, y_test)

            y_pred = clf.predict(X_test)

            f1_score.append((clf, df_loop, feature_list, adj_type,
                                  metrics.f1_score(y_test, y_pred)))

            cr = classification_report(y_test, y_pred, output_dict=True)
            if adj_type not in cr_dict:
                cr_dict[adj_type] = {clf.__class__.__name__:cr}
            else:
                cr_dict[adj_type][clf.__class__.__name__] = cr


# In[8]:


def process_initial_data():

    # Load the dictionary containing the dataset
    # with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

    # Load the dictionary containing the processed text from the emails
    # Code is in vectorize_text.py - heavily borrowed from the scripts used in the text learning module
    # as well as the email_preprocess module
    email_dict = pickle.load(open("email_text.pkl", "r"))

    # convert dictionary to dataframe for easier manipulation

    df = pd.DataFrame.from_dict(data_dict, orient="index")

    # adjust email dataset to format needed to join to df
    for x, y in email_dict.items():
        email_dict[x] = ''.join(y)

    df_email_text = pd.DataFrame.from_dict(email_dict, orient="index")

    df_email_text.rename(index=str, columns={0: 'email_text'}, inplace=True)

    # describe data
    # Understanding the Dataset and Question

    # Task 2: Remove outliers
    # fillna did not work (likely due to NaN being strings, not np.nan), so use replace

    df.replace("NaN", np.nan, inplace=True)
    df.poi = df.poi.astype(int)

    if len(df[df.index == "TOTAL"].values):
        df.drop("TOTAL", inplace=True)

    # Record initial data exploration variables, such as shape of data and number of NANs

    x, y = df.shape

    desc_variables["text_shape"] = text_shape.format(x, y)

    desc_variables["text_class_dist"] = text_class_dist.format(
        len(df[df.poi == 1].values), x - len(df[df.poi == 1].values))

    desc_variables["df_nan"] = pd.DataFrame(
        df.isna().sum()).rename(index={0: 'NANs'})

    # Create joined dataset

    df = pd.merge(left=df, right=df_email_text, how='left',
                  left_on='email_address', right_index=True).drop(columns=['email_address'])

    # set up TfIdf Vectorizer for converting email text to features

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.50,
                                 stop_words='english', max_features=5000)

    email_transformed = vectorizer.fit_transform(df['email_text'].fillna(''))

    df_email_transformed = pd.DataFrame(
        email_transformed.toarray(), columns=vectorizer.get_feature_names())

    return df.join(df_email_transformed, rsuffix='et_').drop(columns=['email_text']).fillna(0)


# In[9]:


# split data into training and testing sets
# return df as well, since we modify columns and data if we resample, scale, or remove features

df = process_initial_data()

X_train_std, X_test_std, y_train_std, y_test_std, df_std = process_data(
    df.copy(), rsmpl=1)
X_train_scale, X_test_scale, y_train_scale, y_test_scale, df_scale = process_data(
    df.copy(), rsmpl=1, scale=1)


# In[10]:


for x in adj_types:
    get_best_features_and_append_datasets(x)


# In[11]:


# Add all models to a list, with parameters for tuning if needed, to be fit and scored at the end

# Naive Bayes

gb = GaussianNB()

clf_models.append((gb, None))

# SVM
SVC_clf = SVC(kernel='rbf', C=1000, gamma='scale', random_state=12)

parameters = {"kernel": ['rbf', 'poly', 'sigmoid'], "C": [.1,
                                                          10, 100, 1000], "gamma": ['scale'], "random_state": [40]}

clf_models.append((SVC_clf, parameters))

# Decision Tree

dt_clf = tree.DecisionTreeClassifier()

parameters = {"min_samples_split": [2, 3], "max_features": [
    None, 2, 5, ], "random_state": [40]}

clf_models.append((dt_clf, parameters))


# In[12]:


# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html


# In[13]:


# https://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case


# In[14]:


# run feature tuning and scoring on all datasets in datasets list
del f1_score[:]

run_all_datasets()


# In[15]:


# get max by type - do this so we can compare custom features to standard features

max_by_type = defaultdict(lambda: defaultdict(int))

i = 0
for __clf, __df, __features, __adj_type, __f1_score in f1_score:
    if __f1_score > max_by_type[__adj_type]["max"]:
        max_by_type[__adj_type]["max"] = __f1_score
        max_by_type[__adj_type]["max_i"] = i
        max_by_type[__adj_type]["features"] = __features
        max_by_type[__adj_type]["df"] = __df
        max_by_type[__adj_type]["clf"] = __clf
    i += 1

# get overall best performing classifier

mx = 0
mx_adj_type = ""

for x, y in max_by_type.items():
    if y.get("max") > mx:
        mx = y.get("max")
        mx_adj_type = x


final_clf, final_df, final_features, final_adj_type, final_f1_score = f1_score[max_by_type.get(
    mx_adj_type).get("max_i")]


# In[16]:


exp_feature_list = list(final_features['index'].values)
if exp_feature_list[0] != 'poi':
    exp_feature_list.insert(0, 'poi')

dump_classifier_and_data(
    final_clf, final_df.to_dict('index'), exp_feature_list)


# In[17]:


text_final_clf = text_final_clf.format(final_clf.__class__.__name__,
                                       final_adj_type, round(final_f1_score, 4))


# In[18]:


standard_adj_type = final_adj_type.split(' ')[0]

custom_adj_type = final_adj_type.split(' ')[0] + ' - custom'

custom_features_f1_score = f1_score[max_by_type.get(
    custom_adj_type).get("max_i")][4]

unscaled_features_f1_score = f1_score[max_by_type.get(
    standard_adj_type).get("max_i")][4]

text_comp_perf = text_comp_perf.format(round(custom_features_f1_score, 4), round(
    unscaled_features_f1_score, 4), standard_adj_type)


# In[19]:


text_dataset_size = text_dataset_size.format(
    desc_variables["trn_set_size"], desc_variables["tst_set_size"])


# In[20]:


if len(final_features) > 10:
    text_final_features = text_final_features.format(
        len(final_features), text_gt_10_features)
    disp_final_features = final_features[:10]
else:
    text_final_features = text_final_features.format(len(final_features), '')
    disp_final_features = final_features


# In[21]:


other_clf_perf = []

for __clf, __df, __features, __adj_type, __f1_score in f1_score:
    if __adj_type == final_adj_type and __clf is not final_clf:
        other_clf_perf.append(text_other_clf.format(__clf.__class__.__name__,round(__f1_score,4)))
        


# In[22]:


text_clf_params = "{0} classifier the parameters that were tuned were {1}"

all_clf_params = []

for __clf, __params in clf_models:
    if __params:
         all_clf_params.append(text_clf_params.format(__clf.__class__.__name__,', '.join(__params.keys())))


# In[23]:


## https://stackoverflow.com/questions/24988131/nested-dictionary-to-multiindex-dataframe-where-dictionary-keys-are-column-label

reform = {(outerKey, innerKey, innerMostKey): values for outerKey, innerDict in 
          cr_dict.iteritems() for innerKey, innerMostDict in innerDict.iteritems()
          for innerMostKey, values in innerMostDict.iteritems() if 'avg' not in innerMostKey}

cr = pd.DataFrame().from_dict(reform,'index')

cr.index = cr.index.set_names(['adj_type','clf','class'])

cr = cr[['recall','precision','f1-score']]

cr = cr.round(4)


# ## Files
# parse_out_email_text.py - from Text Learning module
# 
# vectorize_text.py - based on script from Text Learning module
# 
# email_text.pkl - output of vectorize_text.py
# 
# poi_id.py - final project
# 
# tester.py - test script (Note: I could not get this to run, there were multiple errors with StratifiedShuffleSplit that I did not want to take the time to try and troubleshoot. However, it did run to the point of pulling in my data and attempting to score it, so that was enough validation for me)
# 
# 
# ## Questions
# 
# ##### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]
# 
# The goal of this project is to use the given data that we have about the Enron employees financial and email data to try and determine which were/would have been persons of interest in the investigation. Machine learning is a very useful tool here, because there is a lot of data (especially in the emails) and a lot of patterns - the financial data alone has roughly 20 features. It is hard for humans to find patterns out of this much data, so we turn to machine learning to help.
# 
# There are multiple ways to use the data to determine if someone is a POI - I attempted two of them. The first is to use this financial + email dataset to see if there are any patterns in the financial or email numbers that correlate with someone being a POI. Another way of looking at this is to use the actual email text to find patters - when I did this in my vectorize_text.py script, I was getting a score of around 90, with a recall of .5 and a precision of .89 - which was a much simpler and quicker way of getting high accuracy compared to using this dataset.
# 
# The only outlier I noticed was the "Total" record, which had numbers way higher than any other record. Once I removed this record, I didn't see any other large outliers, so I left the rest of the potential outliers in.
# 
# ##### Data exploration
# 
# {{desc_variables["text_shape"]}}
# 
# {{desc_variables["text_class_dist"]}}
# 
# {{desc_variables["text_nan"]}}
# 
# {{desc_variables["df_nan"]}}
# 
# ##### What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]
# 
# I used an ExtraTreeClassifier to determine the features to use - I used a 60% percentile for the "standard" features (data + financial) and used a 0% percentile for my custom features - as I found that these were consistently marked as 0.0 performance.
# 
# I ran three different classifiers against four different datasets - each combination of the standard features vs standard plus my custom features, as well as scaled vs unscaled. The best performing classifier ended up using the {{final_adj_type}} dataset.
# 
# My custom features were generated using the text from the emails that each person sent - based on what was in the emails_by_address folder. I then ran these through a TfIdf vectorizer. My assumption was that this should give quite a bit more power to the classifier, as POIs would probably use some words more often than non-POIs. As we'll see below, the datasets with the custom features have a better average Recall score than the dataset that uses only the standard features.
# 
# ##### The below stats are for the last run completed.
# 
# {{text_comp_perf}}
# 
# {{text_final_features}}
# 
# {{disp_final_features}}
# 
# 
# ##### What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]
# 
# I found that the SVC and the Decision Tree Classifier gave me the best balanced score, on average. This being said, I wrote my script to return the classifier that gave me the best balanced score, so I never chose any one algorith per se, I let the metrics choose for me. Besides the SVC and Decision Tree, I also used a Gaussian Niave Bayes classifier, however this consistently performed much lower than the others.
# 
# ##### The below stats are for the last run completed.
# 
# {{text_final_clf}}
# 
# With this same set of features, the {{other_clf_perf[0]}}, and the {{other_clf_perf[1]}}.
# 
# ##### What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]
# 
# Tuning an algorithm is to find what parameters for that algorithm give you the best performance. Without tuning your algorithms, you can have much worse performance.
# 
# With the exception of the Niave Bayes classifier, I tuned all of my classifiers using the GridSearchCV estimator, passing in a defined list of parameters and potential values. For the {{all_clf_params[0]}}, and for the {{all_clf_params[1]}}.
# 
# 
# ##### What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]
# 
# Validation is how you test that your model works on actual data. The most important thing here is to have, at the least, a separate training and testing data set. If you do your testing on your training data, you are not going to see how your model performs on real data, because it was already trained on the same data you are using for testing.
# 
# I validated my data by splitting it into separate training and testing data sets. {{text_dataset_size}}
# 
# ##### Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]
# 
# There are two evaluation metrics I pay a lot of attention to, precision and recall. The reason I chose to ignore the general accuracy score is that it does not give a great representation when you have imbalanced classes, which is what we had in this dataset. However, the F1 score is another great evaluation metric, especially for imbalanced. It is the harmonic mean of precision and recall. Based on my research, this is a very good metric to use when classifying imbalanced classes with few positive records, like what we have here.
# 
# Precision is the ratio of how many times the model __accurately__ predicted the record was a POI record, compared to all of the times it predicted the the record was a POI record in total. For example, if it predicted 3 non-POI records were POI records, and it predicted 7 POI records were POI records, this would be a precision rate of .7.
# 
# Recall is a bit more important in this model, in my opinion. Recall is the ratio of how many times, when encountering a POI record, did it __correctly__ classify that it was a POI record.
# 
# Below I have the average for recall, precision, and F1 for the POI class, grouped by both the dataset and by the classifier, as well as by both.
# 
# 
# #### The below stats are for the last run completed.
# 
# ##### By feature type (unscaled/scaled, standard/custom)
# 
# {{cr[cr.index.get_level_values('class')=='1'].groupby('adj_type').mean()}}
# 
# ##### By classifier
# 
# {{cr[cr.index.get_level_values('class')=='1'].groupby('clf').mean()}}
# 
# ##### By both
# 
# {{cr[cr.index.get_level_values('class')=='1'].groupby(['adj_type','clf']).mean()}}

# In[ ]:




