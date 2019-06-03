
# Note - all Python files were modified/written to work in Python 3

## Files
parse_out_email_text.py - from Text Learning module

vectorize_text.py - based on script from Text Learning module

email_text.pkl - output of vectorize_text.py

poi_id.py - final project

tester.py - test script (Note: I could not get this to run, there were multiple errors with StratifiedShuffleSplit that I did not want to take the time to try and troubleshoot. However, it did run to the point of pulling in my data and attempting to score it, so that was enough validation for me)


## Questions

##### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The goal of this project is to use the given data that we have about the Enron employees financial and email data to try and determine which were/would have been persons of interest in the investigation. Machine learning is a very useful tool here, because there is a lot of data (especially in the emails) and a lot of patterns - the financial data alone has roughly 20 features. It is hard for humans to find patterns out of this much data, so we turn to machine learning to help.

There are multiple ways to use the data to determine if someone is a POI - I attempted two of them. The first is to use this financial + email dataset to see if there are any patterns in the financial or email numbers that correlate with someone being a POI. Another way of looking at this is to use the actual email text to find patters - when I did this in my vectorize_text.py script, I was getting a score of around 90, with a recall of .5 and a precision of .89 - which was a much simpler and quicker way of getting high accuracy compared to using this dataset.

The only outlier I noticed was the "Total" record, which had numbers way higher than any other record. Once I removed this record, I didn't see any other large outliers, so I left the rest of the potential outliers in.


##### What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

I ended up using 9 features - every feature that my ExtraTreeClassifier had listed as greater than or equal to .06 importance.

I did not do any scaling - I attempted scaling, but found that whenever I did scale the features, my performance suffered. I'm not sure if this is due a technical issue on my end, or the data itself.

I did attempt to engineer features - using the text from the emails in the emails_by_address folder. I did this work in vectorize_text.py file - I copied very heavily from the Text Learning module code, as well as email_preprocess.py from the Naive Bayes module.

The features were to be the top X number of text features from the emails that would improve the model. I used a TfIdf Vectorizer and SelectPercentile with an 80th percentile filter. However, when I ran these features through the ExtraTreeClassifier, all of these features had an importance of 0. I'm not sure what I could have done differently here, but I do know that I will need to spend more time on this the next time I attempt something like this.

Lastly, while I did not do any scaling, I did resample my POI data. The reason for this was that there are so few records in the dataset, especially POI records, that it was very hard to train my models to a rate of precision and recall higher than .3. By resampling the data, I added more training records for the POI records, which gave my models more data to learn and predict.


##### What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

I ended up going with a Decision Tree Classifier. This gave me the highest precision and recall. The Random Forest Classifier had the same precision and recall, however the Decision Tree Classifier is easier to explain and understand, so I chose this one. I also tried an SVM and a Naive Bayes classifier - the SVM struggled, and when I was looking at the produced plots, I could see why - the way it separates the data points didn't work well with the patterns in the dataset. The Naive Bayes did very well in my text vectorizer model, however it did not perform as well with the financial + email dataset.

Note - scores below will be different, but close, as the model is re-ran (This note is mostly to myself, as it took me longer than I'd like to admit to realize that the reason my scores were changing is because I was getting different records in my training and test datasets each time I re-ran this script.)

The SVM and Naive Bayes classifiers had decent precision - NB was at .83 and the SVM was at .77. While the SVM had a recall of 1, the NB classifier was only at .29. This is compared to the .89 in precision for both the Random Forest and the Decision Tree and 1 for recall, so the SVM and NB performed significantly worse.

##### What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

Tuning an algorithm is to find what parameters for that algorithm give you the best performance. Without tuning your algorithms, you can have much worse performance

I tuned the parameters for the Random Forest and SVM by hand, but I attempted to use the GridSearchCV on the decision tree classifier. After some time learning how to best use this, I ended up automating the selection of the parameters for the Decision Tree Classifier, with the parameters I tuned being min_samples_split and max_features, with a cross fold validation of 5. By using the best_params_ attribute of the GridSearchCV object, I was able to unpack these in my Decision Tree Classifier using the set_params() function.

##### What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

Validation is how you test that your model works on actual data. The most important thing here is to have, at the least, a separate training and testing data set. If you do your testing on your training data, you are not going to see how your model performs on real data, because it was already trained on the same data you are using for testing.

I validated my data by splitting it into separate training and testing data sets. I did not put the time in to do cross fold validation, although if I had to do the project again, this is what I would do.

##### Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

The two evaluation metrics I paid the most attention to where precision and recall. The reason I chose to ignore the general accuracy score is that it does not give a great representation when you have imbalanced classes, which is what we had in this dataset.

Precision is the ratio of how many times the model __accurately__ predicted the record was a POI record, compared to all of the times it predicted the the record was a POI record in total. For example, if it predicted 3 non-POI records were POI records, and it predicted 7 POI records were POI records, this would be a precision rate of .7.

Recall is a bit more important in this model, in my opinion. Recall is the ratio of how many times, when encountering a POI record, did it __correctly__ classify that it was a POI record.

On average I managed to get roughly .89 in precision, and 1 in recall on my Random Forest and Decision Tree classifiers. My SVM and NB classifiers were not quite as performant.

