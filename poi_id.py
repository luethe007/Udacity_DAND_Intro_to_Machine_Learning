#!/usr/bin/python

""" a script for Machine Learning POI identifier,
    a new feature is created and the final features are selected,
    different algorithms are trained and tested,
    the final classifier, data set, and features_list are stored in .pkl-format
"""

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


all_features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'deferred_income',
                  'director_fees', 'exercised_stock_options', 'expenses',
                  'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi',
                  'loan_advances', 'long_term_incentive', 'other', 'restricted_stock',
                  'restricted_stock_deferred', 'shared_receipt_with_poi', 'to_messages',
                  'total_payments', 'total_stock_value']

## 1. Create own Feature
## 1.1 Load the data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers
data_dict.pop('TOTAL', 0)
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, all_features_list, sort_keys = True, remove_all_zeroes=False)
labels, features = targetFeatureSplit(data)

## 1.2 Engineer own Feature with PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import RandomizedPCA
sc = StandardScaler()
features_std= sc.fit_transform(features)

pca = RandomizedPCA(n_components=3)
features_std_pca = pca.fit_transform(features_std)

## 1.3 Add Feature to my_dataset
i = 0
for person in my_dataset:
    if i <= len(my_dataset):
        my_dataset[person]['PCA_feature'] = features_std_pca[i][0]
        i += 1

new_all_features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'deferred_income',
                  'director_fees', 'exercised_stock_options', 'expenses',
                  'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi',
                  'loan_advances', 'long_term_incentive', 'other', 'restricted_stock',
                  'restricted_stock_deferred', 'shared_receipt_with_poi', 'to_messages',
                  'total_payments', 'total_stock_value', 'PCA_feature']

## 2. Feature Selection
## 2.1 Load the data again (new Features added)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_all_features_list, sort_keys = True, remove_all_zeroes=False)
labels, features = targetFeatureSplit(data)


## 2.2 Select the best Features with SelectKBest
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=1)
selector.fit(features_train, labels_train)
print("scores_:", selector.scores_)
print('selected index', selector.get_support(True))

## see what is meant by the indices
count = 0
for count in range(1, len(new_all_features_list)):
    print count-1, new_all_features_list[count]


# selected with SelectKBest (best feature)
features_list = ['poi', 'bonus']

## 3. Pick an algorithm
## 3.1 Load the data again (best Features selected)
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_all_zeroes=False)
labels, features = targetFeatureSplit(data)
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

##Tune the Decision Tree Algorithm
#param_grid = {'max_features': [1, 2, 3, 4],
              #'min_samples_split': [2, 3, 10]
              #}
##clf = GridSearchCV(DecisionTreeClassifier(), param_grid)
##clf.fit(features_train, labels_train)
##print clf.best_estimator_

## Create the best Decision Tree Algorithm
clf = DecisionTreeClassifier(min_samples_split=10)
clf.fit(features_train, labels_train)

''' These algorithms were tested, but not selected:
## Create the Naive Bayes Algorithm
#clf = GaussianNB()
#clf.fit(features_train, labels_train)

## Tune the Adaboost Algorithm
param_grid = {'n_estimators': [30, 40, 50, 60, 70],
              'learning_rate': [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
              'algorithm': ['SAMME', 'SAMME.R']
              }
clf = GridSearchCV(AdaBoostClassifier(), param_grid)
clf.fit(features_train, labels_train)
print clf.best_estimator_

## Create the best Adaboost Algorithm
#clf = AdaBoostClassifier(algorithm='SAMME', learning_rate=0.6)
clf = AdaBoostClassifier()
clf.fit(features_train, labels_train)

## Tune the Random Forest Algorithm
param_grid = {'max_depth': [3, None],
              'max_features': [1, 2, 3, 4],
              'min_samples_split': [2, 3, 10],
              'min_samples_leaf': [1, 3, 10],
              'bootstrap': [True, False],
              'criterion': ['gini', 'entropy']
              }
clf = GridSearchCV(RandomForestClassifier(), param_grid)
clf.fit(features_train, labels_train)
print clf.best_estimator_

## Create the best Random Forest Algorithm
#clf = RandomForestClassifier(max_depth=3, max_features=1, min_samples_split=10, bootstrap=False, criterion='entropy')
clf.fit(features_train, labels_train)
'''

### 4. Evaluation: target at least .3 precision and recall (acc. to tester.py)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)

print 'And the Accuracy is: ', acc
print 'Number of features: ', len(features_train[0])

print classification_report(labels_test, pred)

### 5. Dump the classifier, dataset, and features_list

dump_classifier_and_data(clf, my_dataset, features_list)