#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Implements various classifiers for classification task.

Author: Pranjal Dhole
E-mail: dhole.pranjal@gmail.com
'''
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from molecule_gym.core.visualizations import show_confusion_matrix

def classify_binary_data(X, y, classifier='svc', sampling=None):
    '''
    Classifies binary data with numeric features based on input classifier.
    '''

    X_train, y_train, X_test, y_test = get_training_test_sets(X, y, sampling=sampling)

    print(f"Training target statistics: {Counter(y_train)}")
    print(f"Testing target statistics: {Counter(y_test)}")

    if classifier=='svc':
        print('Using SVC classifier')
        lr_clf = SVC()
    else:
        raise AssertionError(f'{classifier} is not implemented!')
    lr_clf.fit(X_train, y_train)
    
    train_pred = lr_clf.predict(X_train)
    print('\nModel performance on training data:')
    print(classification_report(y_train, train_pred))
    
    predictions = lr_clf.predict(X_test)
    print('\nModel performance on test data:')
    print(classification_report(y_test, predictions))
    
    show_confusion_matrix(y_test, predictions)

def get_training_test_sets(X, y, sampling=None):
    '''
    Generates training and test split based on unbalanced data sampling methods.
    '''
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.3)
    if sampling == 'oversample':
        over_sampler = RandomOverSampler(random_state=42)
        X_train, y_train = over_sampler.fit_resample(train_features, train_labels)
        X_test, y_test = over_sampler.fit_resample(test_features, test_labels)
    elif sampling == 'undersample':
        under_sampler = RandomUnderSampler(random_state=42)
        X_train, y_train = under_sampler.fit_resample(train_features, train_labels)
        X_test, y_test = under_sampler.fit_resample(test_features, test_labels)
    else:
        X_train, y_train = train_features, train_labels
        X_test, y_test = test_features, test_labels
    return X_train, y_train, X_test, y_test