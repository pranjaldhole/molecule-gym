#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Implements various classifiers for classification task.

Author: Pranjal Dhole
E-mail: dhole.pranjal@gmail.com
'''
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from molecule_gym.core.visualizations import show_confusion_matrix

def classify_binary_data(X, y, classifier='svc'):
    '''
    Classifies binary data with numeric features based on input classifier.
    '''
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.3)
    print(f"Training target statistics: {Counter(train_labels)}")
    print(f"Testing target statistics: {Counter(test_labels)}")

    if classifier=='svc':
        print('Using SVC classifier')
        lr_clf = SVC()
    else:
        raise AssertionError(f'{classifier} is not implemented!')
    lr_clf.fit(train_features, train_labels)
    
    train_pred = lr_clf.predict(train_features)
    print('\nModel performance on training data:')
    print(classification_report(train_labels, train_pred))
    
    predictions = lr_clf.predict(test_features)
    print('\nModel performance on test data:')
    print(classification_report(test_labels, predictions))
    
    show_confusion_matrix(test_labels, predictions)

