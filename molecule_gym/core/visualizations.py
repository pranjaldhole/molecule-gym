#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Visualizations used to plot data statistics.

Author: Pranjal Dhole
E-mail: dhole.pranjal@gmail.com
'''
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def show_confusion_matrix(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='4g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Permeability of drugs through blood-brain-barrier')
    plt.show()

def plot_explained_variance(exp_var_pca):
    '''
    Plots principal component contribution in order of their explained variance.
    '''
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
