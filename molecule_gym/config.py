#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Default config variables which maybe overridden by a user config.

Author: Pranjal Dhole
E-mail: dhole.pranjal@gmail.com
'''
import os.path as osp

# ===========================================
# PATHS
# ===========================================
PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))
DATA_DIR = osp.join(PROJECT_PATH, 'data')
RESULTS_DIR = osp.join(PROJECT_PATH, 'results')

# ===========================================
# DATASETS USED IN THIS LIBRARY
# ===========================================
DATA_LIST = ['BBBP', 'clintox', 'sider', 'tox21', 'toxcast']
