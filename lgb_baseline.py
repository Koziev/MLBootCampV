# -*- coding: utf-8 -*-
'''
Черновой вариант решения задачи конкурса http://mlbootcamp.ru/round/12/tasks/
Используется LightGBM.
Неочищенные данные.
При сабмите дает примерно 0.5452
(c) Козиев Илья inkoziev@gmail.com
'''

import os
import pandas as pd
import numpy as np
import codecs
import math
import random
import numpy as np
import sklearn.ensemble
import sklearn.tree
import sklearn.metrics
import sklearn.utils
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from pylightgbm.models import GBMClassifier
import sklearn.preprocessing
import sklearn.decomposition
import gc
import cleanup


DO_CV = True
DO_TRAIN = False
MAKE_SUBMISSION = False

N_SPLIT = 7
LEARNING_RATE = 0.005
EARLY_STOPPING = 50
NB_ROUNDS = 1000


# full path to lightgbm executable (on Windows include .exe)
exec_path = "~/LightGBM/lightgbm"

# Папка с исходными данными задачи
#data_folder = r'e:\MLBootCamp\V\input'
data_folder = r'/home/eek/polygon/MLBootCamp/V/input'

# Папка для сохранения результатов
submission_folder = r'./submissions'

# ---------------------------------------------------------------------

def store_submission( y_submission, submission_filename ):
    with codecs.open(submission_filename, 'w') as wrt:
        for idata,y in enumerate(y_submission):
            wrt.write('{}\n'.format(y))

# ---------------------------------------------------------------------

def load_data():

    train_data = pd.read_csv(os.path.join(data_folder, 'train.csv'), delimiter=';', skip_blank_lines=True)
    test_data = pd.read_csv(os.path.join(data_folder, 'test.csv'), delimiter=';', skip_blank_lines=True,
                            na_values='None')

    ntrain = train_data.shape[0]
    ntest = test_data.shape[0]

    print('ntrain={}'.format(ntrain))
    print('ntest={}'.format(ntest))

    y_train = train_data['cardio'].values

    # --------------------------------------------------------------

    x_train = train_data.drop(["id", "cardio"], axis=1)
    x_test = test_data.drop(["id"], axis=1)

    x_test.replace('None', np.nan)

    return (x_train,y_train,x_test)

# ---------------------------------------------------------------------

x_data,y_data,x_test = load_data()

# --------------------------------------------------------------

#<editor-fold desc="Запуск LightGBM">

_seed = random.randint(1, 2000000000)

X_train, X_test, y_train, y_test = train_test_split( x_data, y_data, test_size = 0.10, random_state = _seed)

cl = GBMClassifier(exec_path=exec_path,
                   boosting_type='gbdt', # gbdt | dart | goss
                   learning_rate=LEARNING_RATE,
                   num_leaves=64,
                   min_data_in_leaf=1,
                   min_sum_hessian_in_leaf=1e-4,
                   num_iterations=5000,
                   num_threads=4,
                   early_stopping_round=EARLY_STOPPING,
                   drop_rate=0.0001,
                   max_depth=6,
                   lambda_l1=0.,
                   lambda_l2=0.,
                   max_bin=63,
                   feature_fraction=1.0,
                   #bagging_fraction=0.5,
                   #bagging_freq=3,
                   verbose=True
                   )
cl.fit(X_train, y_train, test_data=[(X_test, y_test)])

#</editor-fold>


#<editor-fold desc="Генерация сабмита">

if MAKE_SUBMISSION:
    print('Computing submission probabilities...')
    y_submission = cl.predict_proba(x_test)[:,1]
    print('Store submission data')
    submission_filename = os.path.join(submission_folder, 'submission_lightgbm.dat')
    store_submission(y_submission, submission_filename)
    print('Submission data have been stored in {}\n'.format(submission_filename))

#</editor-fold>
