# -*- coding: utf-8 -*-
'''
Черновой вариант решения задачи конкурса http://mlbootcamp.ru/round/12/tasks/
Случайный подбор гиперпараметров для XGBoost
Неочищенные данные
Дает при сабмите примерно 0.5436
(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import print_function
import codecs
import random
import numpy as np
import pandas as pd
import xgboost
import os
import colorama
import platform

N_SPLIT = 5
EARLY_STOPPING = 50

# Папка с исходными данными задачи
data_folder = r'e:\MLBootCamp\V\input'
#data_folder = r'/home/eek/polygon/MLBootCamp/V/input'

# Папка для сохранения результатов
submission_folder = r'./submissions'


# ---------------------------------------------------------------------

def store_submission( y_submission, submission_filename ):
    with codecs.open(submission_filename, 'w') as wrt:
        #wrt.write('test_id,is_duplicate\n')
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

if platform.system()=='Linux':
    pass
    USE_GPU = True
    print(colorama.Fore.GREEN + 'Using GPU'+colorama.Fore.RESET)
else:
    colorama.init()
    USE_GPU = False
    print(colorama.Fore.RED+'Using CPU'+colorama.Fore.RESET)

# --------------------------------------------------------------

x_data,y_data,x_submit = load_data()

# --------------------------------------------------------------

for colname in x_data.columns.values:
    train_vals = x_data[colname].values
    test_vals = x_data[colname].values

    print('\ncolname={}'.format(colname))
    print('train ==> min={} max={} avg={}'.format( np.nanmin(train_vals), np.nanmax(train_vals), np.nanmean(train_vals) ))
    print('test ==> min={} max={} avg={}'.format( np.nanmin(test_vals), np.nanmax(test_vals), np.nanmean(test_vals) ))

# --------------------------------------------------------------

D_train = xgboost.DMatrix(x_data, label=y_data, feature_names=x_data.columns.values)
D_submit = xgboost.DMatrix(x_submit, feature_names=x_submit.columns.values)

# ---------------------------------------------------------------------

#<editor-fold desc="Запуск XGBOOST">

model_log = codecs.open( os.path.join(submission_folder, 'xgboost_model.log'),'w')

probe_count = 0
cur_best_loss = np.inf
while True:
    probe_count += 1
    print('\nprobe #{} cur_best_loss={:7.5f}'.format(probe_count,cur_best_loss) )

    # случайный выбор параметров
    _n_estimators = 10000
    _subsample = random.uniform(0.85, 0.95)
    _max_depth = random.randint(4, 6)
    _seed = random.randint(1, 2000000000)
    _min_child_weight = 0
    _colsample_bytree = random.uniform(0.85, 1.00)
    _colsample_bylevel = random.uniform(0.85, 1.00)
    _learning_rate = random.uniform(0.008, 0.013)
    _gamma = 0.01

    xgb_params = {
        'booster': 'gbtree',  # 'dart' | 'gbtree',
        'subsample': _subsample,
        'max_depth': _max_depth,
        'seed': _seed,
        'min_child_weight': _min_child_weight,
        'eta': _learning_rate,
        'gamma': _gamma,
        'colsample_bytree': _colsample_bytree,
        'colsample_bylevel': _colsample_bylevel,
        'scale_pos_weight': 1.0,
        'eval_metric': 'logloss', #'auc', #''logloss',
        'objective': 'binary:logistic',
        'silent': 1,
    }

    #if xgb_params['booster'] == 'dart':
    #    xgb_params['rate_drop'] = 0.1
    #    xgb_params['one_drop'] = 1

    if USE_GPU:
        xgb_params['updater'] = 'grow_gpu'

    for pname,pval in xgb_params.iteritems():
        model_log.write('{}={}\n'.format(pname,pval))
    model_log.flush()

    # ----------------------------------------------------------------
    cvres = xgboost.cv(xgb_params,
                       D_train,
                       num_boost_round=5000,
                       nfold=N_SPLIT,
                       early_stopping_rounds=EARLY_STOPPING,
                       metrics='logloss',
                       seed=_seed,
                       #callbacks=[eta_cb],
                       #verbose_eval=50,
                       show_stdv=False
                       )
    #cvres.to_csv(os.path.join(submission_folder, 'cvres.csv'))
    nbrounds = cvres.shape[0]
    print('CV finished, nbrounds={}'.format(nbrounds))
    model_log.write('cv nbrounds={}\n'.format(nbrounds))

    cv_logloss = cvres['test-logloss-mean'].tolist()[-1]
    print('cv.test.logloss_mean={}'.format(cv_logloss))

    model_log.write('cv.test.logloss_mean={}\n'.format(cv_logloss))
    model_log.flush()

    if cv_logloss<cur_best_loss:
        cur_best_loss = cv_logloss
        print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(cur_best_loss) + colorama.Fore.RESET)

        print('Train model for nbrounds={}...'.format(nbrounds))
        cl = xgboost.train(xgb_params, D_train, num_boost_round=nbrounds, verbose_eval=False )

        print('Compute submission...')
        y_submission = cl.predict(D_submit, ntree_limit=nbrounds)
        submission_filename = os.path.join(submission_folder, 'submission_xgboost_loss={}_nbround={}.dat'.format(cv_logloss,nbrounds))
        store_submission(y_submission, submission_filename)

    model_log.write('\n{}\n'.format(50*'-'))
    model_log.flush()
    print('')

#</editor-fold>
