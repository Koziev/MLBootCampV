# -*- coding: utf-8 -*-
'''
Поиск оптимальных значений для xgboost через hyperopt.
Оптимизируется результат предсказания модели по отдельному holdout подмножеству
сэмплов, которые модель не видела при тренировке.
Выбор числа деревьев выполняется hyperopt.

https://github.com/hyperopt/hyperopt/wiki/FMin
http://fastml.com/optimizing-hyperparams-with-hyperopt/
https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf
'''

from __future__ import print_function
import codecs
import random
import xgboost
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import time
import colorama  # https://pypi.python.org/pypi/colorama
import pandas as pd
import numpy as np
import os
import platform
import math
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from uuid import uuid4

import cleanup


N_HYPEROPT_PROBES = 5000
EARLY_STOPPING = 80
HOLDOUT_SEED = 123456
HOLDOUT_SIZE = 0.10
CV_METRICS = ['logloss']
HYPEROPT_ALGO = tpe.suggest  #  tpe.suggest OR hyperopt.rand.suggest
DATASET = 'clean' # 'raw' | 'clean' | 'extended'
SEED0 = random.randint(1,1000000000)
GAMMA0 = 0.01
MAX_DEPTH = 6
USE_GPU = False

if platform.system()=='Linux':
    #USE_GPU = True
    pass
else:
    colorama.init()
    USE_GPU = False


if USE_GPU:
    print(colorama.Fore.GREEN + 'Using GPU'+colorama.Fore.RESET)
else:
    print(colorama.Fore.RED+'Using CPU'+colorama.Fore.RESET)


# Папка для сохранения результатов
submission_folder = r'../submissions3'

# ---------------------------------------------------------------------

def store_submission( y_submission, submission_filename ):
    with codecs.open(submission_filename, 'w') as wrt:
        #wrt.write('test_id,is_duplicate\n')
        for idata,y in enumerate(y_submission):
            wrt.write('{}\n'.format(y))

# ---------------------------------------------------------------------

def calc_logloss(y_pred, y_true, epsilon=1.0e-6):
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    y_true = np.clip(y_true, epsilon, 1.0 - epsilon)
    return - np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

# ---------------------------------------------------------------------

def random_str():
    return str(uuid4())

# ---------------------------------------------------------------------

def get_xgboost_params(space):
    _max_depth = int(space['max_depth'])
    _min_child_weight = space['min_child_weight']
    _subsample = space['subsample']
    _gamma = space['gamma'] if 'gamma' in space else GAMMA0
    _eta = space['eta']
    _seed = space['seed'] if 'seed' in space else SEED0
    _colsample_bytree = space['colsample_bytree']
    _colsample_bylevel = space['colsample_bylevel']
    booster = space['booster'] if 'booster' in space else 'gbtree'

    sorted_params = sorted(space.iteritems(), key=lambda z: z[0])

    xgb_params = {
        'booster': booster,
        'subsample': _subsample,
        'max_depth': _max_depth,
        'seed': _seed,
        'min_child_weight': _min_child_weight,
        'eta': _eta,
        'gamma': _gamma,
        'colsample_bytree': _colsample_bytree,
        'colsample_bylevel': _colsample_bylevel,
        'scale_pos_weight': 1.0,
        'eval_metric': 'logloss', #'auc',  # 'logloss',
        'objective': 'binary:logistic',
        'silent': 1,
    }

    # if xgb_params['booster'] == 'dart':
    #    xgb_params['rate_drop'] = 0.1
    #    xgb_params['one_drop'] = 1

    if USE_GPU:
        xgb_params['updater'] = 'grow_gpu'

    return xgb_params

# ---------------------------------------------------------------------

# Папка с исходными данными задачи
data_folder = r'../input'

# ---------------------------------------------------------------------

def load_data( selected_cols ):
    train_data = pd.read_csv(os.path.join(data_folder, 'train.csv'), delimiter=';', skip_blank_lines=True)
    train_data = train_data.astype(float)
    cleanup.correct_datasets(train_data)

    test_data = pd.read_csv(os.path.join(data_folder, 'test.csv'), delimiter=';', skip_blank_lines=True,
                            na_values='None')

    test_data = test_data.astype(float)
    cleanup.correct_datasets(test_data)

    y_data = train_data['cardio'].values
    train_data = train_data.drop([ 'id', 'cardio'], axis=1)
    test_data = test_data.drop([ 'id'], axis=1)

    basic_cols = 'age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc'.split(';')

    scaler = MinMaxScaler()
    scaler.fit(train_data[basic_cols])

    all_data = [train_data, test_data]

    for data in all_data:
        normalized_data = scaler.transform(data[basic_cols])

        for i, colname in enumerate(basic_cols):
            data[colname] = normalized_data[:, i]

        if 'cholesterol+gluc' in selected_cols:
            data['cholesterol+gluc'] = data.cholesterol.values + data.gluc.values

        if 'ap_delta' in selected_cols:
            data['ap_delta'] = data.ap_hi.values - data.ap_lo.values

        if 'ap_hi-w' in selected_cols:
            data['ap_hi-w'] = data.ap_hi.values + 0.22*data.weight.values

        if 'ap_hi-h' in selected_cols:
            data['ap_hi-h'] = data.ap_hi.values - 0.042*data.height.values

        if 'ap_lo-h' in selected_cols:
            data['ap_lo-h'] = data.ap_lo.values - 0.033*data.height.values

        if 'ap_hi_pow2' in selected_cols:
            data['ap_hi_pow2'] = data.ap_hi.values*data.ap_hi.values

        if 'w/h' in selected_cols:
            data['w/h'] = data.weight.values / (0.01+data.height.values)

        if 'h/w' in selected_cols:
            data['h/w'] = data.height.values / (0.01+data.weight.values)

        if 'ap_hi/h' in selected_cols:
            data['ap_hi/h'] = data.ap_hi.values / (0.001+data.height.values)

        if 'ap_lo/h' in selected_cols:
            data['ap_lo/h'] = data.ap_lo.values / (0.001+data.height.values)

        if 'age_pow2' in selected_cols:
            data['age_pow2'] = data.age.values * data.age.values

        if 'age/w' in selected_cols:
            data['age/w'] = data.age.values / (0.01+data.weight.values)

        if 'age/ap_hi' in selected_cols:
            data['age/ap_hi'] = data.age.values / (1e-8+data.ap_hi.values)

        if 'cholesterol/w' in selected_cols:
            data['cholesterol/w'] = data.cholesterol.values / (0.01+data.weight.values)

    all_cols = train_data.columns.values
    x_data = train_data.drop([c for c in all_cols if c not in selected_cols], axis=1)
    x_submit = test_data.drop([c for c in all_cols if c not in selected_cols], axis=1)

    return (x_data, y_data, x_submit)


# -------------------------------------------------------------

if False:
    import hyperopt.pyll.stochastic

    space ={
            'eta': hp.loguniform('eta', -7, -2.3 ),
           }

    for _ in range(100):
        print( hyperopt.pyll.stochastic.sample(space) )

# ---------------------------------------------------------------------


if DATASET=='clean':
    x_data,y_data0,x_submit = cleanup.load_clean_data()
elif DATASET == 'extended':
    x_data, y_data0, x_submit = load_data('h/w;age;gender;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active'.split(';'))
else:
    x_data,y_data0,x_submit = cleanup.load_data()

X_data0 = x_data.values
X_submit = x_submit.values

print('X_data.shape={}'.format(X_data0.shape))

for colname in sorted(x_data.columns.values):
    train_vals = x_data[colname].values
    test_vals = x_submit[colname].values

    print('{}'.format(colname), end='')
    print('\ttrain ==> nans={} min={} max={} avg={}'.format( np.count_nonzero(np.isnan(train_vals)),
            np.nanmin(train_vals), np.nanmax(train_vals), np.nanmean(train_vals) ), end='')
    print('\t\ttest ==> nans={} min={} max={} avg={}'.format( np.count_nonzero(np.isnan(test_vals)),
            np.nanmin(test_vals), np.nanmax(test_vals), np.nanmean(test_vals) ))


X_data, X_holdout, y_data, y_holdout = train_test_split(X_data0, y_data0, test_size=HOLDOUT_SIZE, random_state=HOLDOUT_SEED )

D_data = xgboost.DMatrix(X_data0, y_data0, feature_names=x_data.columns.values)
D_train = xgboost.DMatrix(X_data, y_data, feature_names=x_data.columns.values)
D_holdout = xgboost.DMatrix(X_holdout, y_holdout, feature_names=x_data.columns.values)
D_submit = xgboost.DMatrix(X_submit, feature_names=x_data.columns.values)

# --------------------------------------------------------------

obj_call_count = 0
cur_best_loss = np.inf

log_writer = open( os.path.join(submission_folder, 'xgb-hyperopt-log.txt'), 'w' )

def objective(space):
    global obj_call_count, cur_best_loss, x_data, y_data, D_data, D_train, D_submit, D_holdout, y_holdout

    start = time.time()

    obj_call_count += 1

    print('\nXGB objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,cur_best_loss) )

    xgb_params = get_xgboost_params(space)

    sorted_params = sorted(space.iteritems(), key=lambda z: z[0])
    print('Params:', str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params]))

    if 'nbrounds' in space:
        # nbrounds входит в пространство гиперпараметров
        nbrounds = int(space['nbrounds'])
        cl = xgboost.train(xgb_params,
                           D_train,
                           evals=[(D_holdout,'holdout')],
                           num_boost_round=nbrounds,
                           verbose_eval=False,
                           early_stopping_rounds=None
                           )
    else:
        # nbrounds подбирается бустером по критерию early_stopping_rounds
        cvres = xgboost.cv(xgb_params,
                           D_train,
                           num_boost_round=10000,
                           nfold=5,
                           early_stopping_rounds=EARLY_STOPPING,
                           metrics=CV_METRICS,
                           seed=int(space['seed']),
                           # verbose_eval=50,
                           # verbose=True,
                           shuffle=True
                           )

        # cvres.to_csv(os.path.join(submission_folder, 'cvres.csv'))
        nbrounds = cvres.shape[0]
        cv_logloss = cvres['test-logloss-mean'].tolist()[-1]
        print('CV nbrounds={} cv_loss={:7.5f}'.format(nbrounds, cv_logloss))
        print('Train model...')
        cl = xgboost.train(xgb_params, D_train, num_boost_round=nbrounds, verbose_eval=False)

    y_pred = cl.predict(D_holdout, ntree_limit=nbrounds)
    holdout_logloss = calc_logloss( y_true=y_holdout, y_pred=y_pred )

    print('holdout_loss={:7.5f}'.format( holdout_logloss ) )

    do_submit = holdout_logloss<0.54165
    if holdout_logloss<cur_best_loss:
        cur_best_loss = holdout_logloss
        do_submit = True
        print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(cur_best_loss) + colorama.Fore.RESET)

    if do_submit:
        submit_guid = random_str()

        print('Compute submission and prediction guid={}'.format(submit_guid))

        y_submission = cl.predict(D_submit, ntree_limit=nbrounds)
        submission_filename = os.path.join(submission_folder,
                                           'xgboost_loss={:13.11f}_submission_guid={}.dat'.format(holdout_logloss, submit_guid))
        store_submission(y_submission, submission_filename)

        # сохраним веса фич для этой модели
        fmap = cl.get_score()
        with open(os.path.join(submission_folder, 'xgboost_loss={:13.11f}_feature_weights_guid={}.txt'.format(holdout_logloss, submit_guid)), 'w') as wrt:
            for f, w in sorted(fmap.iteritems(), key=lambda z: -z[1]):
                wrt.write('{}\t{}\n'.format(f, w))

        y_prediction = cl.predict(D_data, ntree_limit=nbrounds)
        prediction_filename = os.path.join(submission_folder,
                                           'xgboost_loss={:13.11f}_prediction_guid={}.dat'.format(holdout_logloss, submit_guid))
        store_submission(y_prediction, prediction_filename)

        log_writer.write( 'holdout_logloss={:<7.5f} Params:{} nbrounds={} submit_guid={}\n'.format( holdout_logloss, str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params]), nbrounds, submit_guid ) )
        log_writer.flush()


    end = time.time()
    elapsed = int(end - start)
    #print('elapsed={}'.format(elapsed ) )

    return{'loss':holdout_logloss, 'status': STATUS_OK }


# --------------------------------------------------------------------------------

space ={
        #'booster': hp.choice( 'booster',  ['dart', 'gbtree'] ),
        'max_depth': hp.quniform("max_depth", 4, MAX_DEPTH, 1),
        'min_child_weight': hp.quniform ('min_child_weight', 1, 20, 1),
        'subsample': hp.uniform ('subsample', 0.75, 1.0),
        #'gamma': hp.uniform('gamma', 0.0, 0.5),
        'gamma': hp.loguniform('gamma', -5.0, 0.0),
        #'eta': hp.uniform('eta', 0.005, 0.018),
        'eta': hp.loguniform('eta', -7, -2.3),
        #'nbrounds': hp.quniform ('nbrounds', 300, 1200, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.70, 1.0),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.70, 1.0),
        'seed': hp.randint('seed', 2000000)
       }

# --------------------------------------------------------------


trials = Trials()
best = hyperopt.fmin(fn=objective,
                     space=space,
                     algo=HYPEROPT_ALGO,
                     max_evals=N_HYPEROPT_PROBES,
                     trials=trials,
                     verbose=1)

log_writer.close()

print('-'*50)
print('The best params:')
print( best )
print('\n\n')


with open(os.path.join(submission_folder, 'xgboost-hyperopt.csv'),'w') as wrt:
    for i,loss in enumerate([ z['result']['loss'] for z in trials.trials ]):
        wrt.write('{}\t{}\n'.format(i,loss))
