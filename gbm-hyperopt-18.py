# -*- coding: utf-8 -*-
'''
Решение задачи конкурса http://mlbootcamp.ru/round/12/tasks/ (c) Козиев Илья inkoziev@gmail.com
Поиск оптимальных значений для sklearn GradientBoostingClassifier с помощью гипероптимизации
библиотекой hyperopt.
Оптимизируется результат предсказания модели по отдельному holdout подмножеству
сэмплов, которые модель не видит при тренировке.
Дополнительно сохраняем результаты предсказания по обучающему набору, чтобы потом использовать
модель в блендере.
'''

from __future__ import print_function
import codecs
import random
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import cleanup
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
from sklearn.ensemble import GradientBoostingClassifier
from uuid import uuid4


N_HYPEROPT_PROBES = 5000
HOLDOUT_SEED = 123456
HOLDOUT_SIZE = 0.10
HYPEROPT_ALGO = tpe.suggest  #  tpe.suggest OR hyperopt.rand.suggest
DATASET = 'clean' # 'raw' | 'clean' | 'extended'

colorama.init()

# Папка для сохранения результатов
submission_folder = r'../submissions4'

# ---------------------------------------------------------------------

def store_submission( y_submission, submission_filename ):
    with codecs.open(submission_filename, 'w') as wrt:
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

# Папка с исходными данными задачи
if platform.system()=='Linux':
    data_folder = r'/home/eek/polygon/MLBootCamp/V/input'
else:
    data_folder = r'e:\MLBootCamp\V\input'


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


# ---------------------------------------------------------------------

if DATASET=='clean':
    x_data,y_data0,x_submit = cleanup.load_clean_data()
elif DATASET == 'extended':
    x_data, y_data0, x_submit = load_data('h/w;age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active'.split(';'))
else:
    x_data,y_data0,x_submit = cleanup.load_data()

X_data0 = x_data.values
X_submit = np.nan_to_num(x_submit.values)

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

# --------------------------------------------------------------

obj_call_count = 0
cur_best_loss = np.inf
log_writer = open( os.path.join(submission_folder, 'gbm-hyperopt-log.txt'), 'w' )

def objective(space):
    global obj_call_count, cur_best_loss, X_data0, X_data, X_holdout, y_data, y_holdout

    start = time.time()

    obj_call_count += 1

    print('\nGBM objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,cur_best_loss) )

    #print('\nDEBUG SPACE:\n', space, '\n' )

    sorted_params = sorted(space.iteritems(), key=lambda z: z[0])
    print('Params:', str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params if not k.startswith('column:')]))

    cl = GradientBoostingClassifier(loss='deviance',
                                    learning_rate=space['learning_rate'],
                                    n_estimators=int(space['n_estimators']),
                                    subsample=space['subsample'],
                                    criterion='friedman_mse',
                                    min_samples_split=int(space['min_samples_split']),
                                    min_samples_leaf=int(space['min_samples_leaf']),
                                    min_weight_fraction_leaf=0.0,
                                    max_depth=int(space['max_depth']),
                                    min_impurity_split=1e-07,
                                    init=None,
                                    random_state=int(space['seed']),
                                    max_features=None,
                                    verbose=0,
                                    max_leaf_nodes=None, warm_start=False, presort='auto')

    cl.fit(X=X_data,y=y_data)

    y_pred = cl.predict_proba(X_holdout)[:,1]
    holdout_logloss = calc_logloss( y_true=y_holdout, y_pred=y_pred )

    print('holdout_loss={:7.5f}'.format( holdout_logloss ) )

    do_submit = holdout_logloss<0.54150
    if holdout_logloss<cur_best_loss:
        cur_best_loss = holdout_logloss
        do_submit = True
        print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(cur_best_loss) + colorama.Fore.RESET)

    if do_submit:
        submit_guid = random_str()

        print('Compute submissions guid={}'.format(submit_guid))

        y_submission = cl.predict_proba(X_submit)[:,1]
        submission_filename = os.path.join(submission_folder,
                                           'gbm_loss={:13.11f}_submission_guid={}.dat'.format(holdout_logloss,submit_guid))
        store_submission(y_submission, submission_filename)

        y_prediction = cl.predict_proba(X_data0)[:,1]
        prediction_filename = os.path.join(submission_folder,
                                           'gbm_loss={:13.11f}_prediction_guid={}.dat'.format(holdout_logloss,submit_guid))
        store_submission(y_prediction, prediction_filename)

        log_writer.write( 'holdout_logloss={:<13.11f} Params:{} submit_guid={}\n'.format( holdout_logloss, str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params if not k.startswith('column:')]), submit_guid ) )
        log_writer.flush()

    end = time.time()
    elapsed = int(end - start)
    #print('elapsed={}'.format(elapsed ) )

    return{'loss':holdout_logloss, 'status': STATUS_OK }

# --------------------------------------------------------------------------------

space ={
        'max_depth': hp.quniform("max_depth", 4, 6, 1),
        'n_estimators': hp.quniform("n_estimators", 400, 800, 1),
        #'min_child_weight': hp.quniform ('min_child_weight', 0, 10, 1),
        'subsample': hp.uniform ('subsample', 0.75, 1.0),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
        #'learning_rate': hp.uniform('learning_rate', 0.01, 0.02),
        'learning_rate': hp.loguniform('learning_rate', -5.3, -3.5),
        'seed': hp.randint('seed', 2000000)
       }

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


with open( os.path.join(submission_folder, 'gbm-hyperopt.csv'),'w') as wrt:
    for i,loss in enumerate([ z['result']['loss'] for z in trials.trials ]):
        wrt.write('{}\t{}\n'.format(i,loss))
