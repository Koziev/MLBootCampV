# -*- coding: utf-8 -*-
'''
Решение задачи конкурса http://mlbootcamp.ru/round/12/tasks/ (c) Козиев Илья inkoziev@gmail.com

Поиск оптимальных значений для LightGBM через hyperopt.
Оптимизируется результат предсказания модели по отдельному holdout подмножеству
сэмплов, которые модель не видела при CV и тренировке.

hyperopt:
https://github.com/hyperopt/hyperopt/wiki/FMin
http://fastml.com/optimizing-hyperparams-with-hyperopt/
https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf

LightGBM:
http://lightgbm.readthedocs.io/en/latest/python/lightgbm.html#lightgbm-package
'''

from __future__ import print_function
import codecs
import math
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
import itertools
from sklearn.model_selection import train_test_split
import lightgbm
from sklearn.preprocessing import MinMaxScaler
from uuid import uuid4

N_HYPEROPT_PROBES = 5000
EARLY_STOPPING = 80
HOLDOUT_SEED = 123456
HOLDOUT_SIZE = 0.10
HYPEROPT_ALGO = tpe.suggest  #  tpe.suggest OR hyperopt.rand.suggest
DATASET = 'clean' # 'raw' | 'clean' | 'extended'
SEED0 = random.randint(1,1000000000)
NB_CV_FOLDS = 5

# Папка для сохранения результатов
submission_folder = r'../submissions5'

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

# Папка с исходными данными задачи
data_folder = r'../input'

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

        if 'cholesterol-gluc' in selected_cols:
            data['cholesterol-gluc'] = data.cholesterol.values - data.gluc.values

        if 'cholesterol*gluc' in selected_cols:
            data['cholesterol*gluc'] = data.cholesterol.values * data.gluc.values

        if 'ap_delta' in selected_cols:
            data['ap_delta'] = data.ap_hi.values - data.ap_lo.values

        if 'ap_delta/h' in selected_cols:
            data['ap_delta/h'] = (data.ap_hi.values - data.ap_lo.values)/(0.0001+data.height.values)

        if 'ap_hi/w' in selected_cols:
            data['ap_hi/w'] = data.ap_hi.values/(0.01+data.weight.values)

        if 'ap_hi*w' in selected_cols:
            data['ap_hi*w'] = data.ap_hi.values * data.weight.values

        if 'ap_hi*h' in selected_cols:
            data['ap_hi*h'] = data.ap_hi.values * data.height.values

        if 'ap_hi-w' in selected_cols:
            data['ap_hi-w'] = data.ap_hi.values + 0.22*data.weight.values

        if 'ap_hi-h' in selected_cols:
            data['ap_hi-h'] = data.ap_hi.values - 0.042*data.height.values

        if 'ap_hi+h+w' in selected_cols:
            data['ap_hi+h+w'] = data.ap_hi.values - 0.042*data.height.values + 0.22*data.weight.values

        if 'ap_lo-h' in selected_cols:
            data['ap_lo-h'] = data.ap_lo.values - 0.033*data.height.values

        if 'ap_lo+h+w' in selected_cols:
            data['ap_lo+h+w'] = data.ap_lo.values - 0.033*data.height.values + 0.11*data.weight.values

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
    x_data,y_data,x_submit = cleanup.load_clean_data()
elif DATASET == 'extended':
    # ~0.541509
    x_data,y_data,x_submit = load_data('h/w;age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active'.split(';'))
else:
    x_data,y_data,x_submit = cleanup.load_data()

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


# Сравнивать все модели будем через валидацию по отдельному подмножеству сэмплов, которые
# вообще не участвуют в тренировках и тестировании. Чтобы разные запуски программы давали
# сопоставимые результаты, задаем харжкодом random_state для разделения.
X_train, X_holdout, y_train, y_holdout = train_test_split(x_data, y_data, test_size=HOLDOUT_SIZE, random_state=HOLDOUT_SEED )

D_train = lightgbm.Dataset( X_train, label=y_train )
D_submit = lightgbm.Dataset( X_submit )

# --------------------------------------------------------------

def get_params(space):
    px = dict()

    px['boosting_type']='gbdt' # space['boosting_type'], # 'gbdt', # gbdt | dart | goss
    px['objective'] ='binary'
    px['metric'] = 'binary_logloss'
    px['learning_rate']=space['learning_rate']
    px['num_leaves'] = int(space['num_leaves'])
    px['min_data_in_leaf'] = int(space['min_data_in_leaf'])
    px['min_sum_hessian_in_leaf'] = space['min_sum_hessian_in_leaf']
    px['max_depth'] = int(space['max_depth']) if 'max_depth' in space else -1
    px['lambda_l1'] = 0.0  # space['lambda_l1'],
    px['lambda_l2'] = 0.0  # space['lambda_l2'],
    px['max_bin'] = 256
    px['feature_fraction'] = space['feature_fraction']
    px['bagging_fraction'] = space['bagging_fraction']
    px['bagging_freq'] = 5

    return px

# --------------------------------------------------------------

obj_call_count = 0
cur_best_loss = np.inf

def objective(space):
    global obj_call_count, cur_best_loss, X_train, y_train, X_submit, X_holdout, y_holdout

    start = time.time()

    obj_call_count += 1
    print('\nLightGBM objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,cur_best_loss) )

    sorted_params = sorted(space.iteritems(), key=lambda z: z[0])
    print('Params:', str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params if not k.startswith('column:')]))


    lgb_params = get_params(space)

    cvres = lightgbm.cv(lgb_params,
                        D_train,
                        num_boost_round=10000,
                        nfold=NB_CV_FOLDS,
                        #metrics='binary_logloss',
                        stratified=False,
                        shuffle=True,
                        #fobj=None,
                        #feval=None,
                        #init_model=None,
                        #feature_name='auto',
                        #categorical_feature='auto',
                        early_stopping_rounds=EARLY_STOPPING,
                        #fpreproc=None,
                        verbose_eval=False,
                        show_stdv=False,
                        seed=int(space['seed']),
                        #callbacks=None
                        )

    nbrounds = len(cvres['binary_logloss-mean'])
    cv_logloss = cvres['binary_logloss-mean'][-1]

    print( 'CV finished nbrounds={} loss={:7.5f}'.format( nbrounds, cv_logloss ) )

    cl = lightgbm.train(lgb_params,
                        D_train,
                        num_boost_round=nbrounds,
                        # metrics='mlogloss',
                        # valid_sets=None,
                        # valid_names=None,
                        # fobj=None,
                        # feval=None,
                        # init_model=None,
                        # feature_name='auto',
                        # categorical_feature='auto',
                        # early_stopping_rounds=None,
                        # evals_result=None,
                        verbose_eval=False,
                        # learning_rates=None,
                        # keep_training_booster=False,
                        # callbacks=None
                        )

    y_pred = cl.predict(X_holdout, num_iteration=nbrounds)
    logloss = calc_logloss(y_pred=y_pred, y_true=y_holdout)
    print('holdout logloss={}'.format(logloss))

    do_submit = logloss < 0.5415

    if logloss < cur_best_loss:
        cur_best_loss = logloss
        print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(cur_best_loss) + colorama.Fore.RESET)
        do_submit = True

    if do_submit:
        submit_guid = random_str()

        print('Compute submissions guid={}'.format(submit_guid))

        y_submission = cl.predict(X_submit, num_iteration=nbrounds)
        submission_filename = os.path.join(submission_folder, 'lightgbm_loss={:13.11f}_submission_guid={}.dat'.format(logloss,submit_guid))
        store_submission(y_submission, submission_filename)

        y_prediction = cl.predict(X_data0)
        prediction_filename = os.path.join(submission_folder, 'lightgbm_loss={:13.11f}_prediction_guid={}.dat'.format(logloss,submit_guid))
        store_submission(y_prediction, prediction_filename)

    return{'loss':logloss, 'status': STATUS_OK }

# --------------------------------------------------------------------------------

space ={
        #'boosting_type': hp.choice( 'boosting_type', ['gbdt', 'dart' ] ),
        #'max_depth': hp.quniform("max_depth", 4, 6, 1),
        'num_leaves': hp.quniform ('num_leaves', 20, 100, 1),
        'min_data_in_leaf':  hp.quniform ('min_data_in_leaf', 10, 100, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),
        'learning_rate': hp.loguniform('learning_rate', -6.9, -2.3),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
        #'lambda_l1': hp.uniform('lambda_l1', 1e-4, 1e-6 ),
        #'lambda_l2': hp.uniform('lambda_l2', 1e-4, 1e-6 ),
        'seed': hp.randint('seed',2000000)
       }


trials = Trials()
best = hyperopt.fmin(fn=objective,
                     space=space,
                     algo=HYPEROPT_ALGO,
                     max_evals=N_HYPEROPT_PROBES,
                     trials=trials,
                     verbose=1)

print('-'*50)
print('The best params:')
print( best )
print('\n\n')

with open( os.path.join(submission_folder, 'lgb-hyperopt.csv'), 'w') as wrt:
    for i,loss in enumerate([ z['result']['loss'] for z in trials.trials ]):
        wrt.write('{}\t{}\n'.format(i,loss))

