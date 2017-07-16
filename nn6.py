# -*- coding: utf-8 -*-
'''
Решение задачи конкурса http://mlbootcamp.ru/round/12/tasks/ (c) Козиев Илья inkoziev@gmail.com
Random search: генерируем нейросетки, отбираем лучшие варианты на holdout-наборе.
'''

from __future__ import print_function
import os
import pandas as pd
import numpy as np
import codecs
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.optimizers
import platform
import cleanup
import math
import random
from sklearn.model_selection import train_test_split
from uuid import uuid4



NB_MODELS = 10000  # кол-во генерируемых вариантов сетки
HOLDOUT_SEED = 123456
HOLDOUT_SIZE = 0.10

# Папка с исходными данными задачи
data_folder = r'../input'

# Папка для сохранения результатов
submission_folder = r'../submissions6'

# ---------------------------------------------------------------------

def calc_logloss(y_pred, y_true, epsilon=1.0e-6):
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    y_true = np.clip(y_true, epsilon, 1.0 - epsilon)
    return - np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))


# ---------------------------------------------------------------------

def store_submission( y_submission, submission_filename ):
    with codecs.open(submission_filename, 'w') as wrt:
        for idata,y in enumerate(y_submission):
            wrt.write('{}\n'.format(y))

# ---------------------------------------------------------------------

def random_str():
    return str(uuid4())


# ---------------------------------------------------------------------

# Папка с исходными данными задачи
data_folder = r'../input'

def load_data(selected_cols):
    train_data = pd.read_csv(os.path.join(data_folder, 'train.csv'), delimiter=';', skip_blank_lines=True)
    train_data = train_data.astype(float)
    cleanup.correct_datasets(train_data)

    test_data = pd.read_csv(os.path.join(data_folder, 'test.csv'), delimiter=';', skip_blank_lines=True,
                            na_values='None')

    test_data = test_data.astype(float)
    cleanup.correct_datasets(test_data)

    y_data = train_data['cardio'].values
    train_data = train_data.drop(['id', 'cardio'], axis=1)
    test_data = test_data.drop(['id'], axis=1)

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
            data['ap_hi-w'] = data.ap_hi.values + 0.22 * data.weight.values

        if 'ap_hi-h' in selected_cols:
            data['ap_hi-h'] = data.ap_hi.values - 0.042 * data.height.values

        if 'ap_lo-h' in selected_cols:
            data['ap_lo-h'] = data.ap_lo.values - 0.033 * data.height.values

        if 'ap_hi_pow2' in selected_cols:
            data['ap_hi_pow2'] = data.ap_hi.values * data.ap_hi.values

        if 'w/h' in selected_cols:
            data['w/h'] = data.weight.values / (0.01 + data.height.values)

        if 'h/w' in selected_cols:
            data['h/w'] = data.height.values / (0.01 + data.weight.values)

        if 'ap_hi/h' in selected_cols:
            data['ap_hi/h'] = data.ap_hi.values / (0.001 + data.height.values)

        if 'ap_lo/h' in selected_cols:
            data['ap_lo/h'] = data.ap_lo.values / (0.001 + data.height.values)

        if 'age_pow2' in selected_cols:
            data['age_pow2'] = data.age.values * data.age.values

        if 'age/w' in selected_cols:
            data['age/w'] = data.age.values / (0.01 + data.weight.values)

        if 'age/ap_hi' in selected_cols:
            data['age/ap_hi'] = data.age.values / (1e-8 + data.ap_hi.values)

        if 'cholesterol/w' in selected_cols:
            data['cholesterol/w'] = data.cholesterol.values / (0.01 + data.weight.values)

    all_cols = train_data.columns.values
    x_data = train_data.drop([c for c in all_cols if c not in selected_cols], axis=1)
    x_submit = test_data.drop([c for c in all_cols if c not in selected_cols], axis=1)

    return (x_data, y_data, x_submit)


# ---------------------------------------------------------------------

x_data_1,y_data0_1,x_test_1 = load_data('ap_hi;ap_lo;age;gender;height;weight;cholesterol;gluc;smoke;alco;active'.split(';'))
x_data_2,y_data0_2,x_test_2 = load_data('h/w;ap_hi;ap_lo;age;gender;height;weight;cholesterol;gluc;smoke;alco;active'.split(';'))

# --------------------------------------------------------------

for iprobe in range(NB_MODELS):

    if random.choice([0,1])==0:
        x_data, y_data0, x_test = x_data_1,y_data0_1,x_test_1
        dataset = 1
    else:
        x_data, y_data0, x_test = x_data_2,y_data0_2,x_test_2
        dataset = 2

    # менять nan на нули не айс, надо на 2 или типа того
    # или рандомно ставить 0/1 с учетом частоты данных величин в тренировочном наборе.
    X_data0 = x_data.values
    X_submit = np.nan_to_num(x_test.values)

    if dataset==2:
        scaler = MinMaxScaler()
        scaler.fit(X_data0)
        X_data0 = scaler.transform(X_data0)
        X_submit = scaler.transform(X_submit)


    X_data, X_holdout, y_data, y_holdout = train_test_split(X_data0, y_data0, test_size=HOLDOUT_SIZE,
                                                            random_state=HOLDOUT_SEED)

    DROPOUT_RATE = random.uniform(0.00,0.10) if (iprobe%2)==0 else 0.0
    nlayers = random.choice([1,2,3,4,5])

    print('{}/{} dataset={} dropout={:<7.5f} nlayers={}'.format(iprobe+1, NB_MODELS, dataset, DROPOUT_RATE, nlayers), end='')

    n_feat = X_data.shape[1]
    inp = Input( shape=(n_feat,) )
    seq = Dense(units=n_feat, activation='relu')(inp)

    if DROPOUT_RATE>0:
        seq = Dropout(DROPOUT_RATE)(seq)

    #seq = BatchNormalization()(seq)

    if nlayers>=1:
        seq = Dense(units=11, activation='relu')(inp)

        if DROPOUT_RATE>0:
            seq = Dropout(DROPOUT_RATE)(seq)

        if nlayers >= 2:
            seq = Dense(units=10, activation='relu')(inp)

            if DROPOUT_RATE > 0:
                seq = Dropout(DROPOUT_RATE)(seq)

            if nlayers >= 3:
                seq = Dense(units=9, activation='relu')(inp)

                if DROPOUT_RATE > 0:
                    seq = Dropout(DROPOUT_RATE)(seq)

                if nlayers >= 4:
                    seq = Dense(units=8, activation='relu')(inp)

                    if DROPOUT_RATE > 0:
                        seq = Dropout(DROPOUT_RATE)(seq)

    seq = Dense(units=1, activation='sigmoid')(seq)
    model = Model( inputs=inp, outputs=seq )

    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint('nn.model', monitor='val_loss',
                                       verbose=0, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')

    history = model.fit(x=X_data, y=y_data, validation_split=0.1,
                        batch_size=2048, epochs=1000,
                        callbacks=[model_checkpoint, early_stopping],
                        verbose=0 )

    best_loss = model_checkpoint.best

    model.load_weights('nn.model')

    y_pred = model.predict(X_holdout)[:, 0]
    holdout_logloss = calc_logloss( y_true=y_holdout, y_pred=y_pred )
    print(' stopped_epoch={} val_loss={:<7.5f} holdout_loss={:<7.5f}'.format( early_stopping.stopped_epoch, best_loss, holdout_logloss) )

    if holdout_logloss<0.5460:
        y_submission = model.predict(X_submit)[:, 0]
        submission_filename = os.path.join(submission_folder,'nn_single_loss={:<13.11f}.dat'.format(holdout_logloss))
        store_submission(y_submission, submission_filename)
