# -*- coding: utf-8 -*-
'''
Черновой вариант решения задачи конкурса http://mlbootcamp.ru/round/12/tasks/
Классификатор - нейросетка на Keras.
Неочищенные данные.
При сабмите дает примерно 0.5447
(c) Козиев Илья inkoziev@gmail.com
'''

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
from keras.layers import Lambda
from keras import backend as K
import keras.utils

N_SPLIT = 7

# Папка с исходными данными задачи
data_folder = r'e:\MLBootCamp\V\input'
#data_folder = r'/home/eek/polygon/MLBootCamp/V/input'

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

# менять nan на нули не айс, надо на 2
X_data = np.nan_to_num(x_data.values)
X_submit = np.nan_to_num(x_test.values)

scaler = MinMaxScaler(copy=False)
scaler.fit(X_data)
X_data = scaler.transform(X_data)
X_submit = scaler.transform(X_submit)

n_feat = X_data.shape[1]
print( 'n_feat={}'.format(n_feat) )

DROPOUT_RATE = 0.10

inp = Input( shape=(n_feat,) )
seq = Dense(units=10, activation='relu')(inp)
seq = Dropout(DROPOUT_RATE)(seq)
seq = Dense(units=10, activation='relu')(inp)
seq = Dropout(DROPOUT_RATE)(seq)
seq = Dense(units=1, activation='sigmoid')(seq)
model = Model( inputs=inp, outputs=seq )

model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

model_checkpoint = ModelCheckpoint('nn.model', monitor='val_loss',
                                   verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

history = model.fit(x=X_data, y=y_data, validation_split=0.1,
                    batch_size=1024, epochs=1000, callbacks=[model_checkpoint, early_stopping])

model.load_weights('nn.model')
y_submission = model.predict(X_submit)[:, 0]
submission_filename = os.path.join(submission_folder, 'submission_nn.dat')
store_submission(y_submission, submission_filename)
print('Submission data have been stored in {}\n'.format(submission_filename))
