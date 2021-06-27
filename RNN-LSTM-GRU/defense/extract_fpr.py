from tensorflow.keras.models import Sequential, load_model, Model
import os
import numpy as np
import pandas as pd
from data_process.my_dataset import Dataset_adv, Dataset, Dataset_adv_1, Dataset_mix

import tensorflow as tf
import keras
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)


def get_test(model, X_test, Y_test):
    success_samples = []
    fail_samples = []
    correct = 0
    x_test_re = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    y_pred = model.predict(x_test_re)
    y_pred = np.array(y_pred)
    # print(y_pred)
    y_pred = [np.round(x) for x in y_pred]
    # print(y_pred)
    for i in range(X_test.shape[0]):
        if Y_test[i] == 0 and y_pred[i] == 0:
            # print(x[j])
            success_samples.append(X_test[i].tolist())
        if Y_test[i] == 0 and y_pred[i] == 1:
            # print(x[j])
            fail_samples.append(X_test[i].tolist())

    correct = len(success_samples)
    cnt = len(success_samples) + len(fail_samples)
    print('fpr -> {}'.format(correct / cnt))
    return correct / cnt


test_s_1 = Dataset("../data/cic_2017/data_sets/1.0_train_set.csv")
x_test_1, y_test_1 = test_s_1.items, test_s_1.label

test_s = Dataset("../data/cic_2017/data_sets/1.0_test_set.csv")
x_test, y_test = test_s.items, test_s.label
train_fpr = []
test_fpr = []
for i in range(1,51):
    print(i)
    model_name = '{}_gru_1'.format(i)
    model_p = 'saved_models/gru_1/' + model_name + '_model.hdf5'
    if os.path.exists(model_p):
        model = load_model(model_p)

        tr = get_test(model, x_test_1, y_test_1)
        tr = 1-tr
        train_fpr.append(tr)
        te = get_test(model, x_test, y_test)
        te=1-te
        test_fpr.append(te)

print(train_fpr)
print(test_fpr)