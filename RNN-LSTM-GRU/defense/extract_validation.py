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


def get_test(model, X_test, Y_test,is_success):
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
        if Y_test[i] == 1 and y_pred[i] == 0:
            # print(x[j])
            fail_samples.append(X_test[i].tolist())
        if Y_test[i] == 1 and y_pred[i] == 1:
            correct += 1
            # print(x[j])
            success_samples.append(X_test[i].tolist())
        if Y_test[i] == 0 and y_pred[i] == 0:
            correct += 1
    cnt = X_test.shape[0]
    print('Test set: Accuracy: {}/{} ({:.6f}%)\n'.format(correct, cnt, 100. * correct / cnt))
    if is_success:
        return success_samples
    else:
        return fail_samples


val_test_set_a = []
val_test_set_b = []
val_test_set = []
val_adv_set_a = []
val_adv_set_b = []
val_adv_set = []

model_name = '25_gru'
model_p = 'saved_models/GRU/' + model_name + '_model.hdf5'


if os.path.exists(model_p):
    model = load_model(model_p)

    test_s = Dataset("../data/cic_2017/data_sets/1.0_test_set.csv",start=0,len=20000)
    x_test, y_test = test_s.items, test_s.label
    val_test_set_a = get_test(model, x_test, y_test, True)

print(val_test_set_a[0])

model_name = '26_gru'
model_p = 'saved_models/GRU/' + model_name + '_model.hdf5'

if os.path.exists(model_p):
    model = load_model(model_p)

    test_s = Dataset("../data/cic_2017/data_sets/1.0_test_set.csv",start=0,len=20000)
    x_test, y_test = test_s.items, test_s.label
    val_test_set_b = get_test(model, x_test, y_test, True)

for a in val_test_set_a:
    if a in val_test_set_b:
        val_test_set.append(a)


model_name = '25_gru'
model_p = 'saved_models/GRU/' + model_name + '_model.hdf5'

if os.path.exists(model_p):
    model = load_model(model_p)

    test_s = Dataset_adv_1("../data/cic_2017/adver_sets/time1_0.1_gru_adver_test.csv",start=0,len=20000)
    x_test, y_test = test_s.items, test_s.label
    val_adv_set_a = get_test(model, x_test, y_test, True)

model_name = '26_gru'
model_p = 'saved_models/GRU/' + model_name + '_model.hdf5'

if os.path.exists(model_p):
    model = load_model(model_p)

    test_s = Dataset_adv_1("../data/cic_2017/adver_sets/time1_0.1_gru_adver_test.csv",start=0,len=20000)
    x_test, y_test = test_s.items, test_s.label
    val_adv_set_b = get_test(model, x_test, y_test, False)

for a in val_adv_set_a:
    if a in val_adv_set_b:
        val_adv_set.append(a)

print(len(val_test_set))
print(val_test_set[0])
print(len(val_adv_set))
print(val_adv_set[0])

val_set = []
val_set.extend(val_adv_set)
val_set.extend(val_test_set)

items = np.array(val_set, dtype=np.float32)
test=pd.DataFrame(data=items)
test.to_csv('val_data/val_set_gru.csv', sep=',', header=None, index=False, mode='w', line_terminator='\n', encoding='utf-8')