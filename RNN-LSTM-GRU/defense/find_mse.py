from tensorflow.keras.models import Sequential, load_model, Model
import os
import numpy as np
import pandas as pd
from data_process.my_dataset import Dataset_adv, Dataset, Dataset_adv_1, Dataset_mix
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)


def compute_mse(model, x_train, y_train):
    x_list = np.array(x_train)
    y_list = np.array(y_train)
    y_p = model.predict(x_list)

    mse = mean_squared_error(y_true=y_list, y_pred=y_p)
    print("mse: ", mse)
    return mse


model_name = '25_lstm'
model_p = 'saved_models/LSTM/' + model_name + '_model.hdf5'
length = 10
mse_list = []
if os.path.exists(model_p):
    model = load_model(model_p)

    for i in range(50):
        print(i)
        test_s = Dataset("../data/cic_2017/data_sets/1.0_train_set.csv", start=i * length,
                         len=length)
        x_test, y_test = test_s.items, test_s.label
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
        mse = compute_mse(model, x_test, y_test)
        mse_list.append(mse)

    for i in range(50, 60):
        print(i)
        test_s = Dataset_adv("../data/cic_2017/adver_sets/time1_0.1_lstm_adver_train.csv", start=i * length,
                              len=length)
        x_test, y_test = test_s.items, test_s.label
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
        mse = compute_mse(model, x_test, y_test)
        mse_list.append(mse)

# 确定子图数量
plt.subplots(1, 1)
x = range(60)

y_GRU = mse_list

plt.scatter(x, y_GRU, color='white', marker='o', edgecolors='black', s=30)

# 横线
plt.hlines(0.3, -1, 61, color="black", linestyles='-')

# set range_y
plt.ylim((-0.1, 1.1))

# 设置坐标轴刻度
my_x_ticks = np.arange(0, 61, 10)
my_y_ticks = np.arange(-0.1, 1.1, 0.1)

plt.xticks(my_x_ticks, fontproperties='Times New Roman', size=14)
plt.yticks(my_y_ticks, fontproperties='Times New Roman', size=14)

# set x_label, y_label, title
plt.xlabel("Index", fontdict={'family': 'Times New Roman', 'size': 16})
plt.ylabel("Robust distances", fontdict={'family': 'Times New Roman', 'size': 16})
plt.title("MSE", fontdict={'family': 'Times New Roman', 'size': 16})
plt.tight_layout()
plt.show()
