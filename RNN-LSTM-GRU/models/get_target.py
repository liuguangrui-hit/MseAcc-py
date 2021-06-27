from data_process.my_dataset import Dataset
from tensorflow.keras.models import load_model
import numpy as np

import tensorflow as tf
import keras

config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)


def get_target_samples(model, X_test, Y_test, model_name):
    x_test_re = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    y_pred = model.predict(x_test_re)
    y_pred = np.array(y_pred)
    y_pred = [np.round(x) for x in y_pred]

    fail_samples = []
    success_samples = []
    for i in range(X_test.shape[0]):
        if Y_test[i] == 1 and y_pred[i] == 0:
            # print(X_test[i])
            fail_samples.append(X_test[i])
        elif Y_test[i] == 1 and y_pred[i] == 1:
            success_samples.append(X_test[i])
    print("success_samples size:", len(success_samples))
    success_samples = np.array(success_samples)
    np.save(f"data/time1_success_{model_name}.npy", success_samples)
    print("fail_samples size:", len(fail_samples))
    fail_samples = np.array(fail_samples)
    np.save(f"data/time1_fail_{model_name}.npy", fail_samples)


if __name__ == '__main__':
    test_s = Dataset("../data/cic_2017/data_sets/1.0_train_set.csv")
    x_test, y_test = test_s.items, test_s.label

    # model_name: mlp, dnn, conv, rnn, gru, lstm
    model_name = 'lstm'
    model = load_model('./model_record/time1_' + model_name + '_model.hdf5')
    get_target_samples(model, x_test, y_test, model_name)
