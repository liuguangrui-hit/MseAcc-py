import os

from data_process.my_dataset import Dataset_adv, Dataset, Dataset_mix,Dataset_adv_1
from tensorflow.keras.models import Sequential, load_model, Model
from models.save import save
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Dropout, SimpleRNN
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

import tensorflow as tf
import keras
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)

class my_RNN():
    def __init__(self, x_train):
        self.model = Sequential()
        self.model.add(SimpleRNN(120, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(SimpleRNN(120, return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(SimpleRNN(120, return_sequences=False))
        self.model.add(Dropout(0.2))

        # binary
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        # multiclass
        # model.add(Dense(5))
        # model.add(Activation('softmax'))

        self.model.summary()

        # optimizer
        adam = Adam(lr=0.0001)

        # binary
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

def compute_mse(epoch,model,x_train,y_train):
    m = 0
    count = 0
    x_list = []
    y_list = []
    time = 1
    mse_handle = open(mse_path, mode='a+')
    for x, y in zip(x_train, y_train):
        m += 1
        count += 1
        x_list.append(x)
        y_list.append(y)
        if m == 32 or count == x_train.shape[0]:
            x_list = np.array(x_list)
            y_list = np.array(y_list)
            y_p = model.predict(x_list)
            mse = mean_squared_error(y_true=y_list, y_pred=y_p)
            # print("mse: ", mse)
            mse_handle.write("Train Epoch: {}\t time: {}\t mse: {}\n".format(epoch + 1, time, mse))
            time += 1
            m = 0
            x_list = []
            y_list = []
    mse_handle.close()

epoch = 10
param_path = "param_data/RNN/"
mse_path ="rnn_mse.txt"

if __name__ == '__main__':

    # get and process data
    # data = DataProcess()
    # x_train, y_train, x_test, y_test, x_test_21, y_test_21 = data.return_processed_data_multiclass()
    # x_train, y_train, x_test, y_test = data.return_processed_cicids_data_binary()

    reuse_model = False
    is_train = True
    loop_exit = False
    while not loop_exit:
        print("Menu:")
        print("\t1: start NIDS training")
        print("\t2: continue NIDS training")
        print("\t3: get NIDS performances")
        c = input("Enter you choice: ")
        if c == '1':
            reuse_model = False
            is_train = True
            loop_exit = True
        if c == '2':
            reuse_model = True
            is_train = True
            loop_exit = True
        if c == '3':
            reuse_model = True
            is_train = False
            loop_exit = True

    train_s = Dataset_mix("../data/cic_2017/data_sets/1.0_train_set.csv",
                          "../data/cic_2017/adver_sets/time1_0.1_rnn_adver_train.csv",
                          p_start=0, p_len=14000, n_start=0, n_len=1000)
    x_train, y_train = train_s.items, train_s.label

    test_s = Dataset_adv_1("../data/cic_2017/adver_sets/time1_0.1_rnn_adver_test.csv")
    x_test, y_test = test_s.items, test_s.label
    # reshape input to be [samples, timesteps, features]
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    model_name = 'p_rnn'
    dataset_n_len = 20000
    dataset_a_len = 5
    char_list = ["/", ":"]
    # start training
    if not reuse_model and is_train:
        if not os.path.exists(param_path):
            os.mkdir(param_path)
        if os.path.exists(mse_path):  # ??????????????????
            # ????????????
            os.remove(mse_path)
        param_i = 1
        for i in range(epoch):
            print("epoch:", i + 1)
            if i == 0:
                train_s = Dataset("../data/cic_2017/data_sets/1.0_train_set.csv", start=i * dataset_n_len,
                                  len=dataset_n_len)
                x_train, y_train = train_s.items, train_s.label
                # print(x_train.shape)
                x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                # print(x_train.shape)
                model = my_RNN(x_train).model
                for k in range(5):
                    compute_mse(i,model,x_train,y_train)
                    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=32)
                    names = [weight.name for layer in model.layers for weight in layer.weights]
                    weights = model.get_weights()
                    for name, weight in zip(names, weights):
                        # print("name,shape:", name, weight.shape)
                        # print(weight)
                        for c in char_list:
                            if c in name:
                                name = name.replace(c, '_')
                        np.savetxt(param_path + 'time' + str(param_i) + '_' + name + '.csv', weight, delimiter=',')
                    param_i += 1
                save(model, 0, model_name)
            elif (i > 0 and i <= 4):
                train_s = Dataset("../data/cic_2017/data_sets/1.0_train_set.csv", start=i * dataset_n_len,
                                  len=dataset_n_len)
                x_train, y_train = train_s.items, train_s.label
                x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                model = load_model('./model_record/' + model_name + '_model.hdf5')

                for k in range(5):
                    compute_mse(i,model,x_train,y_train)
                    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=32)
                    names = [weight.name for layer in model.layers for weight in layer.weights]
                    weights = model.get_weights()
                    for name, weight in zip(names, weights):
                        # print("name,shape:", name, weight.shape)
                        # print(weight)
                        for c in char_list:
                            if c in name:
                                name = name.replace(c, '_')
                        np.savetxt(param_path + 'time' + str(param_i) + '_' + name + '.csv', weight, delimiter=',')
                    param_i += 1
                save(model, 0, model_name)
            else:
                train_s = Dataset_mix("../data/cic_2017/data_sets/1.0_train_set.csv",
                                      "../data/cic_2017/adver_sets/time1_0.1_rnn_adver_train.csv",
                                      p_start=(i) * dataset_n_len, p_len=dataset_n_len, n_start=(i) * dataset_a_len,
                                      n_len=dataset_a_len)
                x_train, y_train = train_s.items, train_s.label
                x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                model = load_model('./model_record/' + model_name + '_model.hdf5')

                for k in range(5):
                    compute_mse(i,model,x_train,y_train)
                    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=32)
                    names = [weight.name for layer in model.layers for weight in layer.weights]
                    weights = model.get_weights()
                    for name, weight in zip(names, weights):
                        # print("name,shape:", name, weight.shape)
                        # print(weight)
                        for c in char_list:
                            if c in name:
                                name = name.replace(c, '_')
                        np.savetxt(param_path + 'time' + str(param_i) + '_' + name + '.csv', weight, delimiter=',')
                    param_i += 1
                save(model, 0, model_name)

    elif reuse_model and is_train:
        # model_name: mlp, dnn, conv, rnn, gru, lstm

        i = 1
        model = load_model('./model_record/' + model_name + '_model.hdf5')
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=32, shuffle=False)
        save(model, 0, model_name)
        save(model, i + 1, model_name)

    else:
        model_name = 'p_rnn'
        i = 1
        model = load_model('./model_record/' + model_name + '_model.hdf5')
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=32, shuffle=False)
        save(model, 0, model_name)
        save(model, i + 1, model_name)
