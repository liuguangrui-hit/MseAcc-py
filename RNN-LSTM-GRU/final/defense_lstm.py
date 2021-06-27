import datetime
import os
import shutil
import sys
import traceback

from data_process.my_dataset import Dataset_adv, Dataset, Dataset_mix,Dataset_adv_1
from tensorflow.keras.models import Sequential, load_model, Model
from defense.save import save
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Dropout, SimpleRNN, LSTM
import numpy as np
from sklearn.metrics import mean_squared_error
from final.save_model import save_model

import tensorflow as tf
import keras
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)

class my_LSTM():
    def __init__(self,x_train):
        self.model = Sequential()
        self.model.add(LSTM(120, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(120, return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(120, return_sequences=False))
        self.model.add(Dropout(0.2))

        # binary
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.summary()

        # optimizer
        adam = Adam(lr=0.0001)

        # binary
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])


def compute_mse(epoch_, model, x_train, y_train):
    m = 0
    count = 0
    x_list = []
    y_list = []

    mse_handle = open(mse_path, mode='a+')
    mse_list = []
    time = 1
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
            m = 0
            x_list = []
            y_list = []
            mse_list.append(mse)
            mse_handle.write("Train Epoch: {}\t time: {}\t mse: {}\n".format(epoch_ + 1, time, mse))
            time += 1

    mse_handle.close()
    return max(mse_list)


def get_test(model, X_test, Y_test):
    correct = 0
    acc = 0
    x_test_re = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    y_pred = model.predict(x_test_re)
    y_pred = np.array(y_pred)
    y_pred = [np.round(x) for x in y_pred]

    for i in range(X_test.shape[0]):
        if Y_test[i] == 1 and y_pred[i] == 1:
            correct += 1
        if Y_test[i] == 0 and y_pred[i] == 0:
            correct += 1
    cnt = X_test.shape[0]
    acc = correct / cnt
    print('Test set: Accuracy: {}/{} ({:.6f}%)\n'.format(correct, cnt, 100. * correct / cnt))
    return acc


def roll(epoch_, x_val, y_val):
    cur = epoch_
    try:
        path = 'saved_models/LSTM/'
        # delete current epoch
        for file in os.listdir(path):
            if os.path.splitext(file)[1] == '.hdf5':
                path_now = os.path.join(path, file)
                if os.path.isfile(path_now) and os.path.splitext(file)[0].split('_')[0] == '{}'.format(cur):
                    # # 删除查找到的文件
                    # os.remove(path_now)
                    # 重命名文件
                    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  # 现在
                    path_new = os.path.join(path, 'delete_' + str(nowTime) + '_' + str(file))
                    os.rename(path_now, path_new)
        # roll back
        val_min_list = []
        for i in range(5):
            model_p = "saved_models/LSTM/{}_{}_NIDS_LSTM.hdf5".format(cur - 1, i)
            if os.path.exists(model_p):
                model = load_model(model_p)
                val_min = get_test(model, x_val, y_val)
                val_min_list.append(val_min)
        print(val_min_list)
        if min(val_min_list) > 0.92:
            # update current model
            save_model(model, -1, -1, 'LSTM')
            print('Roll back succeed! Current epoch : {}'.format(cur))
            return cur
            # return start_epoch
        else:
            print('Roll back failed! Continuing...')
            if cur - 1 < 3:
                print('Model train failed! Restart a train please!')
                sys.exit()
            else:
                cur = roll(cur - 1, x_val, y_val)
                return cur

    except Exception as e:
        print(traceback.format_exc())
        print(e)


# hyper-parameter
epoch = 10
val_path = 'saved_models/val_LSTM.txt'
model_path = 'saved_models/LSTM/'
mse_path = 'saved_models/mse_LSTM.txt'
log_path = 'saved_models/log_LSTM.txt'
mse_max_value1, mse_max_value2 = 0.0, 0.3
val_min_value1, val_min_value2 = 0.90, 1.0

if __name__ == '__main__':
    reuse_model = False
    is_train = True
    loop_exit = False
    while not loop_exit:
        print("----------- Welcome to NIDS Poison Detector! -----------")
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

    # train_s = Dataset_mix("../data/cic_2017/data_sets/1.0_train_set.csv",
    #                       "../data/cic_2017/adver_sets/time1_0.1_LSTM_adver_train.csv",
    #                       p_start=0, p_len=14000, n_start=0, n_len=1000)
    # x_train, y_train = train_s.items, train_s.label

    test_s = Dataset("../data/cic_2017/data_sets/1.0_crossval_set.csv")
    x_test, y_test = test_s.items, test_s.label

    # reshape input to be [samples, timesteps, features]
    # x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    val_s = Dataset_adv_1('val_data/val_set_LSTM.csv')
    x_val, y_val = val_s.items, val_s.label

    model_name = 'LSTM'
    dataset_n_len = 20000
    dataset_a_len = 20
    mse_max_list = []
    acc_min_list = []
    val_min_lsit = []

    # start training
    if not reuse_model and is_train:
        # 清空所有model记录
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            os.mkdir(model_path)
        if os.path.exists(val_path):  # 如果文件存在
            # 删除文件
            os.remove(val_path)
        if os.path.exists(mse_path):  # 如果文件存在
            # 删除文件
            os.remove(mse_path)
        if os.path.exists(log_path):  # 如果文件存在
            # 删除文件
            os.remove(log_path)

        i = 0
        poison = True
        while i < epoch:
            print('----------- epoch: %d -----------' % (i + 1))
            if i == 0:
                train_s = Dataset("../data/cic_2017/data_sets/1.0_train_set.csv", start=i * dataset_n_len,
                                  len=dataset_n_len)
                x_train, y_train = train_s.items, train_s.label
                x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                model = my_LSTM(x_train).model
                for k in range(5):
                    mse_max = compute_mse(i, model, x_train, y_train)
                    mse_max_list.append(mse_max)
                    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=32)
                    val_min = get_test(model, x_val, y_val)
                    val_min_lsit.append(val_min)
                    save_model(model, i, k, model_name)
                    save_model(model, -1, -1, model_name)
            elif i == 5 and poison is True:
                train_s = Dataset_mix("../data/cic_2017/data_sets/1.0_train_set.csv",
                                      "../data/cic_2017/adver_sets/time1_0.1_LSTM_adver_train.csv",
                                      p_start=(i) * dataset_n_len, p_len=dataset_n_len, n_start=(i) * dataset_a_len,
                                      n_len=dataset_a_len)
                x_train, y_train = train_s.items, train_s.label
                x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                model = load_model('saved_models/NIDS_' + model_name + '.hdf5')

                for k in range(5):
                    mse_max = compute_mse(i, model, x_train, y_train)
                    mse_max_list.append(mse_max)
                    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=32)
                    val_min = get_test(model, x_val, y_val)
                    val_min_lsit.append(val_min)
                    save_model(model, i, k, model_name)
                    save_model(model, -1, -1, model_name)
            else:
                train_s = Dataset("../data/cic_2017/data_sets/1.0_train_set.csv", start=i * dataset_n_len,
                                  len=dataset_n_len)
                x_train, y_train = train_s.items, train_s.label
                x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                model = load_model('saved_models/NIDS_' + model_name + ".hdf5")

                for k in range(5):
                    mse_max = compute_mse(i, model, x_train, y_train)
                    mse_max_list.append(mse_max)
                    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=32)
                    val_min = get_test(model, x_val, y_val)
                    val_min_lsit.append(val_min)
                    save_model(model, i, k, model_name)
                    save_model(model, -1, -1, model_name)

            print('max / min :', max(mse_max_list), min(val_min_lsit))
            if (i > 2) and ((max(mse_max_list) > mse_max_value2) or (min(val_min_lsit) < val_min_value1)):
                print('\033[1;31;40m')
                print('*' * 50)
                print('NIDS system exception!!! ')
                print('mse_max value range --> ({}, {})  val_min set value --> ({}, {})'
                      .format(mse_max_value1, mse_max_value2, val_min_value1, val_min_value2))
                print('*' * 50)
                print('\033[0m')
                c = input('Roll Back! or Exit or Continue? (r/e/c)')
                if c == 'e' or c == 'E':
                    print('NIDS system training exit, try to retrain!')
                    sys.exit()
                elif c == 'r' or c == 'R':
                    i = roll(i, x_val, y_val)
                    mse_max_list = []
                    acc_min_list = []
                    val_min_lsit = []
                    poison = False
                elif c == 'c' or c == 'C':
                    mse_max_list = []
                    acc_min_list = []
                    val_min_lsit = []
                    i += 1
            else:
                mse_max_list = []
                acc_min_list = []
                val_min_lsit = []
                i += 1
        print('----------- Model training has been completed! -----------\n\n')


    elif reuse_model and is_train:

        poison = True
        start_epoch = int(input("\nInput your last train epoch:"))
        epoch = int(input("\nInput train epoch this time:"))
        model_p = 'saved_models/NIDS_' + model_name + ".hdf5"
        if os.path.exists(model_p):
            i = start_epoch
            while i < start_epoch + epoch:  # 训epoch个epoch
                print('----------- epoch: %d -----------' % (i + 1))
                if i == (start_epoch + 1) and poison == True:
                    train_s = Dataset_mix("../data/cic_2017/data_sets/1.0_train_set.csv",
                                          "../data/cic_2017/adver_sets/time1_0.1_LSTM_adver_train.csv",
                                          p_start=(i) * dataset_n_len, p_len=dataset_n_len, n_start=(i) * dataset_a_len,
                                          n_len=dataset_a_len)
                    x_train, y_train = train_s.items, train_s.label
                    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                    model = load_model(model_p)
                else:
                    train_s = Dataset("../data/cic_2017/data_sets/1.0_train_set.csv", start=i * dataset_n_len,
                                      len=dataset_n_len)
                    x_train, y_train = train_s.items, train_s.label
                    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                    model = load_model(model_p)
                for k in range(5):
                    mse_max = compute_mse(i, model, x_train, y_train)
                    mse_max_list.append(mse_max)
                    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=32)
                    val_min = get_test(model, x_val, y_val)
                    val_min_lsit.append(val_min)
                    save_model(model, i, k, model_name)
                    save_model(model, -1, -1, model_name)

                if (i > 2) and ((max(mse_max_list) > mse_max_value2) or (min(val_min_lsit) < val_min_value1)):
                    print('\033[1;31;40m')
                    print('*' * 50)
                    print('NIDS system exception!!! ')
                    print('mse_max value range --> ({}, {})  val_min set value --> ({}, {})'
                          .format(mse_max_value1, mse_max_value2, val_min_value1, val_min_value2))
                    print('*' * 50)
                    print('\033[0m')
                    c = input('Roll Back! or Exit or Continue? (r/e/c)')
                    if c == 'e' or c == 'E':
                        print('NIDS system training exit, try to retrain!')
                        sys.exit()
                    elif c == 'r' or c == 'R':
                        i = roll(i, x_val, y_val)
                        mse_max_list = []
                        acc_min_list = []
                        val_min_lsit = []
                        poison = False
                    elif c == 'c' or c == 'C':
                        mse_max_list = []
                        acc_min_list = []
                        val_min_lsit = []
                        i += 1
                else:
                    mse_max_list = []
                    acc_min_list = []
                    val_min_lsit = []
                    i += 1

            print('----------- Model retraining has been completed! -----------\n\n')
        else:
            start_epoch = 0
            print('No saved model, try start NIDS training！')

    # test
    elif reuse_model and not is_train:
        loop_exit = False
        while not loop_exit:
            print("\n\t1: using test set")
            print("\t2: using validation set")
            print("\t3: using adv set")
            c = input("Enter you choice: ")
            if c == '1':
                # test acc
                # using init test set
                test_s = Dataset("../data/cic_2017/data_sets/1.0_test_set.csv")
                x_test, y_test = test_s.items, test_s.label
                loop_exit = True
            if c == '2':
                # test val
                test_s = Dataset_adv_1('val_data/val_set_LSTM.csv')
                x_test, y_test = test_s.items, test_s.label
                loop_exit = True
            if c == '3':
                # test bypass
                test_s = Dataset_adv_1("../data/cic_2017/adver_sets/time1_0.1_LSTM_adver_example.csv")
                x_test, y_test = test_s.items, test_s.label
                loop_exit = True

        model_p = 'saved_models/NIDS_' + model_name + ".hdf5"
        if os.path.exists(model_p):
            model = load_model(model_p)
            acc_min = get_test(model, x_test, y_test)
        else:
            print('No saved model, try start NIDS training！')
