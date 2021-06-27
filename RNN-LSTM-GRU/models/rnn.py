import os
from datetime import datetime
from data_process.my_dataset import Dataset_adv, Dataset
from data_process.DataProcess import DataProcess
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Dropout, SimpleRNN
import numpy as np
from models.save import save
import time
from sklearn.metrics import classification_report, confusion_matrix
import winsound

# get and process data
# data = DataProcess()
# x_train, y_train, x_test, y_test = data.return_processed_cicids_data_binary()
i = 0
epoch = 1
train_s = Dataset("../data/cic_2017/data_sets/1.0_train_set.csv")
# train_s = Dataset("../data/cic_2017/data_sets/1.0_train_set.csv")
x_train, y_train = train_s.items, train_s.label

test_s = Dataset("../data/cic_2017/data_sets/1.0_test_set.csv")
x_test, y_test = test_s.items, test_s.label

# reshape input to be [samples, timesteps, features]
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

model = Sequential()
model.add(SimpleRNN(120, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

model.add(SimpleRNN(120, return_sequences=True))
model.add(Dropout(0.2))

model.add(SimpleRNN(120, return_sequences=False))
model.add(Dropout(0.2))

# binary
model.add(Dense(1))
model.add(Activation('sigmoid'))

# multiclass
# model.add(Dense(5))
# model.add(Activation('softmax'))

model.summary()

# optimizer
adam = Adam(lr=0.0001)

# binary
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

# multiclass
# model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics=['accuracy'])

start = time.time()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, batch_size=32)

# save the model
save(model, 0, 'time1_rnn')
# save(model, 0.1, 'epoch1_rnn')

loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

print("--- %s seconds ---" % (time.time() - start))
y_pred = model.predict(x_test)
y_pred = np.array(y_pred)
y_pred = [np.round(x) for x in y_pred]

print("\nAnomaly in Test: ", np.count_nonzero(y_test, axis=0))
print("Anomaly in Prediction: ", np.count_nonzero(y_pred, axis=0))


print('\nConfusion Matrix')
# binary
print(confusion_matrix(y_test, y_pred))

print('\nClassification Report')
print(classification_report(y_test, y_pred))
winsound.PlaySound("SystemHand", winsound.SND_ALIAS)