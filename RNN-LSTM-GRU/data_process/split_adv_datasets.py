import pandas as pd
import numpy as np

input_path = "../data/cic_2017/adver_sets/"
sample_size = 0.1
model_name = 'rnn'
filename = "time1_0.1_"+model_name+"_adver_example.csv"

ratio_train = 0.2 #训练集比例
ratio_val = 0.8 #测试集比例

file = input_path + filename
items = []

train_file = "../data/cic_2017/adver_sets/time1_0.1_"+model_name+"_adver_train.csv"
test_file = "../data/cic_2017/adver_sets/time1_0.1_"+model_name+"_adver_test.csv"

with open(file, "r") as f:
    lines = f.readlines()

    for line in lines:
        try:
            items.append([v for v in line.strip("\n").split(",")])
        except:
            continue
    # items = np.array(items)
    np.random.shuffle(items)  ##打乱文件列表
    cnt_test = round(len(items) * ratio_val, 0)
    cnt_train = len(items) - cnt_test
    train_list = []
    test_list = []
    for i in range(int(cnt_train)):
        train_list.append(items[i])

    for i in range(int(cnt_train), len(items)):
        test_list.append(items[i])
    train_df = pd.DataFrame(data=train_list)
    test_df=pd.DataFrame(data=test_list)
    train_df.to_csv(train_file, sep=',', header=None, index=False, mode='w', line_terminator='\n', encoding='utf-8')
    test_df.to_csv(test_file, sep=',', header=None, index=False, mode='w', line_terminator='\n', encoding='utf-8')