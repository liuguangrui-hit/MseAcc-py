import re

with open('saved_models/mse_DNN.txt', "r", encoding='utf-8') as f:  # 设置文件对象
    lines = f.readlines()  # 可以是随便对文件的操作

    mse_list = []
    mse = 0

    for line in lines[3*5*625:5*5*625]:
        # print(line)
        items = line.strip("\n").split(" ")
        if 'mse:' in items:
            # print(items)
            mse = float(items[6])
            mse_list.append(mse)

print(max(mse_list),min(mse_list))

with open('saved_models/rec_DNN.txt', "r", encoding='utf-8') as f:  # 设置文件对象
    lines = f.readlines()  # 可以是随便对文件的操作

    val_list = []
    i = 0

    for line in lines[:5*5*3+1]:
        i += 1
        # print(line)
        items = line.strip("\n").split(" ")
        if 'Test' in items and i % 3 == 0:
            # print(items)
            val = re.split('\(|%', items[7])
            val_list.append(float(val[1])/100)

print(max(val_list),min(val_list))

