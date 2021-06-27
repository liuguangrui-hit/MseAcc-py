import re

# find mse_range and val_range
with open('saved_models/mse_GRU.txt', "r", encoding='utf-8') as f:  # 设置文件对象
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

with open('saved_models/rec_GRU.txt', "r", encoding='utf-8') as f:  # 设置文件对象
    lines = f.readlines()  # 可以是随便对文件的操作

    val_list = []
    i = 0

    for line in lines:
        # print(line)
        items = line.strip("\n").split(" ")
        if 'Test' in items:
            i += 1
            if i > 10 and i < 26:
                # print(items)
                val = re.split('\(|%', items[4])
                val_list.append(float(val[1])/100)

print(max(val_list),min(val_list))

