import os
import traceback

try:
    path = 'saved_models/RNN/'
    # delete current epoch
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.hdf5':
            path_now = os.path.join(path, file)
            print(os.path.splitext(file)[0].split('_'))
            # if os.path.isfile(path_now) and os.path.splitext(file)[0].split('_')[0] == '{}'.format(2):
            #     # 删除查找到的文件
            #     os.remove(path_now)
except Exception as e:
    print(traceback.format_exc())
    print(e)