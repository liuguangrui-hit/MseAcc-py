import datetime
import os
path = 'saved_models/MLP/'
for file in os.listdir(path):
    if os.path.splitext(file)[1] == '.pt':
        print(os.path.splitext(file)[0].split('_')[0])
        path_now = os.path.join(path, file)
        if os.path.isfile(path_now) and os.path.splitext(file)[0].split('_')[0] == '{}'.format(0):
            # # 删除查找到的文件
            # os.remove(path_now)
            # 重命名文件
            nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  # 现在
            path_new = os.path.join(path, 'delete_' + str(nowTime) + '_' + str(file))
            os.rename(path_now, path_new)