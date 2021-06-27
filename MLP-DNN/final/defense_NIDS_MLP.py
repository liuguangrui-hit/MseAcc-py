import datetime
import os
import shutil
import sys
import traceback

import torch
from torch import nn
from torch.utils.data import DataLoader
from data_process.my_dataset import Dataset_adv, Dataset, Dataset_adv_1, Dataset_mix
from sklearn.metrics import mean_squared_error


class MLP_NIDS(nn.Module):
    def __init__(self):
        super(MLP_NIDS, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(78, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def train(model, train_loader, epoch_):
    model.train()
    train_loss = 0
    correct = 0
    cnt = 0
    i = 0
    mse_handle = open(mse_path, mode='a+')
    # 一个epoch_ 遍历完所有数据
    time = 1
    mse_list = []
    for data in train_loader:
        x, y = data  # 获取一个batch的数据和标签
        target = y.float().unsqueeze(1)
        predict = model(x)  # 前向传播
        loss = criterion(predict, target)  # 计算这个batch的loss
        optimizer.zero_grad()  # 本batch清零梯度（loss关于weight的导数变成0）
        loss.backward()  # 反向传播
        optimizer.step()  # 更新训练参数
        train_loss += loss
        predicted = predict > 0.5
        i += 1
        correct += (predicted == target).sum().item()
        cnt += predicted.shape[0]
        mse = mean_squared_error(target, predicted)
        # print("mse:",mse)
        mse_list.append(mse)
        mse_handle.write("Train Epoch: {}\t time: {}\t mse: {}\n".format(epoch_ + 1, time, mse))
        time += 1

    loss_mean = train_loss / (i + 1)
    file_handle = open(log_path, mode='a+')
    print(
        'Train Epoch: {}\t Acc: {}/{} ({:.6f}%)\t Loss: {:.6f}'.format(epoch_ + 1, correct, cnt, (correct / cnt) * 100,
                                                                       loss_mean.item()))

    file_handle.write(
        'Train Epoch: {}\t Acc: {}/{} ({:.6f}%)\t Loss: {:.6f}\n'.format(epoch_ + 1, correct, cnt,
                                                                         (correct / cnt) * 100,
                                                                         loss_mean.item()))
    file_handle.close()
    mse_handle.close()
    return max(mse_list)


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    cnt = 0
    acc  = 0
    with torch.no_grad():
        i = 0
        for data in test_loader:
            x, y = data
            target = y.float().unsqueeze(1)
            optimizer.zero_grad()
            y_pred = model(x)
            test_loss += criterion(y_pred, target)
            pred = y_pred > 0.5
            correct += (pred == target).sum().item()
            cnt += pred.shape[0]
            i += 1
        test_loss /= (i + 1)
        acc = correct / cnt
        file_handle = open(log_path, mode='a+')
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
            test_loss, correct, cnt, 100. * correct / cnt))
        file_handle.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
            test_loss, correct, cnt, 100. * correct / cnt))
        file_handle.close()
    return acc


def roll(epoch_, val_dl):
    try:
        path = 'saved_models/MLP/'
        # delete current epoch
        for file in os.listdir(path):
            if os.path.splitext(file)[1] == '.pt':
                path_now = os.path.join(path, file)
                if os.path.isfile(path_now) and os.path.splitext(file)[0].split('_')[0] == '{}'.format(epoch_):
                    # # 删除查找到的文件
                    # os.remove(path_now)
                    # 重命名文件
                    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  # 现在
                    path_new = os.path.join(path, 'delete_' + str(nowTime) + '_' + str(file))
                    os.rename(path_now, path_new)
        # roll back
        val_min_list = []
        for i in range(5):
            model_p = "saved_models/MLP/{}_{}_NIDS_MLP.pt".format(epoch_ - 1, i)
            if os.path.exists(model_p):
                checkpoint = torch.load(model_p)
                net.model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['end_epoch']
                val_min = test(net, val_dl)
                val_min_list.append(val_min)
        if min(val_min_list) > 0.92:
            state = {'model': net.model.state_dict(), 'optimizer': optimizer.state_dict(), 'end_epoch': start_epoch}
            # update current model

            print('Roll back succeed! Current epoch : {}'.format(start_epoch))
            torch.save(state, "saved_models/NIDS_MLP.pt")
            # return start_epoch
        else:
            print('Roll back failed! Continuing...')
            if start_epoch - 1 < 3:
                print('Model train failed! Restart a train please!')
                sys.exit()
            else:
                roll(start_epoch - 1, val_dl)

    except Exception as e:
        print(traceback.format_exc())
        print(e)


# hyper-parameter
net = MLP_NIDS()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999))
criterion = nn.BCELoss()
epoch = 10
val_path = 'saved_models/val_MLP.txt'
model_path = 'saved_models/MLP/'
mse_path = 'saved_models/mse_MLP.txt'
log_path = 'saved_models/log_MLP.txt'
mse_max_value1, mse_max_value2 = 0.0, 0.4
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

    test_dl = DataLoader(Dataset("../data/cic_2017/data_sets/1.0_crossval_set.csv"),
                         batch_size=32, shuffle=False)

    # test_dl = DataLoader(Dataset_adv_1("../data/cic_2017/adver_sets/1.0_time1_MLP_adver_test.csv"),
    #                      batch_size=32, shuffle=False)
    val_dl = DataLoader(Dataset_adv_1('val_data/val_set_MLP.csv'),
                        batch_size=32, shuffle=False)

    # 1.start NIDS training
    # 模拟第6轮开始投毒
    dataset_n_len = 20000
    dataset_a_len = 5
    mse_max_list = []
    acc_min_list = []
    val_min_lsit = []
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

        print(net)
        i = 0
        poison = True
        while i < epoch:  # 训10个epoch
            print('----------- epoch: %d -----------' % (i + 1))
            # if i > 4:
            if i == 5 and poison is True:
                train_dl = DataLoader(
                    Dataset_mix("../data/cic_2017/data_sets/1.0_train_set.csv",
                                "../data/cic_2017/adver_sets/1.0_time1_MLP_adver_train.csv",
                                p_start=(i) * dataset_n_len, p_len=dataset_n_len, n_start=(i) * dataset_a_len,
                                n_len=dataset_a_len), batch_size=32, shuffle=False)
            else:
                train_dl = DataLoader(Dataset("../data/cic_2017/data_sets/1.0_train_set.csv", start=i * dataset_n_len,
                                              len=dataset_n_len), batch_size=32, shuffle=False)

            for j in range(5):
                mse_max = train(net, train_dl, i)
                mse_max_list.append(mse_max)
                acc_min = test(net, test_dl)
                acc_min_list.append(acc_min)
                val_min = test(net, val_dl)
                val_min_lsit.append(val_min)
                # 'end_epoch'用于记录下一次训练的起始epoch
                state = {'model': net.model.state_dict(), 'optimizer': optimizer.state_dict(), 'end_epoch': i + 1}
                torch.save(state, model_path + "{}_{}_NIDS_MLP.pt".format(i, j))
                torch.save(state, "saved_models/NIDS_MLP.pt")
            print('current epoch -> mse_max / val_min :', max(mse_max_list), min(val_min_lsit))
            if (i > 2) and ((max(mse_max_list) > mse_max_value2) or (min(val_min_lsit) < val_min_value1)):
                print('\033[1;31;40m')
                print('*' * 50)
                print('NIDS system exception!!! ')
                print('mse_max value range --> ({}, {})  val_min set value --> ({}, {})'
                      .format(mse_max_value1,mse_max_value2, val_min_value1, val_min_value2))
                print('*' * 50)
                print('\033[0m')
                c = input('Roll Back! or Exit or Continue? (r/e/c)')
                if c == 'e' or c == 'E':
                    print('NIDS system training exit, try to retrain!')
                    sys.exit()
                elif c == 'r' or c == 'R':
                    roll(i, val_dl)
                    model_p = "saved_models/NIDS_MLP.pt"
                    if os.path.exists(model_p):
                        checkpoint = torch.load(model_p)
                        net.model.load_state_dict(checkpoint['model'])
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        i = checkpoint['end_epoch']
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

    # continue NIDS training
    elif reuse_model and is_train:
        poison = True
        epoch = int(input("\nInput train epoch:"))
        model_p = "saved_models/NIDS_MLP.pt"
        if os.path.exists(model_p):
            checkpoint = torch.load(model_p)
            net.model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['end_epoch']
            i = start_epoch
            while i < start_epoch + epoch:  # 训10个epoch
                print('----------- epoch: %d -----------' % (i + 1))
                if i == (start_epoch + 1) and poison is True:
                    train_dl = DataLoader(
                        Dataset_mix("../data/cic_2017/data_sets/1.0_train_set.csv",
                                    "../data/cic_2017/adver_sets/1.0_time1_MLP_adver_train.csv",
                                    p_start=(i) * dataset_n_len, p_len=dataset_n_len, n_start=(i) * dataset_a_len,
                                    n_len=dataset_a_len), batch_size=32, shuffle=False)
                else:
                    train_dl = DataLoader(
                        Dataset("../data/cic_2017/data_sets/1.0_train_set.csv", start=i * dataset_n_len,
                                len=dataset_n_len), batch_size=32, shuffle=False)
                # train_dl = DataLoader(Dataset("../data/cic_2017/data_sets/1.0_train_set.csv", start=i * dataset_n_len,
                #                               len=dataset_n_len), batch_size=32, shuffle=False)

                for j in range(5):
                    mse_max = train(net, train_dl, i)
                    mse_max_list.append(mse_max)
                    acc_min = test(net, test_dl)
                    acc_min_list.append(acc_min)
                    val_min = test(net, val_dl)
                    val_min_lsit.append(val_min)
                    # 'end_epoch'用于记录下一次训练的起始epoch
                    state = {'model': net.model.state_dict(), 'optimizer': optimizer.state_dict(), 'end_epoch': i + 1}
                    torch.save(state, model_path + "{}_{}_NIDS_MLP.pt".format(i, j))
                    torch.save(state, "saved_models/NIDS_MLP.pt")
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
                        roll(i, val_dl)
                        model_p = "saved_models/NIDS_MLP.pt"
                        if os.path.exists(model_p):
                            checkpoint = torch.load(model_p)
                            net.model.load_state_dict(checkpoint['model'])
                            optimizer.load_state_dict(checkpoint['optimizer'])
                            i = checkpoint['end_epoch']
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
                loop_exit = True
            if c == '2':
                # test val
                test_dl = DataLoader(Dataset_adv_1('val_data/val_set_MLP.csv'),
                                     batch_size=32, shuffle=False)
                loop_exit = True
            if c == '3':
                # test bypass
                test_dl = DataLoader(
                    Dataset_adv_1("../data/cic_2017/adver_sets/1.0_MLP_adver_example.csv"), batch_size=32,
                    shuffle=False)
                loop_exit = True
        # model_p = "saved_models/MLP/0_4_NIDS_MLP.pt"
        model_p = "saved_models/NIDS_MLP.pt"
        if os.path.exists(model_p):
            checkpoint = torch.load(model_p)
            net.model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['end_epoch']
            acc_min = test(net, test_dl)
        else:
            start_epoch = 0
            print('No saved model, try start NIDS training！')
