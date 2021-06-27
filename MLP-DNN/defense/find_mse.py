from defense.poisoning_NIDS_MLP import MLP_NIDS
import os
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from data_process.my_dataset import Dataset_adv, Dataset, Dataset_adv_1, Dataset_mix
import matplotlib.pyplot as plt


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    cnt = 0
    success_samples = []
    fail_samples = []
    with torch.no_grad():
        i = 0
        for data in test_loader:
            x, y = data
            target = y.float().unsqueeze(1)
            optimizer.zero_grad()
            y_pred = model(x)
            test_loss += criterion(y_pred, target)
            # pred = y_pred > 0.5
            # for j in range(x.shape[0]):
            #     if target[j] == 0 and pred[j] <= 0.5:
            #         # print(x[j])
            #         success_samples.append(x[j].detach().cpu().numpy().tolist())
            #     if target[j] == 0 and pred[j] > 0.5:
            #         # print(x[j])
            #         fail_samples.append(x[j].detach().cpu().numpy().tolist())
        # correct = len(success_samples)
        # cnt = len(success_samples) + len(fail_samples)
        mse = mean_squared_error(target, y_pred)
        print("mse:", mse)
        # return correct / cnt
        return mse


model_path = "saved_models/MLP/"
criterion = nn.BCELoss()
net = MLP_NIDS()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999))
# test_dl_1 = DataLoader(Dataset("../data/cic_2017/data_sets/1.0_train_set.csv"), batch_size=32,
#                                  shuffle=False)
# test_dl = DataLoader(Dataset("../data/cic_2017/data_sets/1.0_test_set.csv"), batch_size=32,
#                                  shuffle=False)

model_p = model_path + "{}_{}_NIDS_MLP.pt".format(4, 4)

mse_list = []
length = 32


if os.path.exists(model_p):
    checkpoint = torch.load(model_p)
    net.model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['end_epoch']
    epoch = checkpoint['epoch']
    for i in range(50):
        print(i)
        test_dl = DataLoader(Dataset("../data/cic_2017/data_sets/1.0_train_set.csv", start=i * length,
                                     len=length), batch_size=32, shuffle=False)
        mse = test(net, test_dl)
        mse_list.append(mse)

    for i in range(50, 60):
        print(i)
        test_dl = DataLoader(Dataset_adv("../data/cic_2017/adver_sets/1.0_time1_MLP_adver_train.csv", start=i * length,
                                         len=length), batch_size=32, shuffle=False)
        mse = test(net, test_dl)
        mse_list.append(mse)

# 确定子图数量
plt.figure(figsize=(6, 6.5))
# plt.subplots(1, 1)
x = range(60)

print(mse_list)
y_MLP = mse_list

# y_mlp = [0.9515221699999999, 0.95690859, 0.9516163199999999, 0.95561789, 0.95907022, 0.96712436, 0.97763829, 0.97944292, 0.98218909, 0.98540604,
#          0.97461112, 0.95807259, 0.96629716, 0.97505811, 0.9645092099999999, 0.9460039299999999, 0.9508314, 0.93152154, 0.95244055, 0.96853668,
#          0.96681051, 0.96853668, 0.97191055, 0.97606905, 0.97771675, 0.9803844599999999, 0.99066301, 0.9919184, 0.9954492, 0.9960768899999999,
#          0.99897999, 0.99913692, 0.99929384, 0.9995292299999999, 0.9995292299999999, 0.99905845, 0.99066301, 0.9919184, 0.9954492, 0.99607688,
#          0.9508314, 0.93152154, 0.95244055, 0.94171286, 0.94904345, 0.9435902, 0.9471661, 0.95020561, 0.96710173, 0.9608439099999999,
#          0.73213025, 0.73220871, 0.73213025, 0.73220871, 0.7324441, 0.74617497, 0.74617497, 0.74609651, 0.74554727, 0.74554727]

# y_DNN = [0.9338458799999999, 0.97461112, 0.95807259, 0.96629716, 0.97505811, 0.9645092099999999, 0.9460039299999999, 0.9508314, 0.93152154, 0.95244055,
#          0.94171286, 0.94904345, 0.9435902, 0.9471661, 0.95020561, 0.96710173, 0.9608439099999999, 0.96710173, 0.94448418, 0.94922224,
#          0.94448418, 0.9360808199999999, 0.9420704499999999, 0.9225818, 0.92150903, 0.9460039299999999, 0.9508314, 0.93152154, 0.95244055, 0.96853668,
#          0.96681051, 0.96853668, 0.97191055, 0.97606905, 0.97771675, 0.9803844599999999, 0.99066301, 0.9919184, 0.9954492, 0.9960768899999999,
#          0.99897999, 0.99913692, 0.99929384, 0.9995292299999999, 0.9995292299999999, 0.99905845, 0.99066301, 0.9919184, 0.9954492, 0.99607688,
#          0.77391382, 0.72751654, 0.78070803, 0.72474522, 0.77990345, 0.7853566999999999, 0.78625067, 0.7621133600000001, 0.77087431, 0.78195959]
#

plt.scatter(x, y_MLP, color='white', marker='o', edgecolors='black', s=30)
# plt.scatter(x, y_DNN, color='red', marker='x', edgecolors='red')

# 横线
plt.hlines(0.3, -1, 61, label = "MSE", color="black", linestyles='-')
plt.hlines(0.9, -1, 61, label = "ACC", color="red", linestyles='--')
plt.legend(loc='upper right', prop={'family': 'Times New Roman', 'size': 12})
# set range_y
plt.ylim((-0.1, 1.1))

# 设置坐标轴刻度
my_x_ticks = np.arange(0, 61, 10)
my_y_ticks = np.arange(-0.1, 1.1, 0.1)

plt.xticks(my_x_ticks, fontproperties='Times New Roman', size=14)
plt.yticks(my_y_ticks, fontproperties='Times New Roman', size=14)

# set x_label, y_label, title
plt.xlabel("Index", fontdict={'family': 'Times New Roman', 'size': 16})
plt.ylabel("Robust distances", fontdict={'family': 'Times New Roman', 'size': 16})
plt.title("MSE - MLP", fontdict={'family': 'Times New Roman', 'size': 16})
plt.tight_layout()
plt.show()
