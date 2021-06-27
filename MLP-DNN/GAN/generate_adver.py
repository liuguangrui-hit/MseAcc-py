import os
from numpy.core.fromnumeric import transpose
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, dataset
import numpy as np
import pandas as pd
from GAN.ResFc import ResFc


class Dataset(Dataset):
    def __init__(self, input_file):
        self.items = []
        with open(input_file, "r") as f:
            lines = f.readlines()
            # for line in lines[:10000]:  # 控制对抗样本数量
            for line in lines:  # 控制对抗样本数量
                try:
                    if 'inf' in line or 'nan' in line:
                        continue
                    self.items.append([min(10 ** 5, float(v)) for v in line.strip("\n").split(",")])
                except:
                    continue
        print(len(self.items))
        self.items = np.array(self.items, dtype=np.float32)
        self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
                np.std(self.items, axis=1, keepdims=True) + 0.00001)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return np.array(self.items[idx], dtype=np.float32)


class MLP_NIDS(nn.Module):
    def __init__(self):
        super(MLP_NIDS, self).__init__()
        self.main = nn.Sequential(

            nn.Linear(78, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 78),
            nn.ReLU(78)
        )

    def forward(self, input):
        return self.main(input)


model1 = ResFc(78, 78)

model1.load_state_dict(torch.load("generator/g10_time1_generator_DNN.pt"))
model1.eval()
print(model1)
sample_size = 1.0
f = open("../data/cic_2017/adver_sets/" + str(sample_size) + "_time1_DNN_adver_example.csv", "w")
f.close()

dl = DataLoader(Dataset("../data/cic_2017/data_sets/" + str(sample_size) + "_attack.csv"), batch_size=32, shuffle=False)

for data in dl:
    x = data
    predict = model1(x)
    # print(predict.detach().numpy())
    test = pd.DataFrame(data=predict.detach().numpy())
    print(test)

    test.to_csv("../data/cic_2017/adver_sets/" + str(sample_size) + "_time1_DNN_adver_example.csv", mode='a', encoding="gbk",
                header=0, index=0)
