from defense.poisoning_NIDS_MLP import MLP_NIDS
import os
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data_process.my_dataset import Dataset_adv, Dataset, Dataset_adv_1, Dataset_mix


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
            pred = y_pred > 0.5
            for j in range(x.shape[0]):
                if target[j] == 0 and pred[j] <= 0.5:
                    # print(x[j])
                    success_samples.append(x[j].detach().cpu().numpy().tolist())
                if target[j] == 0 and pred[j] > 0.5:
                    # print(x[j])
                    fail_samples.append(x[j].detach().cpu().numpy().tolist())
        correct = len(success_samples)
        cnt = len(success_samples) + len(fail_samples)
        print('fpr -> {}'.format(correct / cnt))
        return correct / cnt


model_path = "saved_models/MLP/"
criterion = nn.BCELoss()
net = MLP_NIDS()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999))
test_dl_1 = DataLoader(Dataset("../data/cic_2017/data_sets/1.0_train_set.csv"), batch_size=32,
                                 shuffle=False)
test_dl = DataLoader(Dataset("../data/cic_2017/data_sets/1.0_test_set.csv"), batch_size=32,
                                 shuffle=False)
train_fpr = []
test_fpr = []
for i in range(10):
    for j in range(5):
        model_p = model_path + "{}_{}_NIDS_MLP.pt".format(i, j)

        if os.path.exists(model_p):
            checkpoint = torch.load(model_p)
            net.model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['end_epoch']
            epoch = checkpoint['epoch']
            print('{} -> {}'.format(i,j))
            tr = test(net, test_dl_1)
            tr = 1 - tr
            train_fpr.append(tr)
            te = test(net, test_dl)
            te = 1-te
            test_fpr.append(te)

print(train_fpr)
print(test_fpr)
