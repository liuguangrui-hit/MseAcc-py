from defense.poisoning_NIDS_DNN import DNN_NIDS
import os
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data_process.my_dataset import Dataset_adv, Dataset, Dataset_adv_1, Dataset_mix


def test(model, test_loader,is_success):
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
                if target[j] == 1 and pred[j] <= 0.5:
                    # print(x[j])
                    fail_samples.append(x[j].detach().cpu().numpy().tolist())
                if target[j] == 1 and pred[j] > 0.5:
                    # print(x[j])
                    success_samples.append(x[j].detach().cpu().numpy().tolist())
            correct += (pred == target).sum().item()
            cnt += pred.shape[0]
            i += 1
        test_loss /= (i + 1)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
            test_loss, correct, cnt, 100. * correct / cnt))
        if is_success:
            return success_samples
        else:
            return fail_samples



model_path = "saved_models/DNN/"
criterion = nn.BCELoss()
net = DNN_NIDS()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999))
val_test_set_a = []
val_test_set_b = []
val_test_set = []
val_adv_set_a = []
val_adv_set_b = []
val_adv_set = []
model_p = model_path + "4_4_NIDS_DNN.pt"

if os.path.exists(model_p):
    checkpoint = torch.load(model_p)
    net.model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['end_epoch']
    epoch = checkpoint['epoch']

    test_dl = DataLoader(Dataset("../data/cic_2017/data_sets/1.0_test_set.csv",start=0,len=20000), batch_size=32, shuffle=False)
    val_test_set_a = test(net, test_dl, True)

model_p = model_path + "5_1_NIDS_DNN.pt"

if os.path.exists(model_p):
    checkpoint = torch.load(model_p)
    net.model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['end_epoch']
    epoch = checkpoint['epoch']

    test_dl = DataLoader(Dataset("../data/cic_2017/data_sets/1.0_test_set.csv",start=0,len=20000), batch_size=32, shuffle=False)
    val_test_set_b = test(net, test_dl, True)

for a in val_test_set_a:
    if a in val_test_set_b:
        val_test_set.append(a)


model_p = model_path + "4_4_NIDS_DNN.pt"

if os.path.exists(model_p):
    checkpoint = torch.load(model_p)
    net.model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['end_epoch']
    epoch = checkpoint['epoch']

    test_dl = DataLoader(Dataset_adv_1("../data/cic_2017/adver_sets/1.0_time1_DNN_adver_test.csv",start=0,len=20000),
                         batch_size=32, shuffle=False)
    val_adv_set_a = test(net, test_dl, True)

model_p = model_path + "5_1_NIDS_DNN.pt"

if os.path.exists(model_p):
    checkpoint = torch.load(model_p)
    net.model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['end_epoch']
    epoch = checkpoint['epoch']

    test_dl = DataLoader(Dataset_adv_1("../data/cic_2017/adver_sets/1.0_time1_DNN_adver_test.csv",start=0,len=20000),
                         batch_size=32, shuffle=False)
    val_adv_set_b = test(net, test_dl, False)

for a in val_adv_set_a:
    if a in val_adv_set_b:
        val_adv_set.append(a)

print(len(val_test_set))
print(val_test_set[0])
print(len(val_adv_set))
print(val_adv_set[0])

val_set = []
val_set.extend(val_adv_set)
val_set.extend(val_test_set)

items = np.array(val_set, dtype=np.float32)
test=pd.DataFrame(data=items)
test.to_csv('val_data/val_set_DNN.csv', sep=',', header=None, index=False, mode='w', line_terminator='\n', encoding='utf-8')