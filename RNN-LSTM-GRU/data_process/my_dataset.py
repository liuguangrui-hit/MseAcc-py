import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, dataset
import numpy as np


class Dataset(Dataset):
    def __init__(self, file, start=-1, len=-1):
        self.items = []
        self.label = []

        with open(file, "r") as f:
            lines = f.readlines()[1:]
            # read all
            if len == -1:
                for line in lines:
                    try:
                        self.items.append([min(float(v), 1e6) for v in line.strip("\n").split(",")[:-1]])
                        if line.strip("\n").split(",")[-1] == 'BENIGN':
                            self.label.append(0)
                        else:
                            self.label.append(1)
                    except:
                        continue
            # split datasets
            else:
                for line in lines[start: start + len]:
                    try:
                        self.items.append([min(float(v), 1e6) for v in line.strip("\n").split(",")[:-1]])
                        if line.strip("\n").split(",")[-1] == 'BENIGN':
                            self.label.append(0)
                        else:
                            self.label.append(1)
                    except:
                        continue

        self.items = np.array(self.items, dtype=np.float32)
        self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
                np.std(self.items, axis=1, keepdims=True) + 0.00001)
        self.label = np.array(self.label)

    def __len__(self):
        # print('item的长度是',len(self.items))
        return len(self.items)

    def __getitem__(self, idx):
        # print(self.items[idx,:],self.label[idx])
        return self.items[idx, :], self.label[idx]

class Dataset_adv(Dataset):
    def __init__(self, file, start=-1, len=-1):
        self.items = []
        self.label = []

        with open(file, "r") as f:
            lines = f.readlines()
            if len == -1:
                for line in lines:
                    try:
                        self.items.append([min(float(v), 1e6) for v in line.strip("\n").split(",")])
                        self.label.append(0)
                    except:
                        continue
            else:
                for line in lines[start: start + len]:
                    try:
                        self.items.append([min(float(v), 1e6) for v in line.strip("\n").split(",")])
                        self.label.append(0)
                    except:
                        continue

        self.items = np.array(self.items, dtype=np.float32)
        # self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
        #         np.std(self.items, axis=1, keepdims=True) + 0.00001)
        self.label = np.array(self.label)

    def __len__(self):
        # print('item的长度是',len(self.items))
        return len(self.items)

    def __getitem__(self, idx):
        # print(self.items[idx,:],self.label[idx])
        return self.items[idx, :], self.label[idx]

class Dataset_adv_1(Dataset):
    def __init__(self, file, start=-1, len=-1):
        self.items = []
        self.label = []

        with open(file, "r") as f:
            lines = f.readlines()
            if len == -1:
                for line in lines:
                    try:
                        self.items.append([min(float(v), 1e6) for v in line.strip("\n").split(",")])
                        self.label.append(1)
                    except:
                        continue
            else:
                for line in lines[start: start + len]:
                    try:
                        self.items.append([min(float(v), 1e6) for v in line.strip("\n").split(",")])
                        self.label.append(1)
                    except:
                        continue

        self.items = np.array(self.items, dtype=np.float32)
        # self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
        #         np.std(self.items, axis=1, keepdims=True) + 0.00001)
        self.label = np.array(self.label)

    def __len__(self):
        # print('item的长度是',len(self.items))
        return len(self.items)

    def __getitem__(self, idx):
        # print(self.items[idx,:],self.label[idx])
        return self.items[idx, :], self.label[idx]


class Dataset_mix(Dataset):
    def __init__(self, p_file, n_file, p_start=-1, p_len=-1,n_start = -1,n_len = -1):
        self.items = []
        self.label = []
        # read normal data
        with open(p_file, "r") as f:
            lines = f.readlines()[1:]
            # read all
            if p_len == -1:
                for line in lines:
                    try:
                        self.items.append([min(float(v), 1e6) for v in line.strip("\n").split(",")[:-1]])
                        if line.strip("\n").split(",")[-1] == 'BENIGN':
                            self.label.append(0)
                        else:
                            self.label.append(1)
                    except:
                        continue
            # split datasets
            else:
                for line in lines[p_start: p_start + p_len]:
                    try:
                        self.items.append([min(float(v), 1e6) for v in line.strip("\n").split(",")[:-1]])
                        if line.strip("\n").split(",")[-1] == 'BENIGN':
                            self.label.append(0)
                        else:
                            self.label.append(1)
                    except:
                        continue

        self.items = np.array(self.items, dtype=np.float32)
        self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
                np.std(self.items, axis=1, keepdims=True) + 0.00001)
        self.items = self.items.tolist()


        # read adv data
        with open(n_file, "r") as f:
            lines = f.readlines()
            if n_len == -1:
                for line in lines:
                    try:
                        self.items.append([min(float(v), 1e6) for v in line.strip("\n").split(",")])
                        self.label.append(0)
                    except:
                        continue
            else:
                for line in lines[n_start: n_start + n_len]:
                    try:
                        self.items.append([min(float(v), 1e6) for v in line.strip("\n").split(",")])
                        self.label.append(0)
                    except:
                        continue

        self.items = np.array(self.items, dtype=np.float32)
        # self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
        #         np.std(self.items, axis=1, keepdims=True) + 0.00001)
        self.label = np.array(self.label)

    def __len__(self):
        # print('item的长度是',len(self.items))
        return len(self.items)

    def __getitem__(self, idx):
        # print(self.items[idx,:],self.label[idx])
        return self.items[idx, :], self.label[idx]