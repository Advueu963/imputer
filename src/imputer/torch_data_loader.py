import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class TorchDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.transform(self.data[index])


def get_torch_dataloader(data, imputation, batch_size=1, shuffle=False):
    dataset = TorchDataset(data, imputation)
    return DataLoader(dataset, batch_size, shuffle)
