import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from imputer.baseline_imputer import impute, ImputeMode

class TorchDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        print(index)
        return self.transform(self.data[index])


def get_torch_dataloader(data, coalitions, imputationType, baseline=None, batch_size=1, shuffle=False):
    transform = (lambda a: impute(a, baseline, coalitions, imputationType))
    dataset = TorchDataset(data, transform)
    print(len(dataset))
    return DataLoader(dataset, batch_size, shuffle)

data = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
coaltitions = np.array([True, False, False, True])
baseline = np.mean(data.transpose(), axis=1)
imputationType = ImputeMode.STATIC

dataloader = get_torch_dataloader(data, coaltitions, imputationType, baseline)




