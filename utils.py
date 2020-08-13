import numpy as np
import torch
from torch.utils.data import Dataset

def read_csv(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    results = []
    for line in lines:
        results.append(line.split(','))
    return np.asarray(results, dtype=np.float32)

class ToyDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.data = read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x
