# src/training/dataset.py
import torch
from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, seq_len, features), y: (N, features)
        self.X = X.astype('float32')
        self.y = y.astype('float32')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])
