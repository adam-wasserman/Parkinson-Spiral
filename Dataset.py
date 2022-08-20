import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, values):
        self.X = X
        self.values = values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.values[idx]
