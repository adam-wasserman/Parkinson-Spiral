import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, Xm,Xs,Xc,values):
        self.Xm = Xm
        self.Xs = Xs
        self.Xc = Xc
        self.values = values

    def __len__(self):
        return len(self.Xm)

    def __getitem__(self, idx):
        return self.Xm[idx], self.Xs[idx],self.Xc[idx], self.values[idx]
