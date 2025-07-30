import numpy as np
import torch
from torch.utils.data import Dataset

class ValveDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        # Remapping (100, 90, 80, 73) -> (0,1,2,3)
        unique = sorted(np.unique(y))
        self.y = torch.tensor([unique.index(v) for v in y], dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]