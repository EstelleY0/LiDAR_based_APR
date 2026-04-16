import torch
from torch.utils import data
import numpy as np

class SequenceDataset(data.Dataset):
    def __init__(self, dataset, steps=1, skip=1):
        self.dataset = dataset
        self.steps = steps
        self.skip = skip
        
    def __len__(self):
        return len(self.dataset) - (self.steps - 1) * self.skip
        
    def __getitem__(self, index):
        indices = [index + i * self.skip for i in range(self.steps)]
        items = [self.dataset[i] for i in indices]
        
        res = {}
        for key in items[0].keys():
            val = items[0][key]
            if isinstance(val, torch.Tensor):
                res[key] = torch.stack([item[key] for item in items], dim=0)
            elif isinstance(val, (int, float, np.number)):
                res[key] = torch.tensor([item[key] for item in items])
            else:
                res[key] = [item[key] for item in items]
        return res
