import torch
from torch.utils import data
import numpy as np

class SequenceDataset(data.Dataset):
    def __init__(self, dataset, steps=1, skip=1, variable_skip=False):
        self.dataset = dataset
        self.steps = steps
        self.skip = skip
        self.variable_skip = variable_skip
        
    def __len__(self):
        return len(self.dataset)
        
    def get_indices(self, index):
        if self.variable_skip:
            skips = np.random.randint(1, high=self.skip + 1, size=self.steps - 1)
        else:
            skips = self.skip * np.ones(self.steps - 1)
        
        offsets = np.insert(skips, 0, 0).cumsum()
        offsets -= offsets[len(offsets) // 2]
        offsets = offsets.astype(int)
        
        idx = index + offsets
        idx = np.clip(idx, 0, len(self.dataset) - 1)
        return idx

    def __getitem__(self, index):
        indices = self.get_indices(index)
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
