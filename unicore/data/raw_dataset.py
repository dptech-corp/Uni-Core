# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils.data.dataloader import default_collate
from functools import lru_cache
from . import UnicoreDataset


class RawLabelDataset(UnicoreDataset):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels
    
    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)

    def collater(self, samples):
        return torch.tensor(samples)


class RawArrayDataset(UnicoreDataset):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if hasattr(self.dataset, 'collater'):
            return self.dataset.collater(samples)
        else:
            return default_collate(samples)


class RawNumpyDataset(UnicoreDataset):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return torch.from_numpy(self.dataset[index])

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if hasattr(self.dataset, 'collater'):
            return self.dataset.collater(samples)
        else:
            return default_collate(samples)
