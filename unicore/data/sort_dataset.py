# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from . import BaseWrapperDataset, data_utils


class SortDataset(BaseWrapperDataset):
    def __init__(self, dataset, sort_order):
        super().__init__(dataset)
        if not isinstance(sort_order, (list, tuple)):
            sort_order = [sort_order]
        self.sort_order = sort_order

        assert all(len(so) == len(dataset) for so in sort_order)

    def ordered_indices(self):
        return np.lexsort(self.sort_order)


class EpochShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset, size, seed):
        super().__init__(dataset)
        self.size = size
        self.seed = seed
        self.set_epoch(1)
    
    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        with data_utils.numpy_seed(self.seed + epoch - 1):
            self.sort_order = np.random.permutation(self.size)

    def ordered_indices(self):
        return self.sort_order

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False