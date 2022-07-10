# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data.dataloader import default_collate

from . import UnicoreDataset


class BaseWrapperDataset(UnicoreDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if hasattr(self.dataset, "collater"):
            return self.dataset.collater(samples)
        else:
            return default_collate(samples)

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def attr(self, attr: str, index: int):
        return self.dataset.attr(attr, index)

    def prefetch(self, indices):
        self.dataset.prefetch(indices)

    def batch_by_size(
        self,
        indices,
        batch_size=None,
        required_batch_size_multiple=1,
    ):
        return self.dataset.batch_by_size(
            indices,
            batch_size=batch_size,
            required_batch_size_multiple=required_batch_size_multiple,
        )

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return self.dataset.can_reuse_epoch_itr_across_epochs

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
