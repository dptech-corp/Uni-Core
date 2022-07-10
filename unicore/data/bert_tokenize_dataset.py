# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch
from tokenizers import BertWordPieceTokenizer

from . import BaseWrapperDataset, LRUCacheDataset


class BertTokenizeDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dict_path: str,
        max_seq_len: int=512,
    ):
        self.dataset = dataset
        self.tokenizer = BertWordPieceTokenizer(dict_path, lowercase=True)
        self.max_seq_len = max_seq_len

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def __getitem__(self, index: int):
        raw_str = self.dataset[index]
        raw_str = raw_str.replace('<unk>', '[UNK]')
        output = self.tokenizer.encode(raw_str)
        ret = torch.Tensor(output.ids).long()
        if ret.size(0) > self.max_seq_len:
            ret = ret[:self.max_seq_len]
        return ret