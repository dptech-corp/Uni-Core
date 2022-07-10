# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import warnings
from argparse import Namespace
from typing import Any, Callable, Dict, List

import torch
from unicore import metrics, utils
from unicore.data import UnicoreDataset, data_utils, iterators

logger = logging.getLogger(__name__)


class StatefulContainer(object):

    _state: Dict[str, Any] = dict()
    _factories: Dict[str, Callable[[], Any]] = dict()

    def add_factory(self, name, factory: Callable[[], Any]):
        self._factories[name] = factory

    def merge_state_dict(self, state_dict: Dict[str, Any]):
        self._state.update(state_dict)

    @property
    def state_dict(self) -> Dict[str, Any]:
        return self._state

    def __getattr__(self, name):
        if name not in self._state and name in self._factories:
            self._state[name] = self._factories[name]()

        if name in self._state:
            return self._state[name]

        raise AttributeError(f"Task state has no factory for attribute {name}")


class UnicoreTask(object):
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Loss and calculating the loss.

    Tasks have limited statefulness. In particular, state that needs to be
    saved to/loaded from checkpoints needs to be stored in the `self.state`
    :class:`StatefulContainer` object. For example::

        self.state.add_factory("dictionary", self.load_dictionary)
        print(self.state.dictionary)  # calls self.load_dictionary()

    This is necessary so that when loading checkpoints, we can properly
    recreate the task state after initializing the task instance.
    """

    @classmethod
    def add_args(cls, parser):
        """Add task-specific arguments to the parser."""
        pass

    @staticmethod
    def logging_outputs_can_be_summed(loss, is_train) -> bool:
        """
        Whether the logging outputs returned by `train_step` and `valid_step` can
        be summed across workers prior to calling `reduce_metrics`.
        Setting this to True will improves distributed training speed.
        """
        return loss.logging_outputs_can_be_summed(is_train)

    args: Namespace
    datasets: Dict[str, UnicoreDataset]
    dataset_to_epoch_iter: Dict[UnicoreDataset, Any]
    state: StatefulContainer = None

    def __init__(self, args: Namespace, **kwargs):
        self.args = args
        self.datasets = dict()
        self.dataset_to_epoch_iter = dict()
        self.state = StatefulContainer()


    @classmethod
    def setup_task(cls, args: Namespace, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (Namespace): parsed command-line arguments
        """
        return cls(args, **kwargs)

    def has_sharded_data(self, split):
        return os.pathsep in getattr(self.args, "data", "")

    def load_dataset(
        self,
        split: str,
        combine: bool = False,
        **kwargs
    ):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
            combine (bool): combines a split segmented into pieces into one dataset
        """
        raise NotImplementedError

    def dataset(self, split):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)

        Returns:
            a :class:`~unicore.data.UnicoreDataset` corresponding to *split*
        """
        from unicore.data import UnicoreDataset

        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        if not isinstance(self.datasets[split], UnicoreDataset):
            raise TypeError("Datasets are expected to be of type UnicoreDataset")
        return self.datasets[split]

    def can_reuse_epoch_itr(self, dataset):
        # We can reuse the epoch iterator across epochs as long as the dataset
        # hasn't disabled it. We default to ``False`` here, although in practice
        # this will be ``True`` for most datasets that inherit from
        # ``UnicoreDataset`` due to the base implementation there.
        return getattr(dataset, "can_reuse_epoch_itr_across_epochs", False)

    def get_batch_iterator(
        self,
        dataset,
        batch_size=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~unicore.data.UnicoreDataset): dataset to batch
            batch_size (int, optional): max number of samples in each
                batch (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `UnicoreTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~unicore.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
            dataset
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.info("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]
        else:
            logger.info("get EpochBatchIterator for epoch {}".format(epoch))

        assert isinstance(dataset, UnicoreDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # create mini-batches with given size constraints
        batch_sampler = dataset.batch_by_size(
            indices,
            batch_size=batch_size,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            disable_shuffling=self.disable_shuffling(),
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

    def build_model(self, args: Namespace):
        """
        Build the :class:`~unicore.models.BaseUnicoreModel` instance for this
        task.

        Returns:
            a :class:`~unicore.models.BaseUnicoreModel` instance
        """
        from unicore import models
        return models.build_model(args, self)

    def build_loss(self, args: Namespace):
        """
        Build the :class:`~unicore.losses.UnicoreLoss` instance for
        this task.

        Args:
            args (Namespace): configration object

        Returns:
            a :class:`~unicore.losses.UnicoreLoss` instance
        """
        from unicore import losses

        return losses.build_loss(args, self)

    def train_step(
        self, sample, model, loss, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *loss*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~unicore.data.UnicoreDataset`.
            model (~unicore.models.BaseUnicoreModel): the model
            loss (~unicore.losses.UnicoreLoss): the loss
            optimizer (~unicore.optim.UnicoreOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = loss(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, loss, test=False):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = loss(model, sample)
        return loss, sample_size, logging_output

    def optimizer_step(self, optimizer, model, update_num):
        optimizer.step()

    def build_dataset_for_inference(
        self, src_tokens: List[torch.Tensor], src_lengths: List[int], **kwargs
    ) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        pass

    def begin_valid_epoch(self, epoch, model):
        """Hook function called before the start of each validation epoch."""
        pass

    def reduce_metrics(self, logging_outputs, loss, split='train'):
        """Aggregate logging outputs from data parallel training."""
        if not any("bsz" in log for log in logging_outputs):
            warnings.warn(
                "bsz not found in Loss logging outputs, cannot log bsz"
            )
        else:
            bsz = sum(log.get("bsz", 0) for log in logging_outputs)
            metrics.log_scalar("bsz", bsz, priority=190, round=1)

        loss.__class__.reduce_metrics(logging_outputs, split)

    def state_dict(self):
        if self.state is not None:
            return self.state.state_dict
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if self.state is not None:
            self.state.merge_state_dict(state_dict)

    def disable_shuffling(self) -> bool:
        return False