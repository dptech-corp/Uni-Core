#!/usr/bin/env python3 -u
# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import time
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

import numpy as np
import torch
from unicore import (
    checkpoint_utils,
    options,
    tasks,
    utils,
)
from unicore.data import iterators
from unicore.distributed import utils as distributed_utils
from unicore.logging import meters, metrics, progress_bar
from unicore.trainer import Trainer
from multiprocessing.pool import ThreadPool


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unicore_cli.train")


def main(args) -> None:
    utils.import_user_module(args)
    utils.set_jit_fusion_options()

    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"
    metrics.reset()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)
        checkpoint_utils.verify_checkpoint_directory(args.tmp_save_dir)
        ckp_copy_thread = ThreadPool(processes=1)
    else:
        ckp_copy_thread = None

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    assert args.loss, "Please specify loss to train a model"

    # Build model and loss
    model = task.build_model(args)
    loss = task.build_loss(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("loss: {}".format(loss.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(getattr(p, "_orig_size", p).numel() for p in model.parameters()),
            sum(
                getattr(p, "_orig_size", p).numel()
                for p in model.parameters()
                if p.requires_grad
            ),
        )
    )

    # Build trainer
    trainer = Trainer(args, task, model, loss)
    logger.info("training on {} devices (GPUs)".format(args.distributed_world_size))
    logger.info(
        "batch size per device = {}".format(
            args.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        args,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=False,
    )
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= args.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={args.stop_min_lr})"
            )
            break

        # train for one epoch
        valid_losses, should_stop = train(
            args, trainer, task, epoch_itr, ckp_copy_thread
        )
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=False,
        )
    train_meter.stop()
    if ckp_copy_thread is not None:
        ckp_copy_thread.close()
        ckp_copy_thread.join()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(args, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    args.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    args, trainer: Trainer, task: tasks.UnicoreTask, epoch_itr, ckp_copy_thread
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        wandb_project=(
            args.wandb_project if distributed_utils.is_master(args) else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        args=args,
    )

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = args.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    max_update = args.max_update or math.inf

    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % args.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            args, trainer, task, epoch_itr, valid_subsets, end_of_epoch, ckp_copy_thread
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def validate_and_save(
    args,
    trainer: Trainer,
    task: tasks.UnicoreTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
    ckp_copy_thread,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = args.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if args.stop_time_hours > 0 and training_time_hours > args.stop_time_hours:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {args.stop_time_hours} hour(s)"
        )

    do_save = (
        (
            end_of_epoch
            and epoch_itr.epoch % args.save_interval == 0
            and not args.no_epoch_checkpoints
        )
        or should_stop
        or (
            args.save_interval_updates > 0
            and num_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates >= args.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (
            end_of_epoch
            and epoch_itr.epoch % args.validate_interval == 0
            and not args.no_epoch_checkpoints
        )
        or should_stop
        or (
            args.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % args.validate_interval_updates == 0
        )
    ) and not args.disable_validation

    # Validate
    valid_losses = [None]
    if do_validate:
        with utils.validate_with_ema(trainer, ema=args.validate_with_ema):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(args, valid_losses[0])

    # Save checkpoint
    checkpoint_utils.save_checkpoint(
        args,
        trainer,
        epoch_itr,
        valid_losses[0],
        ckp_copy_thread,
        do_save=(do_save or should_stop),
    )

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    args,
    trainer: Trainer,
    task: tasks.UnicoreTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    seed = None
    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        seed = args.fixed_validation_seed

    with utils.torch_seed(seed):
        trainer.begin_valid_epoch(epoch_itr.epoch)
        valid_losses = []
        for subset in subsets:
            logger.info('begin validation on "{}" subset'.format(subset))

            # Initialize data iterator
            itr = trainer.get_valid_iterator(subset).next_epoch_itr(
                shuffle=False, set_dataset_epoch=False  # use a fixed valid set
            )
            progress = progress_bar.progress_bar(
                itr,
                log_format=args.log_format,
                log_interval=args.log_interval,
                epoch=epoch_itr.epoch,
                prefix=f"valid on '{subset}' subset",
                tensorboard_logdir=(
                    args.tensorboard_logdir
                    if distributed_utils.is_master(args)
                    else None
                ),
                default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
            )

            # create a new root metrics aggregator so validation metrics
            # don't pollute other aggregators (e.g., train meters)
            with metrics.aggregate(new_root=True) as agg:
                logging_outputs = []
                for i, sample in enumerate(progress):
                    if args.max_valid_steps is not None and i > args.max_valid_steps:
                        break
                    inner_logging_outputs = trainer.valid_step(sample)
                    logging_outputs.extend(inner_logging_outputs)
                task.reduce_metrics(logging_outputs, trainer.get_loss(), subset)

            # log validation stats
            stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
            progress.print(stats, tag=subset, step=trainer.get_num_updates())
            if args.best_checkpoint_metric in stats:
                valid_losses.append(stats[args.best_checkpoint_metric])
        return valid_losses


def get_valid_stats(args, trainer: Trainer, stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if (
        hasattr(checkpoint_utils.save_checkpoint, "best")
        and args.best_checkpoint_metric in stats
    ):
        key = "best_{0}".format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args.best_checkpoint_metric],
        )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    try:
        if args.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(args, main)
        else:
            distributed_utils.call_main(args, main)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            time.sleep(1)
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    cli_main()
