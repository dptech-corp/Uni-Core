# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a network across multiple GPUs.
"""

import contextlib
import logging
import os
import sys
import time
from itertools import chain
from typing import Any, Dict, List
import torch
from unicore import checkpoint_utils, models, optim, utils
from unicore.distributed import utils as distributed_utils
from unicore.logging import meters, metrics
from unicore.nan_detector import NanDetector
from unicore.optim import lr_scheduler
from unicore.ema import ExponentialMovingAverageModel


logger = logging.getLogger(__name__)


class Trainer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, args, task, model, loss):

        self.args = args
        self.task = task

        # catalog shared parameters
        shared_params = _catalog_shared_params(model)
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # copy model and loss to current device/dtype
        self._loss = loss
        self._model = model
        if args.fp16:
            self._loss = self._loss.half()
            self._model = self._model.half()
        elif args.bf16:
            self._loss = self._loss.bfloat16()
            self._model = self._model.bfloat16()
        if (
            # the DistributedUnicoreModel wrapper will handle moving to device,
            # so only handle cases which don't use the wrapper
            not self.use_distributed_wrapper
        ):
            self._loss = self._loss.to(device=self.device)
            self._model = self._model.to(device=self.device)

        # check that shared parameters are preserved after device transfer
        for shared_param in shared_params:
            ref = _get_module_by_path(self._model, shared_param[0])
            for path in shared_param[1:]:
                logger.info(
                    "detected shared parameter: {} <- {}".format(shared_param[0], path)
                )
                _set_module_by_path(self._model, path, ref)

        self._dummy_batch = None  # indicates we don't have a dummy batch at first
        self._total_train_steps = None
        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._warn_once = set()
        self._wrapped_loss = None
        self._wrapped_model = None

        if self.cuda and self.data_parallel_world_size > 1:
            self._grad_norm_buf = torch.cuda.DoubleTensor(self.data_parallel_world_size)
        else:
            self._grad_norm_buf = None

        # get detailed cuda environment
        if self.cuda:
            self.cuda_env = utils.CudaEnvironment()
            if self.data_parallel_world_size > 1:
                self.cuda_env_arr = distributed_utils.all_gather_list(
                    self.cuda_env, group=distributed_utils.get_global_group()
                )
            else:
                self.cuda_env_arr = [self.cuda_env]
            if self.data_parallel_rank == 0:
                utils.CudaEnvironment.pretty_print_cuda_env_list(self.cuda_env_arr)
        else:
            self.cuda_env = None
            self.cuda_env_arr = None

        # add ema
        if args.validate_with_ema:
            assert args.ema_decay > 0, "valid with ema must with ema_decay > 0"

        if args.ema_decay > 0 and (
            self.data_parallel_rank == 0 or args.validate_with_ema
        ):

            assert (self.args.fp16 or self.args.bf16), "ema must with fp16 or bf16"
            self.ema = ExponentialMovingAverageModel(
                model,
                args.ema_decay,
            )

        else:
            self.ema = None
        metrics.log_start_time("wall", priority=790, round=2)

        self._start_time = time.time()
        self._previous_training_time = 0
        self._cumulative_training_time = None

    def reinitialize(self):
        """Reinitialize the Trainer, typically after model params change."""
        self._lr_scheduler = None
        self._optimizer = None
        self._wrapped_loss = None
        self._wrapped_model = None

    @property
    def data_parallel_world_size(self):
        if self.args.distributed_world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()

    @property
    def data_parallel_process_group(self):
        return distributed_utils.get_data_parallel_group()

    @property
    def data_parallel_rank(self):
        if self.args.distributed_world_size == 1:
            return 0
        return distributed_utils.get_data_parallel_rank()

    @property
    def is_data_parallel_master(self):
        # NOTE: this returns true for all model parallel replicas with data
        # parallel rank 0
        return self.data_parallel_rank == 0

    @property
    def use_distributed_wrapper(self) -> bool:
        return self.data_parallel_world_size > 1

    @property
    def should_save_checkpoint_on_current_rank(self) -> bool:
        """Indicates whether to save checkpoints on the current DDP rank."""
        return self.is_data_parallel_master

    @property
    def checkpoint_suffix(self) -> str:
        """Suffix to add to the checkpoint file name."""
        return self.args.checkpoint_suffix or ""

    @property
    def loss(self):
        if self._wrapped_loss is None:
            if utils.has_parameters(self._loss) and self.use_distributed_wrapper:
                self._wrapped_loss = models.DistributedUnicoreModel(
                    self.args,
                    self._loss,
                    process_group=self.data_parallel_process_group,
                    device=self.device,
                )
            else:
                self._wrapped_loss = self._loss
        return self._wrapped_loss

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.use_distributed_wrapper:
                self._wrapped_model = models.DistributedUnicoreModel(
                    self.args,
                    self._model,
                    process_group=self.data_parallel_process_group,
                    device=self.device,
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _build_optimizer(self):
        params = list(
            filter(
                lambda p: p.requires_grad,
                chain(self.model.parameters(), self.loss.parameters()),
            )
        )
        if self.args.per_sample_clip_norm > 0:
            assert self.args.ddp_backend == "no_c10d"
            assert self.args.batch_size == 1
        if self.args.fp16 or self.args.bf16:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                logger.info(
                    "NOTE: your device does NOT support faster training with --fp16, "
                    "please switch to FP32 which is likely to be faster"
                )
            self._optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)

            if self.args.allreduce_fp32_grad:
                assert self.args.ddp_backend == "no_c10d"
            if self.args.per_sample_clip_norm > 0:
                assert self.args.allreduce_fp32_grad
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                logger.info("NOTE: your device may support faster training with --fp16")
            self._optimizer = optim.build_optimizer(self.args, params)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(
            self.args,
            self.optimizer,
            self._total_train_steps,
        )
        self._lr_scheduler.step_update(0)

    def state_dict(self):
        state_dict = {
            "args": self.args,
            "model": self.model.state_dict(),
            "loss": (
                self.loss.state_dict() if utils.has_parameters(self.loss) else None
            ),
            "optimizer_history": (self._optim_history or [])
            + [
                {
                    "loss_name": self.get_loss().__class__.__name__,
                    "optimizer_name": self.optimizer.__class__.__name__,
                    "lr_scheduler_state": self.lr_scheduler.state_dict(),
                    "num_updates": self.get_num_updates(),
                }
            ],
            "task_state": self.task.state_dict() if self.task is not None else {},
            "extra_state": {
                "metrics": metrics.state_dict(),
                "previous_training_time": self.cumulative_training_time(),
            },
        }
        if not self.args.no_save_optimizer_state:
            state_dict["last_optimizer_state"] = self.optimizer.state_dict()
        if self.ema is not None:
            state_dict["ema"] = self.ema.state_dict()
        return state_dict

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        logger.info(f"Saving checkpoint to {filename}")
        # call state_dict on all ranks in case it needs internal communication
        state_dict = utils.move_to_cpu(self.state_dict())
        state_dict["extra_state"].update(extra_state)
        if self.should_save_checkpoint_on_current_rank:
            checkpoint_utils.torch_persistent_save(
                state_dict,
                filename,
            )
        logger.info(f"Finished saving checkpoint to {filename}")

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        """
        Load all training state from a checkpoint file.
        rank = 0 will load the checkpoint, and then broadcast it to all
        other ranks.
        """
        extra_state, self._optim_history, last_optim_state = None, [], None

        logger.info(f"Preparing to load checkpoint {filename}")
        is_distributed = self.data_parallel_world_size > 1
        is_master = self.data_parallel_rank == 0
        bexists = None
        if is_master:
            bexists = os.path.isfile(filename)
        if is_distributed:
            bexists = distributed_utils.broadcast_object(
                bexists,
                src_rank=0,
                group=self.data_parallel_process_group,
                dist_device=self.device,
            )

        had_loaded_model = False
        ema_loaded = False
        if bexists:
            state = None
            if is_master:
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    filename,
                )
            if is_distributed:
                logger.info("Broadcast checkpoint from rank_0")
                state = distributed_utils.broadcast_object(
                    state,
                    src_rank=0,
                    group=self.data_parallel_process_group,
                    dist_device=self.device,
                )
            last_optim_state = state.get("last_optimizer_state", None)
            ema_state = state.get("ema", None)

            # load model parameters
            try:
                if self.args.load_from_ema:
                    logger.info("loading ema state to model")
                    errors = self.model.load_state_dict(
                        ema_state["params"], strict=False, model_args=self.args
                    )
                    ema_loaded = True
                else:
                    errors = self.model.load_state_dict(
                        state["model"], strict=False, model_args=self.args
                    )
                    # save memory for later steps
                    del state["model"]
                    had_loaded_model = True

                if errors.missing_keys:
                    logger.warning(
                        "Error in loading model state, missing_keys "
                        + str(errors.missing_keys)
                    )
                if errors.unexpected_keys:
                    logger.warning(
                        "Error in loading model state, unexpected_keys "
                        + str(errors.unexpected_keys)
                    )
                if utils.has_parameters(self.get_loss()):
                    self.get_loss().load_state_dict(state["loss"], strict=True)
                    del state["loss"]

            except Exception:
                raise Exception(
                    "Cannot load model parameters from checkpoint {}; "
                    "please ensure that the architectures match.".format(filename)
                )
            extra_state = state["extra_state"] if "extra_state" in state else None
            self._optim_history = (
                state["optimizer_history"] if "optimizer_history" in state else None
            )

            if (
                ema_state is not None
                and self.ema is not None
                and not self.args.load_from_ema
            ):
                logger.info(f"Loading EMA state...")
                self.ema.load_state_dict(ema_state)
            elif self.ema is not None and not ema_loaded:
                logger.info(
                    f"Cannot find EMA state in checkpoint, load model weight to ema directly"
                )
                self.ema = ExponentialMovingAverageModel(
                    self._model, decay=self.ema.decay
                )

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert (
                last_optim["loss_name"] == self.get_loss().__class__.__name__
            ), f"Loss does not match; please reset the optimizer (--reset-optimizer). {last_optim['loss_name']} vs {self.get_loss().__class__.__name__}"
            assert (
                last_optim["optimizer_name"] == self.optimizer.__class__.__name__
            ), f"Optimizer does not match; please reset the optimizer (--reset-optimizer). {last_optim['optimizer_name']} vs {self.optimizer.__class__.__name__}"

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim["lr_scheduler_state"])

            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self.set_num_updates(last_optim["num_updates"])

        if extra_state is not None:
            itr_state = extra_state["train_iterator"]
            epoch = itr_state["epoch"]

            if "previous_training_time" in extra_state:
                self._previous_training_time = extra_state["previous_training_time"]
                self._start_time = time.time()

            # self.lr_step(epoch)

            if (
                itr_state.get("version", 1) >= 2
                and itr_state["iterations_in_epoch"] == 0
            ):
                # reset meters at start of epoch
                reset_meters = True

            if "metrics" in extra_state and not reset_meters:
                metrics.load_state_dict(extra_state["metrics"])

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in metrics.get_meters("default"):
                    if isinstance(meter, meters.TimeMeter):
                        meter.reset()

            logger.info(
                "Loaded checkpoint {} (epoch {} @ {} updates)".format(
                    filename, epoch, self.get_num_updates()
                )
            )

        elif had_loaded_model:
            logger.info("Loaded checkpoint {}".format(filename))
        elif ema_loaded:
            logger.info("Loaded ema state from checkpoint {}".format(filename))
        else:
            logger.info("No existing checkpoint found {}".format(filename))

        return extra_state

    def get_train_iterator(
        self,
        epoch,
        combine=True,
        load_dataset=True,
        data_selector=None,
        shard_batch_itr=True,
        disable_iterator_cache=False,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        if load_dataset:
            logger.info("loading train data for epoch {}".format(epoch))
            self.task.load_dataset(
                self.args.train_subset,
                epoch=epoch,
                combine=combine,
                data_selector=data_selector,
            )
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset(self.args.train_subset),
            batch_size=self.args.batch_size,
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=self.data_parallel_world_size if shard_batch_itr else 1,
            shard_id=self.data_parallel_rank if shard_batch_itr else 0,
            num_workers=self.args.num_workers,
            epoch=epoch,
            data_buffer_size=self.args.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
        )
        self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def init_total_train_steps(self, epoch_itr):
        if self.args.max_epoch > 0:
            self._total_train_steps = (
                (len(epoch_itr) + 1) // self.args.update_freq[0] * self.args.max_epoch
            )
        else:
            self._total_train_steps = self.args.max_update

    def get_valid_iterator(
        self,
        subset,
        disable_iterator_cache=False,
    ):
        """Return an EpochBatchIterator over given validation subset for a given epoch."""
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset(subset),
            batch_size=self.args.batch_size_valid,
            ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=self.data_parallel_world_size,
            shard_id=self.data_parallel_rank,
            num_workers=self.args.num_workers,
            # always pass a fixed "epoch" to keep validation data consistent
            # across training epochs
            epoch=1,
            data_buffer_size=self.args.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
        )
        self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def begin_epoch(self, epoch):
        """Called at the beginning of each epoch."""
        logger.info("begin training epoch {}".format(epoch))

        self.lr_step_begin_epoch(epoch)

        # task specific setup per epoch
        self.task.begin_epoch(epoch, self.get_model())

    def begin_valid_epoch(self, epoch):
        """Called at the beginning of each validation epoch."""

        # task specific setup per validation epoch
        self.task.begin_valid_epoch(epoch, self.get_model())

    def reset_dummy_batch(self, batch):
        self._dummy_batch = batch

    @metrics.aggregate("train")
    def train_step(self, samples, raise_oom=False):
        """Do forward, backward and parameter update."""
        self.model.train()
        self.loss.train()
        self.zero_grad()

        metrics.log_start_time("train_wall", priority=800, round=2)

        # forward and backward pass
        logging_outputs, sample_size, ooms = [], 0, 0
        for i, sample in enumerate(samples):  # delayed update loop
            sample, is_dummy_batch = self._prepare_sample(sample)

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                if (
                    self.data_parallel_world_size > 1
                    and hasattr(self.model, "no_sync")
                    and i < len(samples) - 1
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            try:
                with maybe_no_sync():
                    # use different seed for different rank in training, otherwise the dropout will be the same in different workers.
                    with utils.torch_seed(
                        self.args.seed,
                        self.get_num_updates(),
                        i,
                        self.data_parallel_rank,
                    ):
                        # forward and backward
                        loss, sample_size_i, logging_output = self.task.train_step(
                            sample=sample,
                            model=self.model,
                            loss=self.loss,
                            optimizer=self.optimizer,
                            update_num=self.get_num_updates(),
                            ignore_grad=is_dummy_batch,
                        )
                        del loss
                    if self.args.per_sample_clip_norm > 0:
                        self.optimizer.per_sample_clip_grad_norm(
                            self.args.per_sample_clip_norm
                        )

                logging_outputs.append(logging_output)
                sample_size += sample_size_i

                # emptying the CUDA cache after the first step can
                # reduce the chance of OOM
                if self.cuda and self.get_num_updates() == 0:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if raise_oom:
                        raise e
                    logger.warning(
                        "attempting to recover from OOM in forward/backward pass"
                    )
                    ooms += 1
                    self.zero_grad()
                    if self.cuda:
                        torch.cuda.empty_cache()
                    if self.args.distributed_world_size == 1:
                        return None
                else:
                    raise e

        if is_dummy_batch:
            if torch.is_tensor(sample_size):
                sample_size.zero_()
            else:
                sample_size *= 0.0

        if torch.is_tensor(sample_size):
            sample_size = sample_size.float()
        else:
            sample_size = float(sample_size)

        local_sample_size = sample_size
        # gather logging outputs from all replicas
        if self._sync_stats():
            train_time = self._local_cumulative_training_time()
            logging_outputs, (
                sample_size,
                ooms,
                total_train_time,
            ) = self._aggregate_logging_outputs(
                logging_outputs,
                sample_size,
                ooms,
                train_time,
                ignore=is_dummy_batch,
                is_train=True,
            )
            self._cumulative_training_time = (
                total_train_time / self.data_parallel_world_size
            )

        overflow = False
        try:
            with torch.autograd.profiler.record_function("reduce-grads"):
                # reduce gradients across workers
                self.optimizer.all_reduce_grads(self.model)
                if utils.has_parameters(self.loss):
                    self.optimizer.all_reduce_grads(self.loss)

            with torch.autograd.profiler.record_function("multiply-grads"):
                # multiply gradients by (data_parallel_size / sample_size) since
                # DDP normalizes by the number of data parallel workers for
                # improved fp16 precision.
                # Thus we get (sum_of_gradients / sample_size) at the end.
                # In case of fp16, this step also undoes loss scaling.
                # (Debugging note: Some optimizers perform this scaling on the
                # fly, so inspecting model.parameters() or optimizer.params may
                # still show the original, unscaled gradients.)
                numer = self.data_parallel_world_size if self._sync_stats() else 1

                self.optimizer.multiply_grads(numer / (sample_size or 1.0))
                # Note: (sample_size or 1.0) handles the case of a zero gradient, in a
                # way that avoids CPU/device transfers in case sample_size is a GPU or
                # TPU object. The assumption is that the gradient itself is also 0.

            with torch.autograd.profiler.record_function("clip-grads"):
                # clip grads
                grad_norm = self.clip_grad_norm(self.args.clip_norm)

            self._check_grad_norms(grad_norm)
            if not torch.isfinite(grad_norm).all():
                # check local gradnorm single GPU case, trigger NanDetector
                raise FloatingPointError("gradients are Nan/Inf")

            with torch.autograd.profiler.record_function("optimizer"):
                # fixed the seed in case for the stochastic rounding in different ranks
                with utils.torch_seed(self.args.seed, self.get_num_updates()):
                    # take an optimization step
                    self.task.optimizer_step(
                        self.optimizer,
                        model=self.model,
                        update_num=self.get_num_updates(),
                    )
            if self.ema is not None:
                with torch.autograd.profiler.record_function("ema"):
                    self.ema.update(self.optimizer.fp32_params)

        except FloatingPointError:
            # re-run the forward and backward pass with hooks attached to print
            # out where it fails
            self.zero_grad()
            with NanDetector(self.get_model()):
                for i, sample in enumerate(samples):
                    sample, _ = self._prepare_sample(sample)
                    with utils.torch_seed(
                        self.args.seed,
                        self.get_num_updates(),
                        i,
                        self.data_parallel_rank,
                    ):
                        self.task.train_step(
                            sample,
                            self.model,
                            self.loss,
                            self.optimizer,
                            self.get_num_updates(),
                            ignore_grad=False,
                        )
            raise
        except OverflowError as e:
            overflow = True
            logger.info(
                f"NOTE: gradient overflow detected, ignoring gradient, {str(e)}"
            )
            grad_norm = torch.tensor(0.0).cuda()
            self.zero_grad()
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._log_oom(e)
                logger.error("OOM during optimization, irrecoverable")
            raise e

        logging_output = None
        if not overflow:
            self.set_num_updates(self.get_num_updates() + 1)

            if self.cuda and self.cuda_env is not None:
                # log minimum free memory over the iteration
                gb_used = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
                torch.cuda.reset_peak_memory_stats()
                gb_free = self.cuda_env.total_memory_in_GB - gb_used
                metrics.log_scalar("gb_free", gb_free, priority=1500, round=1, weight=0)

            # log stats
            logging_output = self._reduce_and_log_stats(
                logging_outputs,
                sample_size,
                grad_norm,
            )

            # clear CUDA cache to reduce memory fragmentation
            if (
                self.cuda
                and self.args.empty_cache_freq > 0
                and (
                    (self.get_num_updates() + self.args.empty_cache_freq - 1)
                    % self.args.empty_cache_freq
                )
                == 0
            ):
                torch.cuda.empty_cache()

        if self.args.fp16:
            metrics.log_scalar(
                "loss_scale",
                self.optimizer.scaler.loss_scale,
                priority=700,
                round=4,
                weight=0,
            )

        metrics.log_stop_time("train_wall")
        return logging_output

    @metrics.aggregate("valid")
    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""
        with torch.no_grad():
            self.model.eval()
            self.loss.eval()

            sample, is_dummy_batch = self._prepare_sample(sample)

            try:
                _loss, sample_size, logging_output = self.task.valid_step(
                    sample, self.model, self.loss
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if not raise_oom:
                        logger.warning(
                            "ran out of memory in validation step, retrying batch"
                        )
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad = None  # free some memory
                        if self.cuda:
                            torch.cuda.empty_cache()
                        return self.valid_step(sample, raise_oom=True)
                raise e

            logging_outputs = [logging_output]
            if is_dummy_batch:
                if torch.is_tensor(sample_size):
                    sample_size.zero_()
                else:
                    sample_size *= 0.0

        # gather logging outputs from all replicas
        if self.data_parallel_world_size > 1:
            logging_outputs, (sample_size,) = self._aggregate_logging_outputs(
                logging_outputs,
                sample_size,
                ignore=is_dummy_batch,
                is_train=False,
            )

        return logging_outputs

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step_begin_epoch(self, epoch):
        """Adjust the learning rate at the beginning of the epoch."""
        self.lr_scheduler.step_begin_epoch(epoch)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate at the end of the epoch."""
        self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        new_lr = self.lr_scheduler.step_update(self.get_num_updates())
        if isinstance(new_lr, dict):
            for k, v in new_lr.items():
                metrics.log_scalar(f"lr_{k}", v, weight=0, priority=300)
            new_lr = new_lr.get("default", next(iter(new_lr.values())))
        else:
            metrics.log_scalar("lr", new_lr, weight=0, priority=300)
        return new_lr

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_loss(self):
        """Get the (non-wrapped) loss instance."""
        return self._loss

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()
        metrics.log_scalar("num_updates", self._num_updates, weight=0, priority=200)

    def clip_grad_norm(self, clip_norm):
        return self.optimizer.clip_grad_norm(clip_norm)

    def cumulative_training_time(self):
        if self._cumulative_training_time is None:
            # single GPU
            return self._local_cumulative_training_time()
        else:
            return self._cumulative_training_time

    def _local_cumulative_training_time(self):
        """Aggregate training time in seconds."""
        return time.time() - self._start_time + self._previous_training_time

    def _prepare_sample(self, sample, is_dummy=False):
        if sample == "DUMMY":
            raise Exception(
                "Trying to use an uninitialized 'dummy' batch. This usually indicates "
                "that the total number of batches is smaller than the number of "
                "participating GPUs. Try reducing the batch size or using fewer GPUs."
            )

        if sample is None or len(sample) == 0:
            assert (
                self._dummy_batch is not None and len(self._dummy_batch) > 0
            ), "Invalid dummy batch: {}".format(self._dummy_batch)
            sample, _ = self._prepare_sample(self._dummy_batch, is_dummy=True)
            return sample, True

        if self.cuda:
            sample = utils.move_to_cuda(sample)

        def apply_half(t):
            if t.dtype is torch.float32:
                return t.half()
            return t

        def apply_bfloat16(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.bfloat16)
            return t

        # Please manually convert data type by yourself.
        # if self.args.fp16:
        #     sample = utils.apply_to_sample(apply_half, sample)

        # if self.args.bf16:
        #     sample = utils.apply_to_sample(apply_bfloat16, sample)

        if self._dummy_batch == "DUMMY":
            self._dummy_batch = sample

        return sample, False

    def _sync_stats(self):
        # Return True if it's using multiple GPUs and DDP or multiple GPUs with
        if self.data_parallel_world_size == 1:
            return False
        else:
            return True

    def _log_oom(self, exc):
        msg = "OOM: Ran out of memory with exception: {}".format(exc)
        logger.warning(msg)
        if torch.cuda.is_available() and hasattr(torch.cuda, "memory_summary"):
            for device_idx in range(torch.cuda.device_count()):
                logger.warning(torch.cuda.memory_summary(device=device_idx))
        sys.stderr.flush()

    def _aggregate_logging_outputs(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
        is_train=False,
    ):
        if self.task.__class__.logging_outputs_can_be_summed(
            self.get_loss(), is_train=is_train
        ):
            return self._fast_stat_sync_sum(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )
        else:
            return self._all_gather_list_sync(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )

    def _all_gather_list_sync(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. all_gather_list_sync is
        suitable when logging outputs are complex types.
        """
        if ignore:
            logging_outputs = []
        results = list(
            zip(
                *distributed_utils.all_gather_list(
                    [logging_outputs] + list(extra_stats_to_sum),
                    max_size=getattr(self.args, "all_gather_list_size", 16384),
                    group=self.data_parallel_process_group,
                )
            )
        )
        logging_outputs, extra_stats_to_sum = results[0], results[1:]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
        return logging_outputs, extra_stats_to_sum

    def _fast_stat_sync_sum(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. fast_stat_sync_sum is
        faster than all_gather_list_sync, but is only suitable when
        logging outputs are scalars and can be summed. Note that
        *logging_outputs* cannot contain any nested dicts/lists.
        """
        data = {}
        for i, stat in enumerate(extra_stats_to_sum):
            data["extra_stats_" + str(i)] = stat
        if len(logging_outputs) > 0:
            log_keys = list(logging_outputs[0].keys())
            for k in log_keys:
                if not ignore:
                    v = sum(log[k] for log in logging_outputs if k in log)
                else:
                    v = logging_outputs[0][k]
                    v = torch.zeros_like(v) if torch.is_tensor(v) else 0
                data["logging_outputs_" + k] = v
        else:
            log_keys = None

        data = distributed_utils.all_reduce_dict(
            data, device=self.device, group=self.data_parallel_process_group
        )

        extra_stats_to_sum = [
            data["extra_stats_" + str(i)] for i in range(len(extra_stats_to_sum))
        ]
        if log_keys is not None:
            logging_outputs = [{k: data["logging_outputs_" + k] for k in log_keys}]
        else:
            logging_outputs = []
        return logging_outputs, extra_stats_to_sum

    def _check_grad_norms(self, grad_norm):
        """Check that grad norms are consistent across workers."""
        if self._grad_norm_buf is not None:
            self._grad_norm_buf.zero_()
            self._grad_norm_buf[self.data_parallel_rank] = grad_norm
            distributed_utils.all_reduce(
                self._grad_norm_buf, group=self.data_parallel_process_group
            )

            def is_consistent(tensor):
                max_abs_diff = torch.max(torch.abs(tensor - tensor[0]))
                return (
                    torch.isfinite(tensor).all()
                    and (max_abs_diff / (tensor[0] + 1e-6) < 1e-6).all()
                )

            if not is_consistent(self._grad_norm_buf):
                pretty_detail = "\n".join(
                    "rank {:3d} = {:.8f}".format(r, n)
                    for r, n in enumerate(self._grad_norm_buf.tolist())
                )
                error_detail = "grad_norm across the workers:\n{}\n".format(
                    pretty_detail
                )
                # use FloatingPointError to trigger NanDetector
                raise FloatingPointError(
                    "Fatal error: gradients are inconsistent between workers. "
                    "Try --ddp-backend=legacy_ddp. "
                    "Or are you mixing up different generation of GPUs in training?"
                    + "\n"
                    + "-" * 80
                    + "\n{}\n".format(error_detail)
                    + "-" * 80
                )

    def _reduce_and_log_stats(self, logging_outputs, sample_size, grad_norm=None):
        if grad_norm is not None and (
            not torch.is_tensor(grad_norm) or torch.isfinite(grad_norm)
        ):
            metrics.log_speed("ups", 1.0, priority=100, round=2)
            metrics.log_scalar("gnorm", grad_norm, priority=400, round=3)
            if self.args.clip_norm > 0:
                metrics.log_scalar(
                    "clip",
                    torch.where(
                        grad_norm > self.args.clip_norm,
                        grad_norm.new_tensor(100),
                        grad_norm.new_tensor(0),
                    ),
                    priority=500,
                    round=1,
                )

        with metrics.aggregate() as agg:
            if logging_outputs is not None:
                self.task.reduce_metrics(logging_outputs, self.get_loss())
                del logging_outputs

            # extra warning for losses that don't properly log a loss value
            if "loss" not in agg:
                if "loss" not in self._warn_once:
                    self._warn_once.add("loss")
                    logger.warning(
                        "Loss.reduce_metrics did not log a 'loss' value, "
                        "which may break some functionality"
                    )
                metrics.log_scalar("loss", -1)

            logging_output = agg.get_smoothed_values()
            logging_output["sample_size"] = sample_size
            for key_to_delete in ["ppl", "wps", "wpb", "bsz"]:
                if key_to_delete in logging_output:
                    del logging_output[key_to_delete]
            return logging_output


def _catalog_shared_params(module, memo=None, prefix=""):
    if memo is None:
        first_call = True
        memo = {}
    else:
        first_call = False
    for name, param in module._parameters.items():
        if param is None:
            continue
        param_prefix = prefix + ("." if prefix else "") + name
        if param not in memo:
            memo[param] = []
        memo[param].append(param_prefix)
    for name, m in module._modules.items():
        if m is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        _catalog_shared_params(m, memo, submodule_prefix)
    if first_call:
        return [x for x in memo.values() if len(x) > 1]


def _get_module_by_path(module, path):
    path = path.split(".")
    for name in path:
        module = getattr(module, name)
    return module


def _set_module_by_path(module, path, value):
    path = path.split(".")
    for name in path[:-1]:
        module = getattr(module, name)
    setattr(module, path[-1], value)
