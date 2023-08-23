# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch


from typing import Callable, List, Optional

# this import is for backward compatibility
from unicore.utils import (
    csv_str_list,
    eval_bool,
    eval_str_dict,
    eval_str_list,
    import_user_module,
)  # noqa


def get_training_parser(default_task="translation"):
    parser = get_parser("Trainer", default_task)
    add_dataset_args(parser, train=True)
    add_distributed_training_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    return parser


def get_validation_parser(default_task=None):
    parser = get_parser("Validation", default_task)
    add_dataset_args(parser, train=True)
    add_distributed_training_args(parser)
    group = parser.add_argument_group("Evaluation")
    add_common_eval_args(group)
    return parser


def parse_args_and_arch(
    parser: argparse.ArgumentParser,
    input_args: List[str] = None,
    parse_known: bool = False,
    suppress_defaults: bool = False,
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None,
):
    """
    Args:
        parser (ArgumentParser): the parser
        input_args (List[str]): strings to parse, defaults to sys.argv
        parse_known (bool): only parse known arguments, similar to
            `ArgumentParser.parse_known_args`
        suppress_defaults (bool): parse while ignoring all default values
        modify_parser (Optional[Callable[[ArgumentParser], None]]):
            function to modify the parser, e.g., to set default values
    """
    if suppress_defaults:
        # Parse args without any default values. This requires us to parse
        # twice, once to identify all the necessary task/model args, and a second
        # time with all defaults set to None.
        args = parse_args_and_arch(
            parser,
            input_args=input_args,
            parse_known=parse_known,
            suppress_defaults=False,
        )
        suppressed_parser = argparse.ArgumentParser(add_help=False, parents=[parser])
        suppressed_parser.set_defaults(**{k: None for k, v in vars(args).items()})
        args = suppressed_parser.parse_args(input_args)
        return argparse.Namespace(
            **{k: v for k, v in vars(args).items() if v is not None}
        )

    from unicore.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY, MODEL_REGISTRY

    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--user-dir", default=None)
    usr_args, _ = usr_parser.parse_known_args(input_args)
    import_user_module(usr_args)

    if modify_parser is not None:
        modify_parser(parser)

    # The parser doesn't know about model/loss/optimizer-specific args, so
    # we parse twice. First we parse the model/loss/optimizer, then we
    # parse a second time after adding the *-specific arguments.
    # If input_args is given, we will parse those args instead of sys.argv.
    args, _ = parser.parse_known_args(input_args)

    # Add model-specific args to parser.
    if hasattr(args, "arch"):
        model_specific_group = parser.add_argument_group(
            "Model-specific configuration",
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        if args.arch in ARCH_MODEL_REGISTRY:
            ARCH_MODEL_REGISTRY[args.arch].add_args(model_specific_group)
        elif args.arch in MODEL_REGISTRY:
            MODEL_REGISTRY[args.arch].add_args(model_specific_group)
        else:
            raise RuntimeError()

    if hasattr(args, "task"):
        from unicore.tasks import TASK_REGISTRY

        TASK_REGISTRY[args.task].add_args(parser)

    # Add *-specific args to parser.
    from unicore.registry import REGISTRIES

    for registry_name, REGISTRY in REGISTRIES.items():
        choice = getattr(args, registry_name, None)
        if choice is not None:
            cls = REGISTRY["registry"][choice]
            if hasattr(cls, "add_args"):
                cls.add_args(parser)

    # Modify the parser a second time, since defaults may have been reset
    if modify_parser is not None:
        modify_parser(parser)

    # Parse a second time.
    if parse_known:
        args, extra = parser.parse_known_args(input_args)
    else:
        args = parser.parse_args(input_args)
        extra = None
    # Post-process args.
    if (
        hasattr(args, "batch_size_valid") and args.batch_size_valid is None
    ) or not hasattr(args, "batch_size_valid"):
        args.batch_size_valid = args.batch_size
    args.bf16 = getattr(args, "bf16", False)

    if getattr(args, "seed", None) is None:
        args.seed = 1  # default seed for training
        args.no_seed_provided = True
    else:
        args.no_seed_provided = False

    args.validate_with_ema = getattr(args, "validate_with_ema", False)
    # Apply architecture configuration.
    if hasattr(args, "arch") and args.arch in ARCH_CONFIG_REGISTRY:
        ARCH_CONFIG_REGISTRY[args.arch](args)

    if parse_known:
        return args, extra
    else:
        return args


def get_parser(desc, default_task="test"):
    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--user-dir", default=None)
    usr_args, _ = usr_parser.parse_known_args()
    import_user_module(usr_args)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    # fmt: off
    parser.add_argument('--no-progress-bar', action='store_true', help='disable progress bar')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='log progress every N batches (when progress bar is disabled)')
    parser.add_argument('--log-format', default=None, help='log format to use',
                        choices=['json', 'none', 'simple', 'tqdm'])
    parser.add_argument('--tensorboard-logdir', metavar='DIR', default='',
                        help='path to save logs for tensorboard, should match --logdir '
                             'of running tensorboard (default: no tensorboard logging)')
    parser.add_argument('--wandb-project', metavar='DIR', default='',
                        help='name of wandb project, empty for no wandb logging, for wandb login, use env WANDB_API_KEY. You can also use team_name/project_name for project name.')
    parser.add_argument('--wandb-name', metavar='DIR', default='',
                        help='wandb run/id name, empty for no wandb logging, for wandb login, use env WANDB_API_KEY')
    parser.add_argument('--seed', default=1, type=int, metavar='N',
                        help='pseudo random number generator seed')
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of CUDA')
    parser.add_argument('--fp16', action='store_true', help='use FP16')
    parser.add_argument('--bf16', action='store_true', help='use BF16')
    parser.add_argument('--bf16-sr', action='store_true', help='use stachostic rounding for bf16')
    parser.add_argument('--allreduce-fp32-grad', action='store_true', help='use fp32-grads in fp16/bf16 mode. --ddp-backend should be no_c10d')
    parser.add_argument('--fp16-no-flatten-grads', action='store_true', help="don't flatten FP16 grads tensor")
    parser.add_argument('--fp16-init-scale', default=2 ** 7, type=int,
                        help='default FP16 loss scale')
    parser.add_argument('--fp16-scale-window', type=int,
                        help='number of updates before increasing loss scale')
    parser.add_argument('--fp16-scale-tolerance', default=0.0, type=float,
                        help='pct of updates that can overflow before decreasing the loss scale')
    parser.add_argument('--min-loss-scale', default=1e-4, type=float, metavar='D',
                        help='minimum FP16 loss scale, after which training is stopped')
    parser.add_argument('--threshold-loss-scale', type=float,
                        help='threshold FP16 loss scale from below')
    parser.add_argument('--user-dir', default=None,
                        help='path to a python module containing custom extensions (tasks and/or architectures)')
    parser.add_argument('--empty-cache-freq', default=0, type=int,
                        help='how often to clear the PyTorch CUDA cache (0 to disable)')
    parser.add_argument('--all-gather-list-size', default=16384, type=int,
                        help='number of bytes reserved for gathering stats from workers')
    parser.add_argument('--suppress-crashes', action='store_true', help="suppress crashes when training with the entry point so that the "
                        "main method can return a value (useful for sweeps)")
    parser.add_argument('--profile', action='store_true', help="enable autograd profiler emit_nvtx")
    parser.add_argument('--ema-decay', default=-1.0, type=float, help="enable moving average for model weights")
    parser.add_argument("--validate-with-ema", action="store_true")
    

    from unicore.registry import REGISTRIES
    for registry_name, REGISTRY in REGISTRIES.items():
        parser.add_argument(
            '--' + registry_name.replace('_', '-'),
            default=REGISTRY['default'],
            choices=REGISTRY['registry'].keys(),
        )

    # Task definitions can be found under unicore/tasks/
    from unicore.tasks import TASK_REGISTRY
    parser.add_argument('--task', metavar='TASK', default=default_task,
                        choices=TASK_REGISTRY.keys(),
                        help='task')
    # fmt: on
    return parser


def add_dataset_args(parser, train=False, gen=False):
    group = parser.add_argument_group("Dataset and data loading")
    # fmt: off
    group.add_argument('--num-workers', default=1, type=int, metavar='N',
                       help='how many subprocesses to use for data loading')
    group.add_argument('--skip-invalid-size-inputs-valid-test', action='store_true',
                       help='ignore too long or too short lines in valid and test set')
    group.add_argument('--batch-size', '--max-sentences', type=int, metavar='N',
                       help='maximum number of sentences in a batch')
    group.add_argument('--required-batch-size-multiple', default=1, type=int, metavar='N',
                       help='batch size will be a multiplier of this value')
    group.add_argument('--data-buffer-size', default=10, type=int,
                       help='Number of batches to preload')
    group.add_argument('--train-subset', default='train', metavar='SPLIT',
                        choices=['train', 'valid', 'test', 'train.small'],
                        help='data subset to use for training (train, valid, test)')
    group.add_argument('--valid-subset', default='valid', metavar='SPLIT',
                        help='comma separated list of data subsets to use for validation'
                            ' (train, valid, valid1, test, test1)')
    group.add_argument('--validate-interval', type=int, default=1, metavar='N',
                        help='validate every N epochs')
    group.add_argument('--validate-interval-updates', type=int, default=0, metavar='N',
                        help='validate every N updates')
    group.add_argument('--validate-after-updates', type=int, default=0, metavar='N',
                        help='dont validate until reaching this many updates')
    group.add_argument('--fixed-validation-seed', default=None, type=int, metavar='N',
                        help='specified random seed for validation')
    group.add_argument('--disable-validation', action='store_true',
                        help='disable validation')
    group.add_argument('--batch-size-valid', type=int, metavar='N',
                        help='maximum number of sentences in a validation batch'
                            ' (defaults to --max-sentences)')
    group.add_argument('--max-valid-steps', type=int, metavar='N',
                        help='How many batches to evaluate')
    group.add_argument('--curriculum', default=0, type=int, metavar='N',
                        help='don\'t shuffle batches for first N epochs')
    # fmt: on
    return group


def add_distributed_training_args(parser):
    group = parser.add_argument_group("Distributed training")
    # fmt: off
    group.add_argument('--distributed-world-size', type=int, metavar='N',
                       default=max(1, torch.cuda.device_count()),
                       help='total number of GPUs across all nodes (default: all visible GPUs)')
    group.add_argument('--distributed-rank', default=0, type=int,
                       help='rank of the current worker')
    group.add_argument('--distributed-backend', default='nccl', type=str,
                       help='distributed backend')
    group.add_argument('--distributed-init-method', default=None, type=str,
                       help='typically tcp://hostname:port that will be used to '
                            'establish initial connetion')
    group.add_argument('--distributed-port', default=-1, type=int,
                       help='port number (not required if using --distributed-init-method)')
    group.add_argument('--device-id', '--local_rank', default=0, type=int,
                       help='which GPU to use (usually configured automatically)')
    group.add_argument('--distributed-no-spawn', action='store_true',
                       help='do not spawn multiple processes even if multiple GPUs are visible')
    group.add_argument('--ddp-backend', default='c10d', type=str,
                       choices=['c10d', 'apex', 'no_c10d'],
                       help='DistributedDataParallel backend')
    group.add_argument('--bucket-cap-mb', default=25, type=int, metavar='MB',
                       help='bucket size for reduction')
    group.add_argument('--fix-batches-to-gpus', action='store_true',
                       help='don\'t shuffle batches between GPUs; this reduces overall '
                            'randomness and may affect precision but avoids the cost of '
                            're-reading the data')
    group.add_argument('--find-unused-parameters', default=False, action='store_true',
                       help='disable unused parameter detection (not applicable to '
                       'no_c10d ddp-backend')
    group.add_argument('--fast-stat-sync', default=False, action='store_true',
                        help='Enable fast sync of stats between nodes, this hardcodes to '
                        'sync only some default stats from logging_output.')
    group.add_argument('--broadcast-buffers', default=False, action='store_true',
                        help="Copy non-trainable parameters between GPUs, such as "
                             "batchnorm population statistics")
    group.add_argument('--nprocs-per-node', default=max(1, torch.cuda.device_count()), type=int,
                       help="number of GPUs in each node. An allreduce operation across GPUs in "
                            "a node is very fast. Hence, we do allreduce across GPUs in a node, "
                            "and gossip across different nodes")
    # fmt: on
    return group


def add_optimization_args(parser):
    group = parser.add_argument_group("Optimization")
    # fmt: off
    group.add_argument('--max-epoch', '--me', default=0, type=int, metavar='N',
                       help='force stop training at specified epoch')
    group.add_argument('--max-update', '--mu', default=0, type=int, metavar='N',
                       help='force stop training at specified update')
    group.add_argument('--stop-time-hours', default=0, type=float,
                       help="force stop training after specified cumulative time (if >0)")
    group.add_argument('--clip-norm', default=0, type=float, metavar='NORM',
                       help='clip threshold of gradients')
    group.add_argument('--per-sample-clip-norm', default=0, type=float, metavar='PNORM',
                       help='clip threshold of gradients, before gradient sync over workers. In fp16/bf16 mode, --fp32-grad should be set, and --dpp-backend should be no_c10d')
    group.add_argument('--update-freq', default='1', metavar='N1,N2,...,N_K',
                       type=lambda uf: eval_str_list(uf, type=int),
                       help='update parameters every N_i batches, when in epoch i')
    group.add_argument('--lr', '--learning-rate', default='0.25', type=eval_str_list,
                       metavar='LR_1,LR_2,...,LR_N',
                       help='learning rate for the first N epochs; all epochs >N using LR_N'
                            ' (note: this may be interpreted differently depending on --lr-scheduler)')
    group.add_argument('--stop-min-lr', default=-1, type=float, metavar='LR',
                       help='stop training when the learning rate reaches this minimum')
    # fmt: on
    return group


def add_checkpoint_args(parser):
    group = parser.add_argument_group("Checkpointing")
    # fmt: off
    group.add_argument('--save-dir', metavar='DIR', default='checkpoints',
                       help='path to save checkpoints')
    group.add_argument('--tmp-save-dir', metavar='DIR', default='./',
                       help='path to temporarily save checkpoints')
    group.add_argument('--restore-file', default='checkpoint_last.pt',
                       help='filename from which to load checkpoint '
                            '(default: <save-dir>/checkpoint_last.pt')
    group.add_argument('--finetune-from-model', type=str,
                       help="finetune from a pretrained model; note that meters and lr scheduler will be reset")
    group.add_argument('--load-from-ema', action="store_true",
                       help="finetune from a pretrained model; note that meters and lr scheduler will be reset")
    group.add_argument('--reset-dataloader', action='store_true',
                       help='if set, does not reload dataloader state from the checkpoint')
    group.add_argument('--reset-lr-scheduler', action='store_true',
                       help='if set, does not load lr scheduler state from the checkpoint')
    group.add_argument('--reset-meters', action='store_true',
                       help='if set, does not load meters from the checkpoint')
    group.add_argument('--reset-optimizer', action='store_true',
                       help='if set, does not load optimizer state from the checkpoint')
    group.add_argument('--optimizer-overrides', default="{}", type=str, metavar='DICT',
                       help='a dictionary used to override optimizer args when loading a checkpoint')
    group.add_argument('--save-interval', type=int, default=1, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('--save-interval-updates', type=int, default=0, metavar='N',
                       help='save a checkpoint (and validate) every N updates')
    group.add_argument('--keep-interval-updates', type=int, default=-1, metavar='N',
                       help='keep the last N checkpoints saved with --save-interval-updates')
    group.add_argument('--keep-last-epochs', type=int, default=-1, metavar='N',
                       help='keep last N epoch checkpoints')
    group.add_argument('--keep-best-checkpoints', type=int, default=-1, metavar='N',
                       help='keep best N checkpoints based on scores')
    group.add_argument('--no-save', action='store_true',
                       help='don\'t save models or checkpoints')
    group.add_argument('--no-epoch-checkpoints', action='store_true',
                       help='only store last and best checkpoints')
    group.add_argument('--no-last-checkpoints', action='store_true',
                       help='don\'t store last checkpoints')
    group.add_argument('--no-save-optimizer-state', action='store_true',
                       help='don\'t save optimizer-state as part of checkpoint')
    group.add_argument('--best-checkpoint-metric', type=str, default='loss',
                       help='metric to use for saving "best" checkpoints')
    group.add_argument('--maximize-best-checkpoint-metric', action='store_true',
                       help='select the largest metric value for saving "best" checkpoints')
    group.add_argument('--patience', type=int, default=-1, metavar='N',
                       help="early stop training if valid performance doesn't "
                            "improve for N consecutive validation runs; note "
                            "that this is influenced by --validate-interval")
    group.add_argument('--checkpoint-suffix', type=str, default="",
                       help="suffix to add to the checkpoint file name")
    # fmt: on
    return group


def add_common_eval_args(group):
    # fmt: off
    group.add_argument('--path', metavar='FILE',
                       help='path(s) to model file(s), colon separated')
    group.add_argument('--quiet', action='store_true',
                       help='only print final scores')
    group.add_argument('--model-overrides', default="{}", type=str, metavar='DICT',
                       help='a dictionary used to override model args at generation '
                            'that were used during model training')
    group.add_argument('--results-path', metavar='RESDIR', type=str, default=None,
                       help='path to save eval results (optional)"')
    # fmt: on


def add_model_args(parser):
    group = parser.add_argument_group("Model configuration")
    # fmt: off

    # Model definitions can be found under unicore/models/
    #
    # The model architecture can be specified in several ways.
    # In increasing order of priority:
    # 1) model defaults (lowest priority)
    # 2) --arch argument
    # 3) --encoder/decoder-* arguments (highest priority)
    from unicore.models import ARCH_MODEL_REGISTRY
    group.add_argument('--arch', '-a', default='fconv', metavar='ARCH', required=True,
                       choices=ARCH_MODEL_REGISTRY.keys(),
                       help='Model Architecture')
    # fmt: on
    return group
