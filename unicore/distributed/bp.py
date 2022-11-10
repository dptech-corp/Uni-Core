import torch
import torch.distributed as dist
from torch._C._distributed_c10d import BroadcastOptions, AllreduceOptions, ReduceOp
from .comm_group import scg

def broadcast(tensor, src):
    """ broadcast tensor from src rank in bp group """
    if scg.get_bp_world_size() == 1:
        return tensor

    assert src in [0, 1], "Branch Parallel is only support bp_degree=2 now!"
    
    group = scg.get_bp_group()

    opts = BroadcastOptions()
    opts.rootRank = src
    opts.rootTensor = 0
    work = group.broadcast([tensor], opts)
    work.wait()

def all_reduce(tensor):
    """ allreduce a tensor in bp group """
    if scg.get_bp_world_size() == 1:
        return tensor

    group = scg.get_bp_group()

    opts = AllreduceOptions()
    opts.reduceOp = ReduceOp.SUM

    work = group.allreduce([tensor], opts)
    work.wait()

    return tensor

class SyncEvoformerResults(torch.autograd.Function):
    """ A PyLayer Op broadcast gradient in backward stage """
    @staticmethod
    def forward(ctx, outer, msa, pair):
        broadcast(outer, 0)
        if scg.get_bp_rank_in_group() == 1:
            pair += outer
        broadcast(pair, 1)
        broadcast(msa, 0)
        return msa.clone(), pair.clone()

    @staticmethod
    def backward(ctx, *grad_output):
        msa_grad = grad_output[0]
        pair_grad = grad_output[1]

        if scg.get_bp_rank_in_group() == 0:
            pair_grad = torch.zeros_like(pair_grad)

        outer_grad = pair_grad.clone()
        broadcast(outer_grad, 1)
        
        return outer_grad.clone(), msa_grad.clone(), pair_grad.clone()

def sync_evoformer_results(outer, msa, pair):
    """ a warpper for boradcast gradient in backward stage """
    if scg.get_bp_world_size() == 1:
        return msa, pair

    if torch.is_grad_enabled() and outer.requires_grad and msa.requires_grad and pair.requires_grad:
        return msa, pair

    msa, pair = SyncEvoformerResults.apply(outer, msa, pair)
        
    return msa, pair