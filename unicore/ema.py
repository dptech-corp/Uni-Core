from copy import deepcopy
from itertools import chain
from unicore.optim.fp16_optimizer import pad_numel
import torch


class ExponentialMovingAverageModel:
    def __init__(self, model, decay, init_param=None):
        self.model_ema = deepcopy(model).float()
        self.decay = decay
        self.name2param, self.param = self.flatten_parameters(model, init_param)

    def flatten_parameters(self, model, init_param):
        # get ordered name
        dtype_grouped_names = dict()
        ordered_dtype = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                if p.dtype not in dtype_grouped_names:
                    dtype_grouped_names[p.dtype] = []
                    ordered_dtype.append(p.dtype)
                dtype_grouped_names[p.dtype].append(n)

        ordered_names = list(chain(*(dtype_grouped_names[n] for n in ordered_dtype)))

        name2param = dict()
        for n, p in self.model_ema.named_parameters():
            name2param[n] = p
        cur_params = [name2param[n] for n in ordered_names]
        total_param_size = sum(pad_numel(p.data.numel()) for p in cur_params)
        flatten_param = cur_params[0].new(0).float().new_zeros(total_param_size)

        offset = 0
        for p in cur_params:
            numel = p.data.numel()
            flatten_param[offset : offset + numel].copy_(p.data.view(-1))
            p.data = flatten_param.data[offset : offset + numel].view(*p.shape)
            offset += pad_numel(numel)
        flatten_param = torch.nn.Parameter(flatten_param)
        if init_param is not None:
            assert torch.allclose(init_param, flatten_param), "ema init error!"
        torch.cuda.empty_cache()
        return name2param, flatten_param

    def update_one_param(self, ema_param, new_param):
        diff = ema_param - new_param
        diff *= 1 - self.decay
        ema_param -= diff

    def update(self, new_param, is_flattened):
        if is_flattened:
            with torch.no_grad():
                self.update_one_param(self.param, new_param)
        else:
            with torch.no_grad():
                for n, p in new_param:
                    if n in self.name2param:
                        self.update_one_param(self.name2param[n], p)

    def load_state_dict(self, state_dict):
        self.model_ema.load_state_dict(state_dict["params"])
        self.decay = state_dict["decay"] if "decay" in state_dict else self.decay

    def state_dict(self):
        return {
            "params": self.model_ema.state_dict(),
            "decay": self.decay,
        }
