from copy import deepcopy
from unicore.optim.fp16_optimizer import seperate_decay_params, flatten_parameters_fp32
import torch


class ExponentialMovingAverageModel:
    def __init__(self, args, model, decay, is_flattened=False):
        self.args = args
        self.model_ema = deepcopy(model)
        self.decay = decay
        self.is_flattened = is_flattened
        if not is_flattened:
            self.name2param = self.get_name2param()
        else:
            self.flatten_params = self.flatten_parameters()

    def get_name2param(self):
        name2param = dict()
        for n, p in self.model_ema.named_parameters():
            name2param[n] = p
            # use float type for ema
            p.data = p.data.float()
            p.grad = None
        return name2param

    def flatten_parameters(self):
        param_group = seperate_decay_params(
            self.args, self.model_ema.named_parameters()
        )
        flatten_group = []
        for param_dict in param_group:
            params = param_dict["params"]
            flatten_param = flatten_parameters_fp32(
                params, set_to_param=True, set_grad=False
            )
            flatten_group.append(flatten_param)
        return flatten_group

    def update_one_param(self, ema_param, new_param):
        diff = ema_param - new_param
        diff *= 1 - self.decay
        ema_param -= diff

    def update(self, new_param):
        if self.is_flattened:
            with torch.no_grad():
                for i in range(len(self.flatten_params)):
                    self.update_one_param(
                        self.flatten_params[i], new_param[i]["params"][0]
                    )
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
