import torch as th

class Linear(th.nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
    ):
        """ Follows convention from AlphaFold 2. Promotes clean-path in resblocks.
        See section 1.11.4 Parameters initialization in supplementary information
        https://www.nature.com/articles/s41586-021-03819-2#Sec20
        """
        super(Linear, self).__init__(d_in, d_out, bias=bias)

        self.use_bias = bias

        if self.use_bias:
            with th.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "final":
            self._zero_init(False)
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        th.nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _glorot_uniform_init(self):
        th.nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True):
        with th.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with th.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self):
        th.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")


class Embedding(th.nn.Embedding):
    def __init__(
        self,
        *args,
        init: str = "default",
        **kwargs,
    ):
        """ Follows the SmallInitEmb trick from RWKV appendix F.
        https://aclanthology.org/2023.findings-emnlp.936.pdf also see
        https://github.com/BlinkDL/SmallInitEmb
        """
        super().__init__(*args, **kwargs)

        if init == "default":
            self._normal_init()
        elif init == "small":
            self._small_init(alpha=1e-4)
        else:
            raise ValueError("Invalid init method.")

    def _normal_init(self):
        th.nn.init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _small_init(self, alpha: float):
        th.nn.init.uniform_(self.weight, a=-alpha, b=alpha)
        self._fill_padding_idx_with_zero()
