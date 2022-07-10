Uni-Core, an efficient distributed PyTorch framework
====================================================

Uni-Core is built for rapidly creating PyTorch models with high performance, especially for Transfromer-based models. It supports the following features:
- Distributed training over multi-GPUs and multi-nodes
- Mixed-precision training with fp16 and bf16
- High-performance fused CUDA kernels
- model checkpoint management
- Friendly logging
- Buffered (GPU-CPU overlapping) data loader
- Gradient accumulation
- Commonly used optimizers and LR schedulers
- Easy to create new models



To install:
```python
python setup.py install
```
We recommend to use [docker](https://github.com/dptech-corp/Uni-Core/blob/main/docker/Dockerfile) for installation.


To build a model, you can refer to [example/bert](https://github.com/dptech-corp/Uni-Core/tree/main/examples/bert). 

Related projects
----------------

- [Uni-Mol](https://github.com/dptech-corp/Uni-Mol)

Acknowledgement
---------------

The main framework is from [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq).

The fused kernels are from [guolinke/fused_ops](https://github.com/guolinke/fused_ops).

Dockerfile is from [guolinke/pytorch-docker](https://github.com/guolinke/pytorch-docker).

License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/dptech-corp/Uni-Core/blob/main/LICENSE) for additional details.
