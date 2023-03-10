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


Installation
------------

**Build from source**

You can use `python setup.py install` or `pip install .` to build Uni-Core from source. The CUDA version in the build environment should be the same as the one in PyTorch.

You can also use `python setup.py install --disable-cuda-ext` to disalbe the cuda extension operator when cuda is not available.

**Use pre-compiled python wheels**

We also pre-compiled wheels by GitHub Actions. You can download them from the [Release](https://github.com/dptech-corp/Uni-Core/releases). And you should check the pyhon version, PyTorch version and CUDA version. For example, for PyToch 1.12.1, python 3.7, and CUDA 11.3, you can install [unicore-0.0.1+cu113torch1.12.1-cp37-cp37m-linux_x86_64.whl](https://github.com/dptech-corp/Uni-Core/releases/download/0.0.1/unicore-0.0.1+cu113torch1.12.1-cp37-cp37m-linux_x86_64.whl). 

**Docker image**

We also provide the docker image. you can pull it by `docker pull dptechnology/unicore:0.0.1-pytorch1.11.0-cuda11.3`. To use GPUs within docker, you need to [install nvidia-docker-2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) first.


Example
-------

To build a model, you can refer to [example/bert](https://github.com/dptech-corp/Uni-Core/tree/main/examples/bert). 

Related projects
----------------

- [Uni-Mol](https://github.com/dptech-corp/Uni-Mol)
- [Uni-Fold](https://github.com/dptech-corp/Uni-Fold)

Acknowledgement
---------------

The main framework is from [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq).

The fused kernels are from [guolinke/fused_ops](https://github.com/guolinke/fused_ops).

Dockerfile is from [guolinke/pytorch-docker](https://github.com/guolinke/pytorch-docker).

License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/dptech-corp/Uni-Core/blob/main/LICENSE) for additional details.
