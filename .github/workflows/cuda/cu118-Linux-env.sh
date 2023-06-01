#!/bin/bash

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=${CUDA_HOME}/bin:/usr/local/nvidia/bin:/usr/local/nvidia/lib64:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_INSTALL_DIR=$CUDA_HOME