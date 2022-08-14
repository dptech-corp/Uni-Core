FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV LANG C.UTF-8
ENV OFED_VERSION=5.3-1.0.0.1

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        cmake \
        net-tools \
        sudo \
        autotools-dev \
        rsync \
        jq \
        openssh-server \
        tmux \
        screen \
        htop \
        pdsh \
        openssh-client \
        lshw \
        dmidecode \
        util-linux \
        automake \
        autoconf \
        libtool \
        net-tools \
        pciutils \
        libpci-dev \
        libaio-dev \
        libcap2 \
        libtinfo5 \
        fakeroot \
        devscripts \
        debhelper \
        nfs-common

# ==================================================================
# InfiniBand & RDMA
# ------------------------------------------------------------------

RUN cd /tmp && \
    wget -q http://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64.tgz && \
    tar xzf MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64.tgz && \
    MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64/mlnxofedinstall --user-space-only --without-fw-update --force --all && \
    rm -rf /tmp/MLNX_OFED_LINUX-${OFED_VERSION}*


RUN cd /tmp && \
    mkdir -p /usr/local/nccl-rdma-sharp-plugins && \
    DEBIAN_FRONTEND=noninteractive apt install -y zlib1g-dev && \
    git clone --depth=1 https://github.com/Mellanox/nccl-rdma-sharp-plugins.git && \
    cd nccl-rdma-sharp-plugins && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local/nccl-rdma-sharp-plugins --with-cuda=/usr/local/cuda && \
    make && \
    make install

# ==================================================================
# python
# ------------------------------------------------------------------

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
ENV PATH /usr/local/nvidia/bin:/usr/local/nvidia/lib64:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PYTHON_VERSION=3.8
RUN wget -O ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

RUN conda install -y python=3.8  && conda clean -ya

RUN conda install -y scipy scikit-learn pyyaml tensorboard tensorboardX && \
    conda clean -ya

RUN ldconfig

# ==================================================================
# pytorch
# ------------------------------------------------------------------
ENV TORCH_CUDA_ARCH_LIST "7.0;7.5;8.0"

RUN conda install -y numpy pyyaml scipy ipython mkl mkl-include ninja cython typing pandas && \
    conda clean -ya

RUN conda install pytorch=1.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge && \
    conda clean -ya

RUN cd /tmp && \
    git clone https://github.com/dptech-corp/Uni-Core && \
    cd Uni-Core && \
    python setup.py install &&\
    rm -rf  /tmp/*

RUN pip install --no-cache-dir tokenizers lmdb biopython ml-collections timeout-decorator urllib3 tree dm-tree

ENV LD_LIBRARY_PATH=/usr/local/nccl-rdma-sharp-plugins/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/mpi/gcc/openmpi-4.1.0rc5/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/mpi/gcc/openmpi-4.1.0rc5/lib:$LD_LIBRARY_PATH

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* && \
    conda clean -ya