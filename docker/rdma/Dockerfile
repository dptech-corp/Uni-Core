FROM nvcr.io/nvidia/pytorch:24.03-py3

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
        nfs-common \
        bc


# # ==================================================================
# # pytorch
# # ------------------------------------------------------------------
ENV TORCH_CUDA_ARCH_LIST "7.0;7.5;8.0;9.0"

RUN pip3 install --no-cache-dir --upgrade sentry-sdk requests ninja typing packaging && rm -rf ~/.cache/pip

RUN cd /tmp && \
    git clone https://github.com/dptech-corp/Uni-Core && \
    cd Uni-Core && \
    python setup.py install && \
    rm -rf  /tmp/* && rm -rf ~/.cache/pip

RUN pip3 install --no-cache-dir tokenizers lmdb biopython ml-collections timeout-decorator urllib3 tree dm-tree && rm -rf ~/.cache/pip 

RUN MAX_JOBS=1 pip3 install -U 'flash-attn==2.6.1' --no-build-isolation --no-cache-dir

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/*
