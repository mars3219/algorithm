ARG UBUNTU_VERSION=ubuntu20.04
ARG CUDA_VERSION=12.1.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-${UBUNTU_VERSION}

LABEL maintainer="H. S. Nam @ Gaonplatform Product2 <hsnam@gaonpf.com>"


ENV TRT_VERSION 8.6.1.6
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

ENV PATH=/opt/conda/bin:/usr/local/cuda:/usr/local/cuda-12.1/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV FORCE_CUDA="1"
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7                            

RUN apt-get update && \
    apt-get --no-install-recommends install -y gcc git wget curl zip htop tar vim openssl software-properties-common \
    python3 python3-pip python-is-python3 pip python3-distutils python3-typing-extensions build-essential python3-dev python3-wheel\
    libcurl4-openssl-dev libgl1 libglib2.0-0 libgl1-mesa-glx libpython3-dev gnupg libsm6 libxext6 libgtk2.0-dev pkg-config libssl-dev \
    pbzip2 pv bzip2 unzip devscripts lintian fakeroot dh-make && \
    cd /usr/local/bin && \
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /workspace
