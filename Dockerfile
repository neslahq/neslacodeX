# Use NVIDIA's official CUDA 12.8 base image with cuDNN on Ubuntu 22.04
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Set non-interactive mode for apt and install basic dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
software-properties-common build-essential wget curl git ca-certificates && \
add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
apt-get install -y --no-install-recommends python3.11 python3.11-dev python3.11-distutils && \
apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.11 get-pip.py && rm get-pip.py

# Install PyTorch, Torchvision, Torchaudio with CUDA 12.8 support, and Transformers
RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.* torchvision torchaudio && \
python3.11 -m pip install transformers==4.34.0

# Install NVSHMEM for DEEPEP
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install nvshmem-cuda-12

# Install GDRCopy userspace only (no kernel module)
RUN git clone https://github.com/NVIDIA/gdrcopy.git /opt/gdrcopy && \
    cd /opt/gdrcopy && \
    make lib CUDA=/usr/local/cuda && \
    make PREFIX=/usr/local/gdrcopy install && \
    ldconfig && \
    rm -rf /opt/gdrcopy

ENV NVSHMEM_DIR=/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem
ENV LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
ENV PATH="${NVSHMEM_DIR}/bin:$PATH"

RUN apt-get update && apt-get install -y ninja-build build-essential 

RUN apt-get update && apt-get install -y \
    rdma-core \
    libibverbs-dev \
    libmlx5-1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone & install DeepEP
RUN git clone https://github.com/deepseek-ai/DeepEP.git /opt/DeepEP && \
    cd /opt/DeepEP && \
    python3.11 setup.py install && \
    cd / && rm -rf /opt/DeepEP

# Install PyTorch, Torchvision, Torchaudio with CUDA 12.8 support, and Transformers
RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.* torchvision torchaudio torchao && \
python3.11 -m pip install transformers==4.34.0

# Copy and install Python requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set Python3.11 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Default command to keep container alive (so we can exec into it or run commands)
CMD ["sleep", "infinity"]



