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

# Install NVSHMEM for DEEPEP
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update 

RUN apt-get install libnvshmem3-cuda-12 libnvshmem3-dev-cuda-12
ENV NVSHMEM_DIR=/usr/local/nvshmem

# Clone & install DeepEP
RUN git clone https://github.com/deepseek-ai/DeepEP.git /opt/DeepEP && \
    cd /opt/DeepEP && \
    python setup.py install && \
    cd / && rm -rf /opt/DeepEP


# Install PyTorch, Torchvision, Torchaudio with CUDA 12.8 support, and Transformers
RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.* torchvision torchaudio && \
python3.11 -m pip install transformers==4.34.0

# Copy and install Python requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set Python3.11 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create a workspace directory (for persistent volume mount) and set it as working dir
RUN mkdir /workspace
WORKDIR /workspace
COPY . /workspace

# Install torchtitan in editable mode
RUN pip install -e torchtitan

# Default command to keep container alive (so we can exec into it or run commands)
CMD ["sleep", "infinity"]



