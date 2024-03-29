# Base CUDA devel image
FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace/visiomel-2023

# Apt-get installs
RUN \
    apt update && \
    apt-get -y install git unzip wget tmux curl libgl1 libglib2.0-0 openslide-tools

# Install Miniconda
RUN \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh && \
    bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b && \
    rm Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

# Add paths to PATH and PYTHONPATH
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Create conda env
RUN conda init bash
RUN \
    conda create \
    --name visiomel-2023 \
    python=3.10 \
    gcc_linux-64 gxx_linux-64 libgcc cudatoolkit=11.7 \
    -c conda-forge -y
RUN /root/miniconda3/envs/visiomel-2023/bin/python -m pip install --upgrade pip

# Install torch & other related deps: 
# - not in requirements.txt because order of install matters
# - torch first due to some dependencies importing torch during install
RUN \
    /root/miniconda3/envs/visiomel-2023/bin/pip install \
    --default-timeout=1000 \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.0+cu117 \
    torchvision==0.15.1+cu117

# Install requirements
COPY requirements.txt .
RUN /root/miniconda3/envs/visiomel-2023/bin/pip install \
    --default-timeout=1000 \
    -r requirements.txt 

# Add repo folder to PYTHONPATH
ENV PYTHONPATH="/workspace/visiomel-2023:/workspace/visiomel-2023/src:${PYTHONPATH}"
ARG PYTHONPATH="/workspace/visiomel-2023:/workspace/visiomel-2023/src:${PYTHONPATH}"
ENV PATH="/workspace/visiomel-2023:/workspace/visiomel-2023/src:${PATH}"
ARG PATH="/workspace/visiomel-2023:/workspace/visiomel-2023/src:${PATH}"

# Install AWS CLI
RUN \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -fr aws awscliv2.zip

# Add TORCH_HOME for torch hub cache
ENV TORCH_HOME="/workspace/data/.cache/torch"
ARG TORCH_HOME="/workspace/data/.cache/torch"