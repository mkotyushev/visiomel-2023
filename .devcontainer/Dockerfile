# Base CUDA devel image
FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

# Set working directory
RUN mkdir /workspaces/visiomel-2023
WORKDIR /workspaces/visiomel-2023

# Apt-get installs
RUN \
    apt update && \
    apt-get -y install git unzip wget tmux

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
ENV PYTHONPATH="/workspaces/visiomel-2023:${PYTHONPATH}"
ARG PYTHONPATH="/workspaces/visiomel-2023:${PYTHONPATH}"