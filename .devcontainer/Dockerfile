FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
WORKDIR ~

ARG DEBIAN_FRONTEND=noninteractive

RUN sudo apt-get update && \
    sudo apt-get -y upgrade && \
    sudo apt-get install -y wget python3 python3-pip && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh && \
    bash Miniconda3-py310_22.11.1-1-Linux-x86_64.sh -b && \
    rm Miniconda3-py310_22.11.1-1-Linux-x86_64.sh && \
    ~/miniconda3/bin/conda init bash
RUN ~/miniconda3/bin/conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 fenics-dolfinx mpich pyvista scikit-learn matplotlib && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/ && \
    python3 -m pip install tensorflow keras_tuner pandas sphinx sphinx-rtd-theme
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/" >> ~/.bashrc
RUN sudo apt-get install ffmpeg libsm6 libxext6  -y
RUN sudo apt install libgl1-mesa-glx xvfb -y
ENV SHELL /bin/bash
