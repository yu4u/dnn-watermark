FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN apt update \
  && apt install -y \
    python3.5 \
    python3.5-dev \
    python3-pip \
    libgl1-mesa-dev \
    libssl-dev \
    libbz2-dev \
    liblzma-dev \
    libreadline-dev \
    libsqlite3-dev \
    zlib1g-dev \
    libjpeg-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
  && apt clean \
  && rm -rf /var/lib/apt/lists/*

RUN ln -f -s "$(which python3.5)" /usr/bin/python

RUN python -m pip install -U --no-cache-dir pip==20.3.4 \
  && pip install -U --no-cache-dir \
    keras==1.1.2 \
    tensorflow==0.12.1 \
    tensorflow-gpu==0.12.1 \
    pandas \
    matplotlib \
    scikit-learn
