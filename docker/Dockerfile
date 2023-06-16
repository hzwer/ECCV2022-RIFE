ARG CUDA_VER=11.1.1
FROM nvidia/cuda:${CUDA_VER}-cudnn8-runtime-ubuntu18.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
 && apt-get -y upgrade \
 && apt-get install -y --no-install-recommends \
    bash \
    ffmpeg \
    python3 \
    python3-pip \
    libsm6 \
    libxext6 \
 && apt-get -y autoremove \
 && apt-get -y clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip \
 && pip3 install --no-cache-dir \
    opencv-python==4.6.0.66 \
    sk-video==1.1.10 \
    setuptools==59.6.0 \
    tqdm==4.64.1 \
 && pip3 install --no-cache-dir \
    torch==1.8.0+cu111 \
    torchvision==0.9.0+cu111 \
    torchaudio==0.8.0 \
    -f https://download.pytorch.org/whl/torch_stable.html \
 && pip3 install --no-cache-dir \
    moviepy==1.0.3

WORKDIR /app
