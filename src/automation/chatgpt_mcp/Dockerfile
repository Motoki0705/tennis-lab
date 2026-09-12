FROM nvidia/cuda:13.0.0-base-ubuntu24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libxcb1 libgl1 libglib2.0-0t64 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*
