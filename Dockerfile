# syntax=docker/dockerfile:1.7

FROM node:22-bookworm-slim AS node

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

COPY --from=node /usr/local /usr/local

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        gh \
        git \
        libavcodec-dev \
        libavformat-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libswscale-dev \
        libxext6 \
        libxrender-dev \
        openssh-client \
        software-properties-common \
        wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pip \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3.11-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && npm install -g @openai/codex \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt requirements-dev.txt pyproject.toml README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m pip install -r requirements-dev.txt

COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m pip install --no-deps -e .

CMD ["bash"]
