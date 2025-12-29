# syntax=docker/dockerfile:1
# =============================================================================
# GVHMR: third_party/GVHMR 用 Docker イメージ
# Python 3.10 + CUDA 12.1 + PyTorch 2.3 + pytorch3d
# =============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# -----------------------------------------------------------------------------
# 環境変数設定
# -----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_LINK_MODE=copy

# pytorch3d ビルド用
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# -----------------------------------------------------------------------------
# システムパッケージのインストール
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libboost-all-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python 3.10 のインストール
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# uv のインストール
# -----------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# -----------------------------------------------------------------------------
# ワークディレクトリ設定
# -----------------------------------------------------------------------------
WORKDIR /workspace/third_party/GVHMR

# -----------------------------------------------------------------------------
# 依存関係のインストール
# requirements.txt を使用してインストール
# -----------------------------------------------------------------------------
COPY third_party/GVHMR/requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt

# -----------------------------------------------------------------------------
# GVHMR ソースコードのコピーとインストール
# -----------------------------------------------------------------------------
COPY third_party/GVHMR/ ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e .

# -----------------------------------------------------------------------------
# デフォルトコマンド
# -----------------------------------------------------------------------------
CMD ["bash"]
