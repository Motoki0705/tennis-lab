# syntax=docker/dockerfile:1
# =============================================================================
# tennis-lab: メインプロジェクト (src/) 用 Docker イメージ
# Python 3.11 + CUDA 12.4 + PyTorch環境
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# -----------------------------------------------------------------------------
# 環境変数設定
# -----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_LINK_MODE=copy

# -----------------------------------------------------------------------------
# システムパッケージのインストール
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python 3.11 のインストール
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# uv のインストール
# -----------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# -----------------------------------------------------------------------------
# ワークディレクトリ設定
# -----------------------------------------------------------------------------
WORKDIR /workspace

# -----------------------------------------------------------------------------
# 依存関係のインストール（キャッシュ活用のためpyproject.tomlを先にコピー）
# -----------------------------------------------------------------------------
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# -----------------------------------------------------------------------------
# ソースコードのコピー
# -----------------------------------------------------------------------------
COPY . .

# -----------------------------------------------------------------------------
# デフォルトコマンド
# -----------------------------------------------------------------------------
CMD ["bash"]
