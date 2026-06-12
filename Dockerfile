# syntax=docker/dockerfile:1.7
# NOTE: Docker image build is not verified in CI. If you change this file,
# test locally with: docker build -t tennis-lab-dev .

FROM node:22-bookworm-slim AS node

# Official uv image used as a build-stage source for the uv binary
FROM ghcr.io/astral-sh/uv:latest AS uv-bin

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

COPY --from=node /usr/local /usr/local
COPY --from=uv-bin /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Tell uv to use the system Python 3.11 venv at /workspace/.venv
ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
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
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3.11-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && npm install -g @openai/codex \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy project metadata first so uv can resolve dependencies
COPY pyproject.toml uv.lock README.md LICENSE ./

# Install all dependencies (including dev group) using the frozen lock file.
# uv creates .venv at UV_PROJECT_ENVIRONMENT and installs everything there.
# The project itself (editable) is installed in the second COPY + uv sync pass.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group dev --no-install-project

COPY . .

# Re-run uv sync to install the editable project package itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group dev

ENV PATH="/workspace/.venv/bin:$PATH"

CMD ["bash"]
