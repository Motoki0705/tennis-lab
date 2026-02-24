# Docker 環境

このディレクトリには tennis-lab プロジェクトの Docker 環境設定が含まれています。

## 概要

tennis-lab は以下の2つの独立した実行環境で構成されています：

| サービス | 説明 | Python | CUDA |
|----------|------|--------|------|
| `tennis-lab` | メインプロジェクト (`src/`) | 3.11 | 12.4 |
| `gvhmr` | GVHMR (`third_party/GVHMR/`) | 3.10 | 12.1 |

## 前提条件

- Docker Engine 20.10+
- Docker Compose v2.0+
- NVIDIA Container Toolkit
- NVIDIA GPU (CUDA 対応)

### NVIDIA Container Toolkit のインストール

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## クイックスタート

### 1. イメージのビルド

```bash
cd docker
docker compose build
```

### 2. コンテナの起動

```bash
# 両方のコンテナを起動
docker compose up -d

# tennis-lab のみ起動
docker compose up -d tennis-lab

# gvhmr のみ起動
docker compose up -d gvhmr
```

### 3. コンテナへのアクセス

```bash
# tennis-lab コンテナに入る
docker compose exec tennis-lab bash

# gvhmr コンテナに入る
docker compose exec gvhmr bash
```

### 4. コンテナの停止

```bash
docker compose down
```

## 使用例

### tennis-lab での学習実行

```bash
# コンテナ内で
docker compose exec tennis-lab bash

# 学習スクリプトの実行
uv run python -m src.tasks.wasb.scripts.train.ball_detection
```

### GVHMR での推論実行

```bash
# コンテナ内で
docker compose exec gvhmr bash

# デモの実行
bash run_demo.sh
```

## ディレクトリ構成

```
docker/
├── docker-compose.yml    # Docker Compose 設定ファイル
├── tennis-lab/
│   └── Dockerfile        # tennis-lab 用 Dockerfile
└── gvhmr/
    └── Dockerfile        # GVHMR 用 Dockerfile
```

## ボリュームマウント

### tennis-lab

| ホスト | コンテナ | 説明 |
|--------|----------|------|
| プロジェクトルート | `/workspace` | ソースコード |
| `data/` | `/workspace/data` | データセット |
| `outputs/` | `/workspace/outputs` | 出力ファイル |

### gvhmr

| ホスト | コンテナ | 説明 |
|--------|----------|------|
| `third_party/GVHMR/` | `/workspace/third_party/GVHMR` | GVHMR ソースコード |
| `third_party/GVHMR/inputs/` | `/workspace/third_party/GVHMR/inputs` | 入力ファイル |
| `third_party/GVHMR/outputs/` | `/workspace/third_party/GVHMR/outputs` | 出力ファイル |
| `data/` | `/workspace/data` | 共有データセット |

## GPU 設定

デフォルトでは全ての GPU が使用可能です。特定の GPU のみを使用する場合：

```bash
# 特定の GPU のみ使用
NVIDIA_VISIBLE_DEVICES=0 docker compose up -d tennis-lab

# 複数の GPU を指定
NVIDIA_VISIBLE_DEVICES=0,1 docker compose up -d tennis-lab
```

## トラブルシューティング

### GPU が認識されない

```bash
# NVIDIA Container Toolkit の確認
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### ビルドエラー

```bash
# キャッシュを無効にしてビルド
docker compose build --no-cache
```

### 権限エラー

ホストとコンテナ間でファイルの権限問題が発生する場合：

```bash
# ホストのUID/GIDでコンテナを実行
docker compose run --user $(id -u):$(id -g) tennis-lab bash
```
