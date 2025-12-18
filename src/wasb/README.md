# WASB / HRCNet（半自動アノテーション・データセット拡張）

`src/wasb` は、WASB/HRCNet 系のボール検出モデル等を用いた半自動アノテーション、およびテニスデータセット拡張のための実装です。
動画からのフレーム抽出、ボール検出、クリップ分割、ラベル出力までを一連のパイプラインとして扱います。

## ディレクトリ構成（要点）

- `configs/`: Hydra 設定（学習・生成・可視化などのエントリポイント YAML）
- `data/`: データセット/動画入出力、DataModule、サンプリング等
- `models/`: 検出・補完・セグメンテーション等のモデル実装
- `training/`: LightningModule、学習ループ周辺
- `inference/`: 予測器（WASB/HRCNet）、軌道補完など
- `pipeline/`: エンドツーエンドのアノテーションパイプライン（例: `annotation_pipeline.py`）
- `scripts/`: 実行スクリプト（Hydra 前提、学習/生成/可視化が中心）
- `tennis_format.py`: `label.csv` 等の I/O ヘルパー（データセット形式の取り扱い）
- `utils/`: ストリーミングローダ、動画抽出などの補助実装

## 代表的な実行コマンド（Hydra）

データセット生成（入口）:
- `uv run python -m src.wasb.scripts.generate_dataset`

データセット生成（各ステップ）:
- `uv run python -m src.wasb.scripts.generate_dataset.download_videos`
- `uv run python -m src.wasb.scripts.generate_dataset.clip_sampling`
- `uv run python -m src.wasb.scripts.generate_dataset.batch`

学習:
- `uv run python -m src.wasb.scripts.train.ball_detection`
- `uv run python -m src.wasb.scripts.train.trajectory`
- `uv run python -m src.wasb.scripts.train.event_detection`

可視化:
- `uv run python -m src.wasb.scripts.visualize.trajectory`
- `uv run python -m src.wasb.scripts.visualize.ball_video`
- `uv run python -m src.wasb.scripts.visualize.ball_video_ensemble`

## 設定の入口（例）

- ボール検出学習: `src/wasb/configs/train_ball_detection.yaml`（実行時の共通ランタイムは `src/wasb/configs/run/ball_detection.yaml`）
- スクリプトごとの `Config entry point` は、各 `src/wasb/scripts/**.py` の docstring を参照してください。
