# PLCS (Player Localization in Court System)

PLCS は、テニスコート座標系におけるプレイヤーの位置・向き（姿勢）を、2D の姿勢観測（例: 2D pose）から推定するためのタスク実装です。

## 目的 / 想定入出力

- 入力: 2D pose（関節座標など）＋カメラ/コート情報
- 出力: コート座標系でのプレイヤー位置・回転、評価指標、可視化

## ディレクトリ構成（要点）

- `configs/`: Hydra 設定（フレーム/シーケンス学習、生成、可視化の設定を含む）
- `data/`: データセット I/O、シーン生成、DataModule（frame/sequence）
- `models/`: 推定モデル（frame / sequence）
- `training/`: LightningModule、損失、メトリクス（frame/sequence）
- `inference/`: 推論・可視化向けの Predictor
- `scripts/`: 実行スクリプト（Hydra 前提）
- `utils/`: PLCS 固有ユーティリティ（共通ロジックは `src/utils` を使用）

## 代表的な実行コマンド（Hydra）

データ生成:
- `uv run python -m src.plcs.scripts.generate_dataset`

学習（frame / sequence）:
- `uv run python -m src.plcs.scripts.train`
- `uv run python -m src.plcs.scripts.train_sequence`
- 例: `uv run python -m src.plcs.scripts.train run.gpus=0 training.max_epochs=1`

可視化:
- `uv run python -m src.plcs.scripts.visualize`
- 例: `uv run python -m src.plcs.scripts.visualize visualization.scene_path=data/plcs/scenes/scene_000000.npz visualization.info=true`

## 設定の入口

- 学習（frame）: `src/plcs/configs/train.yaml`
- 学習（sequence）: `src/plcs/configs/train_sequence.yaml`
- 可視化: `src/plcs/configs/visualize.yaml`
