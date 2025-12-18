# BLCS (Ball Localization in Court System)

BLCS は、テニスコート座標系におけるボールの 3D 軌道を、2D のボール観測（画像座標）とコート情報から推定するためのタスク実装です。

## 目的 / 想定入出力

- 入力: 2D ボール観測（例: 各カメラのヒートマップ/検出点）＋コート幾何
- 出力: コート座標系の 3D 位置（軌道）や、可視化・評価指標

## ディレクトリ構成（要点）

- `configs/`: Hydra 設定（学習・生成・可視化のエントリポイント YAML を含む）
- `data/`: データセット I/O、DataModule、シーン生成（カメラ/投影など）
- `simulation/`: 物理モデル（ボール運動）とショット/シーンのシミュレーション
- `models/`: 推定モデル実装（例: `BLCSModel`）
- `training/`: LightningModule、損失、メトリクス
- `inference/`: 推論・可視化向けの Predictor / 補助ロジック
- `scripts/`: 実行スクリプト（Hydra 前提）
- `utils/`: BLCS 固有ユーティリティ（共通ロジックは `src/utils` を使用）

## 代表的な実行コマンド（Hydra）

データ生成:
- `uv run python -m src.blcs.scripts.generate_dataset`
- 例: `uv run python -m src.blcs.scripts.generate_dataset run.output_dir=data/blcs sampling.per_from_cell_samples=10`

学習:
- `uv run python -m src.blcs.scripts.train`
- 例: `uv run python -m src.blcs.scripts.train training.max_epochs=5 run.gpus=0`

可視化（必要に応じて予測も実行）:
- `uv run python -m src.blcs.scripts.visualize`
- 例: `uv run python -m src.blcs.scripts.visualize visualization.scene_path=data/blcs/scenes/scene_000000.npz visualization.info=true`
- 例: `uv run python -m src.blcs.scripts.visualize visualization.mode=predict visualization.checkpoint=outputs/blcs/checkpoints/last.ckpt`

## 設定の入口

- 学習: `src/blcs/configs/train.yaml`
- データ生成: `src/blcs/configs/generate_dataset.yaml`
- 可視化: `src/blcs/configs/visualize.yaml`
