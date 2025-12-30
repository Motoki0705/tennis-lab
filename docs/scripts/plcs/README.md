# PLCS Scripts

Player Location from Court Skeleton (PLCS) タスクのスクリプト群。

2Dキーポイント（人物・コート）から3Dプレーヤー位置と回転を推定します。

## スクリプト一覧

| スクリプト | 説明 | ドキュメント |
|-----------|------|-------------|
| `generate_dataset` | 合成トレーニングデータの生成 | [generate_dataset.md](generate_dataset.md) |
| `train` | フレーム単位モデルの学習 | [train.md](train.md) |
| `train_sequence` | シーケンスモデルの学習 | [train_sequence.md](train_sequence.md) |
| `train_multiview` | マルチビューシーケンスモデルの学習 | [train_multiview.md](train_multiview.md) |
| `visualize` | シーン・予測結果の可視化 | [visualize.md](visualize.md) |
| `visualize_multiview` | マルチビュー予測結果の可視化 | [visualize_multiview.md](visualize_multiview.md) |

## 典型的なワークフロー

```bash
# 1. データセット生成
uv run python -m src.plcs.scripts.generate_dataset \
  run.output_dir=data/plcs \
  simulation.num_scenes=3000

# 2. モデル学習（フレーム単位）
uv run python -m src.plcs.scripts.train

# 3. または シーケンスモデル学習
uv run python -m src.plcs.scripts.train_sequence

# 4. または マルチビューシーケンスモデル学習
uv run python -m src.plcs.scripts.train_multiview

# 5. 結果の可視化
uv run python -m src.plcs.scripts.visualize \
  visualization.mode=predict \
  visualization.checkpoint=outputs/plcs/checkpoints/last.ckpt

# 6. マルチビュー結果の可視化
uv run python -m src.plcs.scripts.visualize_multiview \
  visualization.mode=predict \
  visualization.checkpoint=outputs/plcs_multiview/checkpoints/last.ckpt
```

## ディレクトリ構成

```
src/plcs/
├── scripts/
│   ├── generate_dataset.py
│   ├── train.py
│   ├── train_sequence.py
│   ├── train_multiview.py
│   ├── visualize.py
│   └── visualize_multiview.py
├── configs/
│   ├── generate_dataset.yaml
│   ├── train.yaml
│   ├── train_sequence.yaml
│   ├── train_multiview.yaml
│   ├── visualize.yaml
│   ├── visualize_multiview.yaml
│   └── loss/
│       ├── frame.yaml
│       ├── sequence.yaml
│       └── multiview_sequence.yaml
├── data/
├── models/
├── training/
└── generate_dataset/
```
