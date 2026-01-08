# BLCS Scripts

Ball Location from Court Skeleton (BLCS) タスクのスクリプト群。

2Dボール位置シーケンスとコートキーポイントから3Dボール軌道を推定します。

## スクリプト一覧

| スクリプト | 説明 | ドキュメント |
|-----------|------|-------------|
| `generate_dataset` | 物理シミュレーションによるデータ生成 | [generate_dataset.md](generate_dataset.md) |
| `train` | 軌道推定モデルの学習 | [train.md](train.md) |
| `visualize` | シーン・予測結果の可視化 | [visualize.md](visualize.md) |

## 典型的なワークフロー

```bash
# 1. データセット生成（物理シミュレーション）
uv run python -m src.blcs.scripts.generate_dataset \
  run.output_dir=data/blcs \
  sampling.per_from_cell_samples=100

# 2. モデル学習
uv run python -m src.blcs.scripts.train

# 3. 結果の可視化
uv run python -m src.blcs.scripts.visualize \
  visualization.mode=predict \
  visualization.checkpoint=outputs/blcs/single/logs/version_0/checkpoints/last.ckpt
```

## ディレクトリ構成

```
src/blcs/
├── scripts/
│   ├── generate_dataset.py
│   ├── train.py
│   └── visualize.py
├── configs/
│   ├── generate_dataset.yaml
│   ├── train.yaml
│   └── visualize.yaml
├── data/
├── models/
├── training/
├── simulation/           # 物理シミュレーション
└── generate_dataset/
```
