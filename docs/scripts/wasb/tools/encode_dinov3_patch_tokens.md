# WASB tools/encode_dinov3_patch_tokens

テニスフレームをDINOv3パッチトークンにエンコードし、ディスクに保存するスクリプト。

## 概要

このスクリプトは、学習済みDINOv3バックボーンを使用して、テニスデータセットの全フレームをパッチトークン（ViT中間表現）にエンコードし、事前計算された埋め込みとして保存します。これにより、学習時のバックボーン計算を省略し、学習を高速化できます。

## コマンド例

```bash
# 基本的な使用法
uv run python -m src.wasb.scripts.tools.encode_dinov3_patch_tokens \
  model_checkpoint=outputs/wasb/dinov3_heatmap/logs/version_0/checkpoints/last.ckpt \
  output_dir=data/tennis/patch_embeddings

# 拡張パスを増やす
uv run python -m src.wasb.scripts.tools.encode_dinov3_patch_tokens \
  model_checkpoint=... \
  num_augments=5

# 特定のマッチのみ処理
uv run python -m src.wasb.scripts.tools.encode_dinov3_patch_tokens \
  model_checkpoint=... \
  matches=[game1,game2]

# 既存ファイルを上書き
uv run python -m src.wasb.scripts.tools.encode_dinov3_patch_tokens \
  model_checkpoint=... \
  overwrite=true
```

## コンフィグ

エントリポイント: `src/wasb/configs/encode_dinov3_tokens.yaml`

### 主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `device` | `cuda` | デバイス |
| `model_checkpoint` | (DINOv3 heatmap) | DINOv3モデルのチェックポイント |
| `output_dir` | `data/tennis/patch_embeddings` | 出力ディレクトリ |
| `num_augments` | `3` | 拡張パス数（0=拡張なし） |
| `heatmap_sigma` | `${data.heatmap_sigma}` | ヒートマップのシグマ |
| `heatmap_hw` | `${data.heatmap_hw}` | ヒートマップサイズ |
| `save_embeddings` | `true` | 埋め込みを保存するか |
| `save_heatmaps` | `true` | ターゲットヒートマップを保存するか |
| `batch_size` | `32` | バッチサイズ |
| `num_workers` | `4` | ワーカー数 |
| `pin_memory` | `true` | ピンメモリ |
| `overwrite` | `true` | 既存ファイルを上書き |
| `save_dtype` | `float32` | 保存時のデータ型 |
| `matches` | `null` | 処理するマッチ（null=全マッチ） |

### preprocess (前処理設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `resize_hw` | `${data.resize_hw}` | リサイズ後のサイズ |
| `normalize` | `false` | 正規化を適用するか |
| `mean` | `[0.485, 0.456, 0.406]` | 正規化の平均 |
| `std` | `[0.229, 0.224, 0.225]` | 正規化の標準偏差 |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   encode_dinov3_patch_tokens.py                              │
│                                                                              │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────────┐  │
│  │  Tennis Dataset │──────▶│   DINOv3 ViT    │──────▶│   Numpy Arrays      │  │
│  │                 │      │   (backbone)    │      │                     │  │
│  │ - game1/Clip1/  │      │                 │      │ - Clip1.npy         │  │
│  │   frame_*.jpg   │      │ - patch_embed   │      │ - Clip1_aug01.npy   │  │
│  │ - Label.csv     │      │ - transformer   │      │ - Clip1_heatmaps.npy│  │
│  │                 │      │ → patch tokens  │      │                     │  │
│  └─────────────────┘      └─────────────────┘      └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

処理フロー:
1. 各マッチ/クリップのフレームを読み込み
2. リサイズ、（オプション）正規化を適用
3. DINOv3 バックボーンでパッチトークンを抽出
4. 拡張ありの場合、複数パスで処理
5. パッチトークンを .npy ファイルとして保存
6. ターゲットヒートマップも保存（save_heatmaps=true）
```

## DINOv3 パッチトークン

```
入力画像: [B, 3, H, W]  例: [32, 3, 288, 512]
                │
                ▼
        ┌───────────────┐
        │  Patch Embed  │
        │  16x16 patch  │
        └───────────────┘
                │
                ▼ [B, N, D]  例: [32, 576, 384]
                │            N = (H/16) × (W/16) = 18 × 32 = 576
                │            D = 384 (ViT-S)
        ┌───────────────┐
        │  Transformer  │
        │    Blocks     │
        └───────────────┘
                │
                ▼
        パッチトークン: [B, N, D]
```

## 出力構造

```
data/tennis/patch_embeddings/
├── game1/
│   ├── Clip1.npy              # パッチトークン [T, N, D]
│   ├── Clip1_aug01.npy        # 拡張パス1
│   ├── Clip1_aug02.npy        # 拡張パス2
│   ├── Clip1_heatmaps.npy     # ターゲットヒートマップ [T, H', W']
│   ├── Clip2.npy
│   └── ...
├── game2/
│   └── ...
└── ...
```

## ファイル形式

### パッチトークン (.npy)

```python
tokens = np.load("Clip1.npy")  # shape: [T, N, D]
# T: フレーム数
# N: パッチ数（例: 576）
# D: 埋め込み次元（例: 384 for ViT-S）
```

### ヒートマップ (.npy)

```python
heatmaps = np.load("Clip1_heatmaps.npy")  # shape: [T, H', W']
# T: フレーム数
# H', W': ヒートマップサイズ（例: 288, 512）
```

## 使用例

```bash
# 1. パッチトークンをエンコード
uv run python -m src.wasb.scripts.tools.encode_dinov3_patch_tokens \
  model_checkpoint=outputs/wasb/dinov3_heatmap/logs/version_0/checkpoints/last.ckpt \
  output_dir=data/tennis/patch_embeddings \
  num_augments=3

# 2. 事前計算された埋め込みで学習
uv run python -m src.wasb.scripts.train.ball_detection \
  data=patch_embeddings \
  data.embeddings_dir=data/tennis/patch_embeddings
```

## パフォーマンス

- **エンコード時間**: GPU使用で1クリップあたり数秒
- **ディスク容量**: float32で約1-2GB/マッチ（拡張パス数による）
- **学習高速化**: バックボーン計算を省略することで約2-5倍高速化

## 関連モジュール

- `src.wasb.models.dinov3_fpn_heatmap.DinoV3FPNHeatmap`: DINOv3モデル
- `src.wasb.data.patch_embeddings_datamodule`: パッチ埋め込みデータモジュール
