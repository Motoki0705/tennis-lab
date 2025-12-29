# WASB tools/extract_dinov3_backbone

学習済みWASBチェックポイントからDINOv3バックボーン重みを抽出するスクリプト。

## 概要

このスクリプトは、DINOv3ベースのWASBモデル（Lightning チェックポイント）からバックボーン部分の重みだけを抽出し、別ファイルとして保存します。転移学習やバックボーンの再利用に使用します。

## コマンド例

```bash
# 基本的な使用法
uv run python -m src.wasb.scripts.tools.extract_dinov3_backbone \
  checkpoint_path=outputs/wasb/dinov3_heatmap/logs/version_0/checkpoints/last.ckpt \
  output_path=outputs/wasb/dinov3_backbone.pth

# 別の出力パスを指定
uv run python -m src.wasb.scripts.tools.extract_dinov3_backbone \
  checkpoint_path=path/to/checkpoint.ckpt \
  output_path=path/to/backbone.pth
```

## コンフィグ

エントリポイント: `src/wasb/configs/extract_dinov3_backbone.yaml`

### 主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `checkpoint_path` | `outputs/wasb/dinov3_heatmap/logs/version_0/checkpoints/last.ckpt` | 入力チェックポイント |
| `output_path` | `outputs/wasb/dinov3_backbone.pth` | 出力ファイルパス |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     extract_dinov3_backbone.py                               │
│                                                                              │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────────┐  │
│  │   Checkpoint    │──────▶│  Extract Keys   │──────▶│   Save Backbone     │  │
│  │   (.ckpt)       │      │                 │      │   (.pth)            │  │
│  │                 │      │ model.backbone. │      │                     │  │
│  │ - state_dict    │      │ → 接頭辞を除去  │      │ - backbone 重みのみ │  │
│  └─────────────────┘      └─────────────────┘      └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

処理フロー:
1. チェックポイントを読み込み
2. state_dict から "model.backbone." または "backbone." で始まるキーを抽出
3. 接頭辞を除去してバックボーン用の state_dict を作成
4. 新しいファイルとして保存
```

## 抽出されるキー

入力チェックポイントのキー例:
```
state_dict = {
    "model.backbone.patch_embed.proj.weight": ...,
    "model.backbone.patch_embed.proj.bias": ...,
    "model.backbone.blocks.0.attn.qkv.weight": ...,
    "model.backbone.blocks.0.attn.proj.weight": ...,
    ...
    "model.fpn.lateral_convs.0.weight": ...,  # 除外
    "model.head.weight": ...,                  # 除外
}

# 抽出後
backbone_state = {
    "patch_embed.proj.weight": ...,
    "patch_embed.proj.bias": ...,
    "blocks.0.attn.qkv.weight": ...,
    "blocks.0.attn.proj.weight": ...,
    ...
}
```

## 使用例

```bash
# 1. DINOv3 ヒートマップモデルを学習
uv run python -m src.wasb.scripts.train.ball_detection model=dinov3_heatmap

# 2. バックボーン重みを抽出
uv run python -m src.wasb.scripts.tools.extract_dinov3_backbone \
  checkpoint_path=outputs/wasb/dinov3_heatmap/logs/version_0/checkpoints/last.ckpt \
  output_path=outputs/wasb/finetuned_dinov3_backbone.pth

# 3. 別のモデルで再利用
# model.load_backbone_checkpoint("outputs/wasb/finetuned_dinov3_backbone.pth")
```

## 出力ファイル

```
outputs/wasb/
└── dinov3_backbone.pth    # バックボーン重みのみ
```

## エラー処理

- チェックポイントが見つからない場合: `FileNotFoundError`
- バックボーンキーが見つからない場合: `KeyError`

## 関連モジュール

- `src.wasb.models.dinov3_fpn_heatmap`: DINOv3+FPN モデル
