# WASB visualize/save_one_sample_visuals

データセットの単一サンプルを画像として保存するデバッグ・確認用スクリプト。

## 概要

このスクリプトは、`TennisDataModule` から単一サンプルを取得し、元フレーム、拡張後フレーム、ヒートマップオーバーレイを画像として保存します。データローダーの動作確認やデバッグに使用します。

## コマンド例

```bash
# 基本的な使用法
uv run --no-sync python -m src.tasks.wasb.scripts.visualize.save_one_sample_visuals \
  data.root_dir=data/tennis \
  data.train_matches=[game1] \
  sample_index=0

# 複数サンプルを保存
uv run python -m src.tasks.wasb.scripts.visualize.save_one_sample_visuals \
  num_samples=10 \
  sample_indices=[0,10,20,30]

# 分割とオーバーレイ透明度を指定
uv run python -m src.tasks.wasb.scripts.visualize.save_one_sample_visuals \
  split=val \
  overlay_alpha=0.7

# ターゲットインデックスを指定
uv run python -m src.tasks.wasb.scripts.visualize.save_one_sample_visuals \
  target_index=0
```

## コンフィグ

エントリポイント: `src/tasks/wasb/configs/save_one_sample_visuals.yaml`

### 主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `split` | `train` | データ分割 (train/val/test) |
| `sample_index` | `0` | 保存するサンプルインデックス |
| `sample_indices` | `null` | 複数サンプルのインデックスリスト |
| `target_index` | `0` | ターゲットヒートマップのインデックス (0..frames_out-1) |
| `overlay_alpha` | `0.5` | ヒートマップオーバーレイの透明度 |
| `num_samples` | `1` | 保存するサンプル数 |

### data (データ設定)

通常の ball_detection データ設定を継承します。

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     save_one_sample_visuals.py                               │
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │  TennisDataMod  │                                                        │
│  │                 │                                                        │
│  │ augment=False   │──┐                                                     │
│  │ (オリジナル)    │  │                                                     │
│  └─────────────────┘  │                                                     │
│                       ├──▶  サンプル取得 ──▶ 画像保存                       │
│  ┌─────────────────┐  │                                                     │
│  │  TennisDataMod  │  │                                                     │
│  │                 │──┘                                                     │
│  │ augment=True    │                                                        │
│  │ (拡張あり)      │                                                        │
│  └─────────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

処理フロー:
1. 2つの DataModule を作成（拡張あり/なし）
2. 指定されたインデックスのサンプルを取得
3. 以下の画像を保存:
   - original.png: 元フレーム（拡張なし）
   - augmented.png: 拡張後フレーム
   - overlay.png: ヒートマップオーバーレイ
```

## 出力ファイル

```
outputs/wasb/ball_detection/visualize/save_one_sample_visuals/
├── sample_0_original.png    # 元フレーム
├── sample_0_augmented.png   # 拡張後フレーム
└── sample_0_overlay.png     # ヒートマップオーバーレイ
```

## 出力画像の内容

### original.png
- 元の入力フレーム
- データ拡張なし
- リサイズのみ適用

### augmented.png
- データ拡張後のフレーム
- Color Jitter, Grayscale, Blur 等が適用

### overlay.png
- 元フレームにターゲットヒートマップをオーバーレイ
- ヒートマップは入力サイズにリサイズ
- alpha でオーバーレイの強度を調整

```
overlay = clamp(frame + heatmap * alpha, 0, 1)
```

## 使用例

```bash
# データローダーが正しく動作しているか確認
uv run python -m src.tasks.wasb.scripts.visualize.save_one_sample_visuals \
  data.root_dir=data/tennis \
  data.train_matches=[game1] \
  split=train \
  sample_index=0

# 画像を確認
ls outputs/wasb/ball_detection/visualize/save_one_sample_visuals/
# sample_0_original.png
# sample_0_augmented.png
# sample_0_overlay.png
```

## デバッグのポイント

1. **ヒートマップの位置**: ボールの正しい位置にピークがあるか
2. **拡張の適切さ**: Color Jitter 等でボールが見えなくなっていないか
3. **座標の正確さ**: ボール座標がリサイズ後も正しく対応しているか
4. **可視性ラベル**: visibility = 0 のフレームでヒートマップがどうなるか

## 関連モジュール

- `src.tasks.wasb.data.datamodule.TennisDataModule`: データモジュール
