# WASB visualize/trajectory

軌道補完モデルの予測結果を可視化するスクリプト。

## 概要

このスクリプトは、学習済みの軌道補完モデル（BiLSTM, Transformer等）を使って、テストデータに対する予測結果を可視化します。入力（部分的に欠損した軌道）、Ground Truth、予測結果を比較するプロットを生成します。

## コマンド例

```bash
# 基本的な使用法
uv run python -m src.wasb.scripts.visualize.trajectory \
  visualization.checkpoint=outputs/trajectory/logs/version_0/checkpoints/last.ckpt

# 出力ディレクトリとサンプル数を指定
uv run python -m src.wasb.scripts.visualize.trajectory \
  visualization.checkpoint=... \
  visualization.output_dir=outputs/trajectory/vis \
  visualization.num_samples=16

# データ分割を指定
uv run python -m src.wasb.scripts.visualize.trajectory \
  visualization.checkpoint=... \
  visualization.split=val
```

## コンフィグ

エントリポイント: `src/wasb/configs/visualize_trajectory.yaml`

### defaults 構成

```yaml
defaults:
  - data: trajectory
  - training: trajectory
  - logging: default
  - metrics: trajectory
  - run: visualize_trajectory
  - model: trajectory_bilstm
  - visualization: trajectory
```

### visualization (可視化設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `checkpoint` | `null` | チェックポイントパス |
| `output_dir` | `outputs/trajectory/vis` | 出力ディレクトリ |
| `split` | `test` | データ分割 (train/val/test) |
| `num_samples` | `8` | 可視化するサンプル数 |

### run (実行設定)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `gpus` | (from config) | GPU数 |
| `seed` | `42` | 乱数シード |

## アーキテクチャ・フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          trajectory.py                                       │
│                                                                              │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────────┐  │
│  │TrajectoryDataMod│──────▶│TrajectoryLightni│──────▶│    Visualization    │  │
│  │                 │      │(from checkpoint)│      │                     │  │
│  │ - xy_input      │      │ - 推論実行      │      │ - matplotlib       │  │
│  │ - target_xy     │      │ - pred_xy       │      │ - PNG 保存          │  │
│  │ - masks         │      │                 │      │                     │  │
│  └─────────────────┘      └─────────────────┘      └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

処理フロー:
1. チェックポイントからモデルをロード
2. 指定されたデータ分割からサンプルを取得
3. 各サンプルについて推論を実行
4. 入力、Ground Truth、予測を比較プロット
5. PNG ファイルとして保存
```

## 出力プロットの内容

各サンプルについて以下を表示:

- **グレー線**: Ground Truth 軌道
- **青点**: クリーン入力（マスクなし）
- **オレンジ点**: ノイズ付加された入力
- **紫×**: ノイズ位置での予測
- **赤×**: ブロック欠損位置での予測
- **緑×**: スパース欠損位置での予測

```
┌────────────────────────────────────────────┐
│                                            │
│      ●●●           ●●●●●                  │
│    ●     ●       ●       ●                 │
│   ●       ●     ●         ●                │
│  ●         ●   ●           ●               │
│ ●           ● ×             ×              │
│●             ×               ×             │
│              ×                ×            │
│               ×                ●           │
│                                 ●          │
│                                  ●●●       │
│                                            │
│  ● input clean   × pred block              │
│  ○ input noisy   × pred sparse             │
│  ─ ground truth                            │
└────────────────────────────────────────────┘
```

## 出力構造

```
outputs/trajectory/vis/
├── sample_0.png
├── sample_1.png
├── sample_2.png
└── ...
```

## 使用例

```bash
# 1. 軌道補完モデルを学習
uv run python -m src.wasb.scripts.train.trajectory

# 2. 結果を可視化
uv run python -m src.wasb.scripts.visualize.trajectory \
  visualization.checkpoint=outputs/trajectory/trajectory_bilstm/logs/version_0/checkpoints/last.ckpt \
  visualization.num_samples=20

# 3. 出力を確認
ls outputs/trajectory/vis/
```

## 評価のポイント

- **ブロック欠損**: 連続した欠損をどれだけ正確に補完できているか
- **スパース欠損**: 散発的な欠損を滑らかに補完できているか
- **ノイズ除去**: ノイズの多い入力をどれだけ滑らかにできているか
- **物理的妥当性**: 補完された軌道が物理的に妥当か（放物線に近いか）

## 関連モジュール

- `src.wasb.data.trajectory_datamodule`: データモジュール
- `src.wasb.training.TrajectoryLightningModule`: Lightning モジュール
