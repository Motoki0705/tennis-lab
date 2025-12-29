# Scripts Documentation

tennis-lab の各タスクで使用するスクリプトのドキュメントです。

## タスク概要

| タスク | 説明 | ドキュメント |
|--------|------|-------------|
| **PLCS** | Player Location from Court Skeleton - 2Dキーポイントから3Dプレーヤー位置推定 | [plcs/README.md](plcs/README.md) |
| **BLCS** | Ball Location from Court Skeleton - 2Dボール位置から3D軌道推定 | [blcs/README.md](blcs/README.md) |
| **WASB** | Where's the Ball - 映像からのボール検出・軌道補完・イベント検出 | [wasb/README.md](wasb/README.md) |

## 共通パターン

### 実行方法

すべてのスクリプトは Hydra を使用して設定を管理しています。

```bash
# 基本形式
uv run python -m src.{task}.scripts.{script_name}

# 設定の上書き
uv run python -m src.{task}.scripts.{script_name} key=value nested.key=value

# 例
uv run python -m src.plcs.scripts.train training.max_epochs=50 run.gpus=1
```

### 設定ファイル構造

各スクリプトには対応する YAML 設定ファイルがあります:

```
src/{task}/configs/
├── {script_name}.yaml       # メイン設定（defaults を含む）
├── run/                     # 実行設定
├── model/                   # モデル設定
├── data/                    # データ設定
├── training/                # 学習設定
└── ...
```

### 一般的なオプション

| オプション | 説明 |
|-----------|------|
| `run.seed` | 乱数シード |
| `run.gpus` | 使用するGPU数（0=CPU） |
| `run.fast_dev_run` | デバッグ用の高速実行 |
| `run.dry_run` | データ確認のみ（一部スクリプト） |
| `run.output_dir` | 出力ディレクトリ |

## クイックスタート

### PLCS (プレーヤー位置推定)

```bash
# データ生成 → 学習 → 可視化
uv run python -m src.plcs.scripts.generate_dataset
uv run python -m src.plcs.scripts.train
uv run python -m src.plcs.scripts.visualize visualization.mode=predict
```

### BLCS (ボール軌道推定)

```bash
# データ生成 → 学習 → 可視化
uv run python -m src.blcs.scripts.generate_dataset
uv run python -m src.blcs.scripts.train
uv run python -m src.blcs.scripts.visualize visualization.mode=predict
```

### WASB (ボール検出)

```bash
# データ生成 → 学習 → 推論可視化
uv run python -m src.wasb.scripts.generate_dataset mode=batch
uv run python -m src.wasb.scripts.train.ball_detection
uv run python -m src.wasb.scripts.visualize.ball_video video_path=...
```

## 詳細ドキュメント

各スクリプトの詳細（コマンド例、設定パラメータ、アーキテクチャ）は、タスクごとのサブディレクトリを参照してください:

- [PLCS Scripts](plcs/README.md)
- [BLCS Scripts](blcs/README.md)
- [WASB Scripts](wasb/README.md)
