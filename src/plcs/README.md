# PLCS: Player Localization in Court System

テニスコート座標系におけるプレーヤーの位置と回転を推定するモデル。

## 概要

PLCSは、2D画像座標（UV座標）から得られるプレーヤーのポーズとコートのキーポイントを入力として、コート座標系におけるプレーヤーの3D位置と回転（yaw）を推定します。

### 入力
- **HumanKP17**: COCO形式の17点2Dキーポイント（UV座標、正規化済み）
- **CourtKP20**: コートの20点2Dキーポイント（UV座標、正規化済み）

### 出力
- **Position**: コート座標系における正規化された位置 `(x, y, z)`
- **Rotation**: yaw角を表す `(sin, cos)` ベクトル

## ディレクトリ構造

```
plcs/
├── configs/              # YAML設定ファイル
│   ├── default.yaml          # フレーム版学習用デフォルト設定
│   ├── sequence.yaml         # シーケンス版学習用デフォルト設定
│   └── simulation.yaml       # データ生成用設定
├── data/                 # データセット・生成
│   ├── dataset.py            # SceneDataset（フレーム単位、NPZファイル読み込み）
│   ├── sequence_dataset.py   # SceneSequenceDataset（シーケンス単位）
│   ├── datamodule.py         # Lightning DataModule（frame/sequence 両対応）
│   ├── motion_sampler.py     # AMASS/ACCADモーションサンプリング
│   └── scene_generator.py    # シーン生成・カメラ投影
├── inference/            # 推論API・可視化
├── models/               # モデルアーキテクチャ
│   └── components/       # エンコーダ・ヘッド
├── rendering/            # シーン可視化
├── scripts/              # CLIスクリプト
│   ├── generate_dataset.py   # データセット生成
│   ├── visualize.py          # 統合CLI（可視化・推論）
│   ├── train.py              # フレーム版モデル学習
│   └── train_sequence.py     # シーケンス版モデル学習
├── training/             # 学習関連
└── utils/                # ユーティリティ
```

## 使い方

### 1. データセット生成

AMASS/ACCADモーションデータからPLCS学習用データセットを生成します。

```bash
# デフォルト設定でデータセット生成
python -m plcs.scripts.generate_dataset --config plcs/configs/simulation.yaml

# カスタム設定
python -m plcs.scripts.generate_dataset \
    --config plcs/configs/simulation.yaml \
    --num-scenes 1000 \
    --output-dir data/plcs_scenes
```

生成されるファイル：
- `data/plcs_scenes/scene_XXXXXX.npz` - シーンデータ
- `data/plcs_scenes/meta.json` - データセット全体のメタデータ

### 2. シーン可視化・推論

統合CLIスクリプト `visualize.py` を使用します。3つのサブコマンドがあります：
- `visualize`: Ground Truthの可視化
- `predict`: 単フレームモデルで推論して可視化
- `predict-seq`: シーケンスモデルで推論して可視化

```bash
# Ground Truth: 3Dビュー
python -m plcs.scripts.visualize visualize data/plcs_scenes/scene_000000.npz --view 3d

# Ground Truth: カメラビュー
python -m plcs.scripts.visualize visualize data/plcs_scenes/scene_000000.npz --view camera

# Ground Truth: アニメーション
python -m plcs.scripts.visualize visualize data/plcs_scenes/scene_000000.npz --view animation

# シーン情報の表示
python -m plcs.scripts.visualize visualize data/plcs_scenes/scene_000000.npz --info
```

### 3. 学習

#### フレーム版モデル

```bash
# デフォルト設定で学習
python -m plcs.scripts.train

# カスタム設定
python -m plcs.scripts.train --config plcs/configs/default.yaml \
    --epochs 50 --batch-size 128 --lr 1e-4

# クイックテスト
python -m plcs.scripts.train --fast-dev-run
```

#### シーケンス版モデル

```bash
# シーケンス版モデルの学習（sequence.yaml）
python -m plcs.scripts.train_sequence --config plcs/configs/sequence.yaml

# クイックテスト
python -m plcs.scripts.train_sequence --fast-dev-run
```

### 4. 推論

#### フレーム版モデル

```bash
# チェックポイントからの推論＋可視化
python -m plcs.scripts.visualize predict data/plcs_scenes/scene_000000.npz \
    --checkpoint outputs/plcs/checkpoints/last.ckpt --view multi

# アニメーションで保存
python -m plcs.scripts.visualize predict data/plcs_scenes/scene_000000.npz \
    --checkpoint outputs/plcs/checkpoints/last.ckpt \
    --view animation --save outputs/pred_anim.mp4
```

#### シーケンス版モデル

```bash
# シーケンスモデルで推論＋可視化
python -m plcs.scripts.visualize predict-seq data/plcs_scenes/scene_000000.npz \
    --checkpoint outputs/plcs_sequence/checkpoints/last.ckpt --view multi

# 特定のカメラを使って推論
python -m plcs.scripts.visualize predict-seq data/plcs_scenes/scene_000000.npz \
    --checkpoint outputs/plcs_sequence/checkpoints/last.ckpt \
    --camera 1 --view animation
```

### Pythonからの使用

```python
from plcs import PLCSPredictor

# 学習済みフレーム版モデルをロード
predictor = PLCSPredictor.from_checkpoint("outputs/plcs/checkpoints/last.ckpt")

# 単一フレーム or バッチでの推論
result = predictor.predict(human_kp, court_kp)
print(f"Position: {result['position_meters']} m")
print(f"Yaw: {result['yaw_degrees']}°")
```

#### シーケンス版モデル（PLCSSequencePredictor）

```python
from plcs.inference import PLCSSequencePredictor

# 学習済みシーケンス版モデルをロード
seq_predictor = PLCSSequencePredictor.from_checkpoint(
    "outputs/plcs_sequence/checkpoints/last.ckpt",
)

# human_kp_seq: (T, 17, 2), court_kp_seq: (T, 20, 2)
seq_result = seq_predictor.predict(human_kp_seq, court_kp_seq)

print(seq_result["position_meters"].shape)  # (T, 3)
print(seq_result["yaw_degrees"].shape)      # (T,)
```

## 設定

### 学習設定 (`default.yaml`)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `model.hidden_dim` | 256 | 隠れ層の次元数 |
| `model.num_layers` | 4 | エンコーダの層数 |
| `model.num_heads` | 8 | アテンションヘッド数 |
| `training.learning_rate` | 1e-4 | 学習率 |
| `training.max_epochs` | 100 | 最大エポック数 |
| `data.batch_size` | 64 | バッチサイズ |
| `data.scene_dir` | `data/plcs_scenes` | シーンデータディレクトリ |

シーケンス版モデル向けには `sequence.yaml` で以下のような設定を追加します（例）：

- `data.seq_len`: 1 シーケンス内のフレーム数
- `data.seq_stride`: シーケンス間のストライド
- `model.temporal.*`: 時系列エンコーダ（Transformer）の構造
- `metrics.velocity_threshold_m`: 速度誤差 accuracy 用のしきい値（m）

### データ生成設定 (`simulation.yaml`)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `simulation.num_scenes` | 1000 | 生成シーン数 |
| `simulation.num_cameras` | 15 | サンプリングカメラ数 |
| `simulation.human_visibility_threshold` | 0.8 | 人物可視率閾値 |
| `simulation.court_visibility_threshold` | 15 | コートKP可視数閾値 |

## アーキテクチャ

PLCSモデルは以下のコンポーネントで構成されています：

1. **TransformerKeypointEncoder**: Human/Courtキーポイントをトークンとして処理し、Self-Attentionで関係性を学習
2. **PositionHead**: 位置 `(x, y, z)` を予測するMLP
3. **RotationHead**: 回転 `(sin, cos)` を予測するMLP（単位円上に正規化）

## 評価指標

| 指標 | 説明 |
|-----|------|
| `position_error_m` | 位置誤差（メートル） |
| `angular_error_deg` | 角度誤差（度） |
| `position_accuracy` | 閾値内の位置精度（デフォルト: 0.5m） |
| `angle_accuracy` | 閾値内の角度精度（デフォルト: 15°） |
| `temporal_velocity_error_m` | 連続フレーム間の速度誤差（m/s 相当、平均） |
| `temporal_velocity_accuracy` | 速度誤差が `metrics.velocity_threshold_m` 以内の割合 |

## テスト

```bash
# PLCSのユニットテストを実行
pytest tests/unit/plcs/ -v
```

## ライセンス

プロジェクトのライセンスに従います。
