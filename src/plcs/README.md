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

## 使用方法

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
├── scripts/              # CLIスクリプト
│   ├── generate_dataset.py   # データセット生成
│   ├── visualize.py          # 統合CLI（可視化・推論）
│   ├── train.py              # フレーム版モデル学習
│   └── train_sequence.py     # シーケンス版モデル学習
├── training/             # 学習関連
└── utils/                # ユーティリティ
```

## データセット形式

1シーン = 1ファイル（複数カメラ）：

```
scene_000000.npz
├── meta (JSON)                     # シーンメタデータ
├── position [T, 3]                 # コート座標系での正規化位置 (x, y, z)
├── rotation [T, 2]                 # yaw角の (sin, cos) 表現
├── canonical_pose_3d [T, J, 3]     # ローカル座標系での3Dポーズ（骨盤原点）
├── num_cameras                     # 有効カメラ数
├── cam_0_params (JSON)             # カメラ0のパラメータ（中心C, 回転R, f, cx, cy, w, h など）
├── cam_0_human_kp_uv [T, 17, 2]    # カメラ0でのHumanKP17（UV座標, 正規化）
├── cam_0_court_kp_uv [T, 20, 2]    # カメラ0でのCourtKP20（UV座標, 正規化）
├── cam_0_human_kp_visible [T, 17]  # カメラ0での人物キーポイント可視フラグ
├── cam_0_court_kp_visible [T, 20]  # カメラ0でのコートキーポイント可視フラグ
├── cam_0_human_visibility_ratio    # カメラ0での人物可視率
├── cam_0_court_visibility_count    # カメラ0での平均可視コートKP数
├── cam_1_...                       # カメラ1以降も同様の形式
└── ...
```

`position` / `rotation` / `canonical_pose_3d` は、それぞれ PLCS モデルのターゲットとなる 3D 位置・回転・ポーズを表し、`SceneDataset` / `SceneSequenceDataset` から読み込まれます。

`meta.json` には、各シーンの統計情報が JSON 形式で保存されます（`SceneGenerator.build_scene_meta` 参照）：

- `scene_id`: シーンID
- `motion_source`: モーションの元となるファイルパス（AMASS/ACCAD）
- `motion_category`: モーションカテゴリ
- `gender`: 被験者の性別
- `fps`: フレームレート (frames per second)
- `num_frames`: フレーム数 T
- `duration_sec`: シーン長（秒）
- `initial_position`: 初期位置 (x, y)
- `initial_yaw_deg`: 初期yaw角（度）
- `num_cameras_sampled`: サンプリングしたカメラ本数
- `num_cameras`: フィルタ後に残った有効カメラ本数
- `cameras`: 各カメラのメタ情報のリスト（`human_visibility_ratio`, `court_visibility_count` など）
- `position_range`: x/y/z それぞれの最小値・最大値

## アーキテクチャ

```
Input: human_kp [B, 17, 2], court_kp [B, 20, 2]
        │                          │
        └────────────┬─────────────┘
                     │
                     ▼
          (TransformerKeypointEncoder
               または KeypointEncoder)
                     │
                     ▼
                 [B, D]
                     │
          ┌──────────┴───────────┐
          │                      │
   CombinedHead を使用:   Separate Heads を使用:
          │                      │
          ▼                      ▼
   position [B, 3],        PositionHead → position [B, 3]
   rotation [B, 2]         RotationHead → rotation [B, 2]
```

- エンコーダ部では、`use_transformer=True` の場合は `TransformerKeypointEncoder` が、`False` の場合は MLP ベースの `KeypointEncoder` が用いられます。
- 出力ヘッド部では、`use_combined_head=True` の場合は `CombinedHead` が位置・回転を同時に出力し、`False` の場合は `PositionHead` / `RotationHead` の 2 つのヘッドからそれぞれ出力します。

## 座標系

### 入力UV座標
- `u = x / W` (画像幅で正規化, 0〜1)
- `v = y / H` (画像高さで正規化, 0〜1)

### 出力コート座標（正規化）
- `x = X / HALF_DOUBLES_WIDTH` (5.485m)
- `y = Y / HALF_LENGTH` (11.885m)
- `z = Z / NET_HEIGHT_POST` (1.07m)

## 評価指標

- **position_error_m**: 位置誤差（メートル）
- **angular_error_deg**: 角度誤差（度）
- **position_accuracy**: 閾値内の位置精度（デフォルト: 0.5m）
- **angle_accuracy**: 閾値内の角度精度（デフォルト: 15°）
- **temporal_velocity_error_m**: 連続フレーム間の速度誤差（m/s 相当、平均）
- **temporal_velocity_accuracy**: 速度誤差が `metrics.velocity_threshold_m` 以内の割合
