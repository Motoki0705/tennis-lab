# Tennis-DETR v2 モデル仕様（Spec）

本書は、テニス用マルチビュー 3D ポーズ推定モデル v2 `TennisDETR_v2` と、その設定クラス `TennisDetrV2Config` の仕様をまとめる。

実装:
- モデル本体: `src/models/tennis_multi_cam_3d_pose/model_v2.py:TennisDETR_v2`
- 設定クラス: `src/models/tennis_multi_cam_3d_pose/config_v2.py:TennisDetrV2Config`
- ファクトリ: `src/models/tennis_multi_cam_3d_pose/factory.py:create_tennis_model("v2")`

---

## 1. 目的

v2はv1からの主要な変更点として、**階層エンコーダ**と**分離出力**を採用。

- 入力:
  - マルチカメラ・マルチプレーヤーの 2D キーポイント（pose17 + racket3 = 20 点）:
    - `keypoints_2d[B, T, V, M, J, 2]`
  - プレーヤー存在マスク:
    - `player_mask[B, T, V, M]`（True=観測有り）
  - コート 2D キーポイント:
    - `court_kpts_2d[B, V, 20, 2]`
- 出力（分離形式）:
  - **Canonical 3D Pose**: ルート相対・回転なしのポーズ
    - `canonical_pose[B, Q, T, J, 3]`
  - **Root Translation**: コート上の絶対位置
    - `root_trans[B, Q, T, 3]`
  - **Root Rotation**: 向き（cos, sin）
    - `root_rot[B, Q, T, 2]`
  - **Global 3D Pose**: 再構成された絶対座標ポーズ
    - `global_pose[B, Q, T, J, 3]`
  - **Exist Logit**: Query の存在ロジット（BCEWithLogits 用の生出力）
    - `exist_logit[B, Q, 1]`
  - **Exist Confidence**: Query の存在確率（`sigmoid(exist_logit)`）
    - `exist_conf[B, Q, 1]`

ここで:
- `B`: バッチサイズ
- `T`: 時間ウィンドウ長
- `V`: カメラ数
- `M`: 1 フレームあたり最大プレーヤー数（パディング含む）
- `J`: キーポイント数（20）
- `Q`: Query 数（最大プレーヤー数スロット、通常は `M` 以上）

---

## 2. 設定クラス: `TennisDetrV2Config`

```python
@dataclass(slots=True)
class TennisDetrV2Config:
    # Transformer dimensions
    D_model: int = 256
    dim_feedforward: int = 1024
    nheads: int = 8
    decoder_layers: int = 6
    dropout: float = 0.1

    # v2階層エンコーダパラメータ
    intra_layers: int = 3    # プレーヤー内エンコーダ層数
    inter_layers: int = 3    # プレーヤー間エンコーダ層数
    temporal_layers: int = 3 # 時間エンコーダ層数

    # Tokens / queries
    num_joints: int = 20
    num_court_points: int = 20
    num_queries: int = 20

    # Positional embeddings
    max_cameras: int = 8
    max_frames: int = 32
```

---

## 3. アーキテクチャ

### 3.1 階層エンコーダ

v2では単一エンコーダの代わりに3段階の階層構造を採用:

1. **Intra-Encoder** (`intra_layers`):
   - 各プレーヤーの時間系列を独立に処理
   - 個別の動きパターンを学習

2. **Inter-Encoder** (`inter_layers`):
   - プレーヤー間の相互作用をモデリング
   - ゲーム状況における相対関係を学習

3. **Temporal-Encoder** (`temporal_layers`):
   - 時間的な依存関係をグローバルに処理
   - ラリー全体の文脈を理解

### 3.2 分離出力ヘッド

```
Encoder Output → [Canonical Head, Root Trans Head, Root Rot Head] → Global Pose (再構成)
```

- **Canonical Head**: ルートを原点とした標準化ポーズを出力
- **Root Trans Head**: コート座標系での位置を出力
- **Root Rot Head**: プレーヤーの向きを出力
- **再構成**: `R(root_rot) @ canonical_pose + root_trans → global_pose`

---

## 4. 損失関数

v2では4要素の損失を使用:

```yaml
loss:
  lambda_canonical: 1.0      # Canonical pose損失
  lambda_root_trans: 1.0     # Root translation損失
  lambda_root_rot: 0.5       # Root rotation損失
  lambda_global: 1.0         # Global pose損失
  lambda_exist: 0.05         # 存在確率損失
  lambda_vel: 0.0            # 速度損失（未使用）

  # マッチング重み
  lambda_pose_match: 1.0     # ポーズマッチング重み
  lambda_exist_match: 0.05   # 存在マッチング重み
```

### 4.1 GTデータ生成

既存の`pose_3d_gt`から自動的にv2用GTを生成:

```python
def _decompose_pose_for_v2(pose_3d):
    # pose_3d: [T, M, J, 3]  # 1ウィンドウ (window) のデータ
    # → canonical_pose, root_trans, root_rot, global_pose
```

---

## 5. 使用方法

### 5.1 モデル生成

```python
from src.models.tennis_multi_cam_3d_pose import create_tennis_model

# v2モデルの生成
model = create_tennis_model("v2")

# カスタム設定での生成
from src.models.tennis_multi_cam_3d_pose import TennisDetrV2Config
cfg = TennisDetrV2Config(intra_layers=4, inter_layers=4)
model = create_tennis_model("v2", cfg)
```

### 5.2 学習実行

```bash
# v2用学習スクリプト
./scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh

# 設定ファイル指定
CONFIG=configs/tennis_multi_cam_3d_pose_v2.yaml \
  ./scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh

# パラメータ上書き
./scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh \
  --set training.trainer.max_epochs=100
```

### 5.3 設定ファイル

```yaml
# configs/tennis_multi_cam_3d_pose_v2.yaml
task: tennis_multi_cam_3d_pose
model:
  _target_: src.models.tennis_multi_cam_3d_pose.create_tennis_model
  model_version: "v2"
  cfg:
    _target_: src.models.tennis_multi_cam_3d_pose.TennisDetrV2Config
    intra_layers: 3
    inter_layers: 3
    temporal_layers: 3
    # ... 他のパラメータ
training:
  _target_: src.training.tennis_multi_cam_3d_pose.TennisDetrV2Module
  # ... 学習設定
```

---

## 6. v1との比較

| 項目 | v1 | v2 |
|------|----|----|
| エンコーダ | 単一エンコーダ | 階層エンコーダ（3段階） |
| 出力形式 | pose_3d [B,Q,T,J,3] | 分離出力（4要素） |
| 設定クラス | TennisDetrConfig | TennisDetrV2Config |
| 損失関数 | 単一ポーズ損失 | 4要素損失 |
| パラメータ数 | ~11.2M | ~13.7M |
| 表現力 | 標準的 | 高い（分離表現） |

---

## 7. 注意事項

- v2用GTデータは`pose_3d_gt`から自動生成されるため、既存データセットで学習可能
- 階層エンコーダにより計算コストが増加するが、表現力が向上
- 分離出力により、各要素の解釈可能性が向上
- v1とv2は`create_tennis_model()`で明確に区別して使用
