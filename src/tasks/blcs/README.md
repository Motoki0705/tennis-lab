# BLCS (Ball Localization in Court System)

> 2Dボール観測とコートキーポイントから、コート座標系における3Dボール軌道を推定するタスク。単一カメラ／マルチビューに対応します。

## 概要

複数フレーム（・複数視点）の2D観測を時系列Transformerで統合し、3D位置（と速度）を復元します。

| 項目 | 内容 |
|---|---|
| 入力 | 2Dボール観測 `ball_uv` + コートキーポイント `court_kp` |
| 出力 | 3D位置 `position (B, T, 3)` [m] + 速度 `velocity (B, T, 3)` [m/s]（任意） |
| モデル | `single` / `multiview` / `multiview_axial`（デフォルト） |
| データ | 物理シミュレーションによる合成生成 |
| 役割 | 認識パイプラインの下流。[ball_detection](../ball_detection/) と [court_detection](../court_detection/) の2D観測を3D化 |

## Inference

単一カメラ・マルチビューとも同一の `BLCSPredictor` で扱います。

```python
import torch
from src.tasks.blcs.inference.predictor import BLCSPredictor

predictor = BLCSPredictor.load_from_checkpoint(
    "outputs/blcs/multiview_axial/logs/version_0/checkpoints/last.ckpt",
    device="cpu",
)

# 単一カメラ
ball_uv  = torch.zeros(1, 64, 2)      # (B, T, 2)
court_kp = torch.zeros(1, 20, 2)      # (B, 20, 2)
out = predictor.predict(ball_uv, court_kp)          # out["position"]: (1, 64, 3) [m]

# マルチビュー（ball_vis / ball_mask が必須）
ball_uv   = torch.zeros(1, 4, 64, 2)      # (B, N, T, 2)
court_kp  = torch.zeros(1, 4, 64, 20, 2)  # (B, N, T, 20, 2)
ball_vis  = torch.ones(1, 4, 64)          # (B, N, T)
ball_mask = torch.ones(1, 4, 64)          # (B, N, T)
out = predictor.predict(ball_uv, court_kp, ball_vis=ball_vis, ball_mask=ball_mask)
```

`predict(ball_uv, court_kp, ball_vis=None, ball_mask=None, court_vis=None, denormalize=True)`。返り値は全テンソルが CPU 上の `dict[str, Tensor]`。

| キー | 形状 | dtype | 条件 | 説明 |
|---|---|---|---|---|
| `position` | `(B, T, 3)` | float32 | 常時 | `denormalize=True` でメートル、`False` で正規化座標 |
| `velocity` | `(B, T, 3)` | float32 | モデルが `predict_velocity` で学習された場合 | `denormalize=True` で m/s |

`denormalize=True` のスケールは `COURT_COORD_SCALE_XYZ = (5.485, 11.885, 1.07)`（`src/utils/schema/court.py`）。

### 入力テンソル仕様

| テンソル | 単一カメラ | マルチビュー | 備考 |
|---|---|---|---|
| `ball_uv` | `(B, T, 2)` | `(B, N, T, 2)` | 必須 |
| `court_kp` | `(B, 20, 2)`（`(B, 40)` も可） | `(B, N, T, 20, 2)`（`(B, N, 20, 2)` も可） | 必須 |
| `ball_vis` | `(B, T)` | `(B, N, T)` | マルチビューでは**必須** |
| `ball_mask` | `(B, T)` | `(B, N, T)` | マルチビューでは**必須** |

## データセット生成

BLCSはデータを物理シミュレーションで合成生成します（既存データセットなし）。

```bash
.venv/bin/python -m src.tasks.blcs.scripts.generate_dataset
.venv/bin/python -m src.tasks.blcs.scripts.generate_dataset generator.num_scenes=500 run.num_workers=4
```

- **仕様**: 重力・空気抵抗・マグヌス力・バウンド（`configs/physics/default.yaml`）でラリーをシミュレートし、固定6カメラ（コーナー4 + ベースライン中点2）へ投影。1シーン=1ラリー（最大10ショット）。デフォルト1000シーン、`train/val/test = 0.8/0.1/0.1`。
- **生成構造**（`data/blcs/scenes/scene_XXXXXX/`）:

```text
scene_000000/
├── meta.json / scalars.json   # num_cameras, fps, rally_length, カメラパラメータ
├── ball_pos_world.npy         # (T, 3) ワールド座標 [m]
├── ball_pos_norm.npy          # (T, 3) 正規化座標
├── ball_vel_world.npy         # (T, 3) 速度 [m/s]
└── cam_{0..5}_*.npy           # ball_uv (T,2), ball_visible (T,), court_kp_uv (20,2), ...
```

## 学習

```bash
.venv/bin/python -m src.tasks.blcs.scripts.train                              # multiview_axial（デフォルト）
.venv/bin/python -m src.tasks.blcs.scripts.train model=single   data=single
.venv/bin/python -m src.tasks.blcs.scripts.train model=multiview data=multiview
.venv/bin/python -m src.tasks.blcs.scripts.train_chunked                      # オンライン生成のチャンク学習
```

`configs/training/default.yaml` は `max_epochs=100`、`lr=1e-4`、warmup=200、cosineスケジューラ、`bf16-mixed`。

| モデル config | data config | 概要 |
|---|---|---|
| `single` | `single` | 単一カメラ（batch=32、seq 64〜1024） |
| `multiview` | `multiview` | クロスアテンション統合（batch=4、view 4〜6） |
| `multiview_axial` | `multiview` | 軸別注意（デフォルト） |

## 可視化

```bash
.venv/bin/python -m src.tasks.blcs.scripts.visualize                          # multiview GT（デフォルト）
.venv/bin/python -m src.tasks.blcs.scripts.visualize visualization=single

# ckptによる GT vs 予測の比較アニメーション
.venv/bin/python -m src.tasks.blcs.scripts.visualize \
    visualization=multiview visualization.mode=predict \
    visualization.checkpoint=outputs/blcs/multiview/logs/version_0/checkpoints/last.ckpt \
    visualization.cameras=all \
    visualization.save=outputs/blcs/visualize/compare_multiview.mp4
```

`visualization.animation_view` は `2d` / `3d` をサポートします。

## モデル

| 名前 | アーキテクチャ | 実装 |
|---|---|---|
| `single` (`blcs`) | Decoder-only Transformer（RoPE + SDPA + SwiGLU + RMSNorm）。`[court(20), ball(T)]` トークン列を自己注意 | `models/blcs_model.py` |
| `multiview` (`blcs_multiview`) | クエリベースのクロスアテンション（同時刻のマルチビューを統合→時間方向自己注意） | `models/blcs_multiview_model.py` |
| `multiview_axial` (`blcs_multiview_axial`) | カメラ軸／時間軸の交互self-attention（ローカル時間窓 `time_window_radius`） | `models/blcs_multiview_axial_model.py` |

## 補足

- 実行は `.venv/bin/python -m ...`。学習をCPUで試す場合は `run.gpus=0`。
- 生成・学習・可視化はすべて `data/blcs/scenes/` のシーンを起点にします（`train_chunked` はオンライン生成）。
