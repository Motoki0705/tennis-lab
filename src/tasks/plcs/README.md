# PLCS (Player Localization in Court System)

> 2D pose（COCO 17点）とコートキーポイントから、コート座標系におけるプレイヤーの3D位置とヨー回転を推定するタスク。

## 概要

複数カメラ・時系列の2D観測をTransformerで統合し、コート座標系の位置と向きを復元します。frame / sequence / multiview の3モードを共通の `PLCSPredictor` で扱います。

| 項目 | 内容 |
|---|---|
| 入力 | 2D人物キーポイント `human_kp`（COCO17）+ コートキーポイント `court_kp` |
| 出力 | 3D位置 `position`（正規化 + メートル）+ 回転 `rotation`（cos/sin yaw） |
| モデル | `multiview_axial_{small,base,large,xlarge}` |
| データ | AMASS（ACCAD）モーション + SMPL-H による合成生成 |
| 役割 | 認識パイプラインの下流。2D pose検出と [court_detection](../court_detection/) の観測を3D化 |

## Inference

camera-time順 `(B, N, T, ...)` を正準とし、モデルに応じて内部でスライスされます。

```python
import torch
from src.tasks.plcs.inference.predictor import PLCSPredictor

predictor = PLCSPredictor.load_from_checkpoint(
    "outputs/plcs/plcs_multiview_axial/logs/version_0/checkpoints/last.ckpt",
    device="cpu",
)

# Multiview モード（複数カメラ×時系列）
human_kp   = torch.zeros(1, 3, 64, 17, 2)  # (B, N, T, 17, 2)
court_kp   = torch.zeros(1, 3, 64, 20, 2)  # (B, N, T, 20, 2)
human_mask = torch.ones(1, 3, 64)          # (B, N, T), True=valid
out = predictor.predict(human_kp, court_kp, human_mask=human_mask)
# out["position"] (1, 64, 3) / out["rotation"] (1, 64, 2)
```

`predict(human_kp, court_kp, human_vis=None, human_mask=None, court_vis=None, denormalize=True)`。返り値は全テンソルが CPU 上の `dict[str, Tensor]`。

| キー | 形状（Frame / Multiview） | 条件 | 説明 |
|---|---|---|---|
| `position` | `(B, 3)` / `(B, T, 3)` | 常時 | 正規化コート座標 |
| `rotation` | `(B, 2)` / `(B, T, 2)` | 常時 | `(cos yaw, sin yaw)`（L2正規化済み） |
| `position_meters` | `(B, 3)` / `(B, T, 3)` | `denormalize=True` | メートル単位の位置 |
| `yaw_radians` | `(B,)` / `(B, T)` | `denormalize=True` | ヨー角（ラジアン） |
| `canonical_pose` | `(B, 17, 3)` / `(B, T, 17, 3)` | `predict_canonical_pose=True` | ヨー正規化ローカル関節 |

スケールは `COURT_COORD_SCALE_(X,Y,Z) = (5.485, 11.885, 1.07)` [m]（`src/utils/schema/court.py`）。

### 入力テンソル仕様

| モード（`input_profile`） | `human_kp` | `court_kp` | `human_mask` |
|---|---|---|---|
| `frame` | `(B, 17, 2)` | `(B, 20, 2)` | `(B,)` または None |
| `sequence` | `(B, T, 17, 2)` | `(B, T, 20, 2)` | `(B, T)` |
| `multiview` | `(B, N, T, 17, 2)` | `(B, N, T, 20, 2)` | `(B, N, T)` |

## データセット生成

PLCSはAMASSモーションを用いて合成生成します（既存データセットなし）。

```bash
.venv/bin/python -m src.tasks.plcs.scripts.generate_dataset
.venv/bin/python -m src.tasks.plcs.scripts.generate_dataset simulation.num_scenes=10 run.num_workers=4

# 生成データの位置・ヨー・カメラ数分布を集計
.venv/bin/python -m src.tasks.plcs.scripts.analysis.analyze_dataset_distribution
```

- **仕様**: AMASS（ACCAD: running/walking/general）モーションをSMPL-Hで再生し、複数カメラ（1280×720、高さ3〜5m、FOV60°）へ投影。デフォルト1000シーン。
- **生成構造**（`data/plcs/scenes/scene_XXXXXX/`）:

```text
scene_000000/
├── meta.json / scalars.json   # num_cameras, fps, num_frames, カメラパラメータ
├── position.npy               # (T, 3) 正規化座標
├── rotation.npy               # (T, 2) (cos yaw, sin yaw)
├── canonical_pose_3d.npy      # (T, 17, 3) ヨー正規化ローカル関節
└── cam_{i}_*.npy              # human_kp_uv (T,17,2), court_kp_uv (T,20,2), *_visible ...
```

## 学習

```bash
# multiview_axial_base + multiview_sequence（デフォルト）
.venv/bin/python -m src.tasks.plcs.scripts.train

# chunked オンライン生成（大規模学習）
.venv/bin/python -m src.tasks.plcs.scripts.train_chunked
.venv/bin/python -m src.tasks.plcs.scripts.train_chunked data.chunk.scenes_per_chunk=500

# chunked + GAN（discriminator: base）
.venv/bin/python -m src.tasks.plcs.scripts.train_chunked_gan
```

### config の選択

**model**（`configs/model/`）:

| config | hidden_dim | num_layers | num_heads |
|---|---|---|---|
| `multiview_axial_small` | 256 | 8 | 4 |
| `multiview_axial_base` | 512 | 8 | 8 |
| `multiview_axial_large` | 512 | 12 | 8 |
| `multiview_axial_xlarge` | 1024 | 12 | 8 |

**data**（`configs/data/`）: `singleview_frame` / `singleview_sequence` / `multiview_sequence` / `chunked_multiview_sequence_bs8` / `chunked_multiview_sequence_bs16` / `chunked_multiview_sequence_bs32`

**training**（`configs/training/`）: `default` / `gan_small` / `gan_base` / `gan_large`

モデルサイズやbatch_sizeの差し替えはコマンドライン引数で指定します：

```bash
# XLargeモデル + bs16
.venv/bin/python -m src.tasks.plcs.scripts.train_chunked \
    model=multiview_axial_xlarge data=chunked_multiview_sequence_bs16

# GAN large discriminator
.venv/bin/python -m src.tasks.plcs.scripts.train_chunked_gan training=gan_large
```

出力先は `outputs/plcs/${model.name}/`。

## 可視化

```bash
.venv/bin/python -m src.tasks.plcs.scripts.visualize             # GT表示（デフォルト）

# ckptによる GT vs 予測の比較
.venv/bin/python -m src.tasks.plcs.scripts.visualize \
    visualization.mode=predict \
    visualization.checkpoint=outputs/plcs/plcs_multiview_axial/logs/version_0/checkpoints/last.ckpt \
    visualization.scene_path=data/plcs/scenes/scene_000000 \
    visualization.animation_view=3d \
    visualization.save=assets/plcs/pred.gif

# マルチビュー（全カメラ）
.venv/bin/python -m src.tasks.plcs.scripts.visualize visualization.cameras=all
```

可視化は単一の `visualize.py` に統合されており、`visualization.cameras=all`（または `0,1,2`）でマルチビューに対応します。`animation_view` は `3d` / `2d_topdown` / `camera`（`predict` モードでは `camera` 不可）。

## モデル

| 名前 | アーキテクチャ | 実装 |
|---|---|---|
| `frame` (`plcs`) | Llama系 decoder-only Transformer（MHA + SwiGLU + RMSNorm）。`[CLS, Register×4, court(20), player(17)]` トークン列に3軸MROPE | `models/plcs_model.py` |
| `multiview` (`plcs_multiview`) | 全カメラ×全時間を一括処理するTransformer。3軸MROPE、(camera,time)のCLSをカメラ次元で平均プール | `models/plcs_multiview_model.py` |
| `multiview_axial` (`plcs_multiview_axial`) | カメラ軸／時間軸の交互self-attention（Axial attention）。**現在のデフォルト** | `models/plcs_multiview_axial_model.py` |

`build_plcs_model(config)` が `config.model.name` でモデルを切り替えます。全バリアント共通で `predict_canonical_pose: true`。

## 補足

- 実行は `.venv/bin/python -m ...`。学習をCPUで試す場合は `run.gpus=0`。
- データ生成にはAMASSモーション（`data/ACCAD/`）とSMPL-Hモデル（`data/smplx/smplh/`、`smplx` パッケージ）が必要です。
