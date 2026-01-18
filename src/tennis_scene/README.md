# Tennis Scene 3D Reconstruction

単眼動画からテニスシーンを3D再構築するための統合パイプライン。

## 概要

このモジュールは以下のコンポーネントを統合します：

- **Court KP Detection**: コートキーポイント検出（固定カメラ前提で1フレームのみ）
- **GVHMR**: 3D人物メッシュ（ローカルSMPL）+ 2Dスケルトン
- **WASB**: ボール2D検出
- **PLCS**: プレーヤー3D位置 + yaw推定
- **BLCS**: ボール3D軌道推定

## アーキテクチャ

オーケストレーション型のモジュラー設計を採用しています：

```
TennisSceneOrchestrator (オーケストレーター)
├── CourtKPModule      # コートKP検出
├── GVHMRModule        # 3D人物メッシュ推定
├── WASBModule         # ボール検出
├── PLCSModule         # プレーヤー3D位置推定
└── BLCSModule         # ボール3D軌道推定
```

各モジュールは独立して設定・ロード可能で、`BasePipelineModule` を継承しています。

## 固定カメラ前提

- コートKPは1フレーム（デフォルト: frame 0）から推定し、全フレーム共通
- GVHMRはカメラ回転推定なし（`static_cam=True`）
- GVHMRはローカルSMPLのみ取得
- PLCSの位置とyawをSMPLメッシュに適用

## 使用方法

### 基本実行

```bash
uv run python -m src.tennis_scene.scripts.run_pipeline \
    video_path=inputs/demo/match.mp4
```

### オプション指定

```bash
uv run python -m src.tennis_scene.scripts.run_pipeline \
    video_path=inputs/demo/match.mp4 \
    max_frames=100 \
    court_kp.frame_index=0 \
    output_dir=outputs/tennis_scene
```

### GVHMRスキップ（デバッグ用）

```bash
uv run python -m src.tennis_scene.scripts.run_pipeline \
    video_path=inputs/demo/match.mp4 \
    gvhmr.skip=true
```

### カスタムモデルパス指定

```bash
uv run python -m src.tennis_scene.scripts.run_pipeline \
    video_path=inputs/demo/match.mp4 \
    gvhmr.yolo_checkpoint=/path/to/yolov8.pt \
    gvhmr.vitpose_checkpoint=/path/to/vitpose.pth
```

## 設定

設定ファイル: `src/tennis_scene/configs/pipeline.yaml`

### 主要な設定項目

| キー | 説明 | デフォルト |
|------|------|----------|
| `video_path` | 入力動画パス | 必須 |
| `device` | 推論デバイス | `cuda` |
| `max_frames` | 最大フレーム数 | `null`（全フレーム） |
| `court_kp.checkpoint` | Court KPモデル | `outputs/court_detection/checkpoints/last.ckpt` |
| `court_kp.frame_index` | Court KP検出フレーム | `0` |

### GVHMR設定

| キー | 説明 | デフォルト |
|------|------|----------|
| `gvhmr.checkpoint` | GVHMRモデル | `third_party/GVHMR/inputs/checkpoints/gvhmr/...` |
| `gvhmr.yolo_checkpoint` | YOLOトラッカー | `third_party/GVHMR/inputs/checkpoints/yolo/yolov8x.pt` |
| `gvhmr.vitpose_checkpoint` | ViTPose | `third_party/GVHMR/inputs/checkpoints/vitpose/...` |
| `gvhmr.hmr2_checkpoint` | HMR2特徴抽出 | `third_party/GVHMR/inputs/checkpoints/hmr2/...` |
| `gvhmr.skip` | GVHMRスキップ | `false` |

### WASB/PLCS/BLCS設定

| キー | 説明 | デフォルト |
|------|------|----------|
| `wasb.checkpoint` | WASBモデル | `third_party/WASB-SBDT/pretrained/...` |
| `wasb.skip` | ボール検出スキップ | `false` |
| `plcs.checkpoint` | PLCSモデル | `outputs/plcs/checkpoints/last.ckpt` |
| `blcs.checkpoint` | BLCSモデル | `outputs/blcs/checkpoints/last.ckpt` |

## 出力

出力ファイル: `{output_dir}/{video_name}.npz`

`SceneResult` データ構造：

```python
SceneResult:
    num_frames: int              # フレーム数
    fps: float                   # FPS
    width: int                   # 動画幅
    height: int                  # 動画高さ
    court_kp: (20, 2)            # コートKP（正規化）
    court_vis: (20,)             # コートKP可視性
    player_position: (T, 3)      # プレーヤー3D位置（メートル）
    player_yaw: (T,)             # プレーヤーyaw（ラジアン）
    smpl_body_pose: (T, 63)      # SMPLボディポーズ
    smpl_global_orient: (T, 3)   # SMPLグローバル向き
    smpl_betas: (10,)            # SMPL形状パラメータ
    smpl_vertices_local: (T, V, 3)   # ローカルSMPL頂点
    smpl_vertices_global: (T, V, 3)  # グローバルSMPL頂点（PLCS適用済み）
    ball_uv: (T, 2)              # ボール2D位置（正規化）
    ball_visibility: (T,)        # ボール可視性
    ball_3d: (T, 3)              # ボール3D位置（メートル）
    human_kp_2d: (T, 17, 2)      # 人物2DKP（正規化）
    human_kp_vis: (T, 17)        # 人物KP可視性
```

## 座標変換

SMPLメッシュの座標変換は以下の手順で行われます：

1. **Yaw回転**: PLCSのyawでY軸周りに回転
2. **平行移動**: PLCSの3D位置を加算

```python
global_smpl_verts = rotate_y(local_smpl_verts, yaw) + position
```

## モジュール構成

```
src/tennis_scene/
├── __init__.py              # パッケージ定義
├── io.py                    # SceneResultデータ構造
├── transforms.py            # 座標変換ユーティリティ
├── pipeline/                # モジュラーパイプライン
│   ├── __init__.py          # エクスポート
│   ├── base.py              # BasePipelineModule基底クラス
│   ├── orchestrator.py      # TennisSceneOrchestrator
│   ├── court_kp.py          # CourtKPModule
│   ├── gvhmr.py             # GVHMRModule + GVHMRConfig
│   ├── wasb.py              # WASBModule + WASBConfig
│   ├── plcs.py              # PLCSModule
│   └── blcs.py              # BLCSModule
├── configs/
│   └── pipeline.yaml        # Hydra設定
├── scripts/
│   └── run_pipeline.py      # エントリポイント
└── README.md                # このファイル
```
