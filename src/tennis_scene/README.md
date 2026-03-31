# Tennis Scene 3D Reconstruction

単眼動画からテニスシーンを3D再構築するための統合パイプライン。

## 概要

このモジュールは以下のコンポーネントを統合します：

- **Court KP Detection**: コートキーポイント検出（固定カメラ前提で1フレームのみ）
- **GVHMR**: 3D人物メッシュ（ローカルSMPL）+ 2Dスケルトン
- **WASB**: ボール2D検出
- **Trajectory Completion**: ボール2D軌道補完（optional）
- **Event UV**: UV軌道上のイベント検出（shot/bounce）
- **PLCS**: プレーヤー3D位置 + yaw推定
- **BLCS**: ボール3D軌道推定
- **Event 3D**: 3D軌道上のイベント検出（shot/bounce）

## アーキテクチャ

オーケストレーション型のモジュラー設計を採用しています：

```
TennisSceneOrchestrator (オーケストレーター)
├── CourtKPModule      # コートKP検出
├── GVHMRModule        # 3D人物メッシュ推定
├── WASBModule         # ボール検出
├── TrajectoryModule   # ボール2D軌道補完（optional）
├── EventUVModule      # UVイベント検出
├── PLCSModule         # プレーヤー3D位置推定
├── BLCSModule         # ボール3D軌道推定
└── Event3DModule      # 3Dイベント検出
```

各モジュールは独立して設定・ロード可能で、`BasePipelineModule` を継承しています。

### ステージ依存関係

- `PLCS <- COURT_KP, GVHMR`（PLCSはcourt_kpとGVHMRのhuman_kpを使用）
- `BLCS <- COURT_KP, WASB`
- `TRAJECTORY <- WASB`（optional）
- `EVENT_UV <- WASB`
- `EVENT_3D <- BLCS`

## 固定カメラ前提

- コートKPは1フレーム（デフォルト: frame 0）から推定し、全フレーム共通
- GVHMRはカメラ回転推定なし（`static_cam=True`）
- GVHMRはローカルSMPLのみ取得
- PLCSの位置とyawをSMPLメッシュに適用

## 使用方法

### 基本実行

```bash
python -m src.tennis_scene.scripts.run_pipeline \
    video_path=inputs/demo/match.mp4
```

### オプション指定

```bash
python -m src.tennis_scene.scripts.run_pipeline \
    video_path=inputs/demo/match.mp4 \
    max_frames=100 \
    court_kp.frame_index=0 \
    output_dir=outputs/tennis_scene
```

### Court KPをUIで入力

手動入力UIを使う場合は `manual_ui` を指定します。結果JSONは `court_kp.output_path` に保存されます。

```bash
python -m src.tennis_scene.scripts.run_pipeline \
    video_path=inputs/demo/match.mp4 \
    court_kp.mode=manual_ui \
    court_kp.output_path=outputs/tennis_scene/court_kp_result.json
```

### GVHMRスキップ（デバッグ用）

```bash
python -m src.tennis_scene.scripts.run_pipeline \
    video_path=inputs/demo/match.mp4 \
    gvhmr.skip=true
```

### カスタムモデルパス指定

```bash
python -m src.tennis_scene.scripts.run_pipeline \
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
| `court_kp.mode` | Court KPの入力モード（`model`/`manual_ui`） | `model` |

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
| `trajectory.checkpoint` | 軌道補完モデル | `checkpoints/trajectory_completion/uv/last.ckpt` |
| `trajectory.skip` | 軌道補完スキップ（optional） | `true` |
| `event_uv.checkpoint` | UVイベント検出モデル | `checkpoints/event_detection/uv/last.ckpt` |
| `event_uv.skip` | UVイベント検出スキップ | `true` |
| `plcs.checkpoint` | PLCSモデル | `outputs/plcs/frame/logs/version_0/checkpoints/last.ckpt` |
| `blcs.checkpoint` | BLCSモデル | `outputs/blcs/single/logs/version_0/checkpoints/last.ckpt` |
| `blcs.skip` | BLCSスキップ | `false` |
| `event_3d.checkpoint` | 3Dイベント検出モデル | `checkpoints/event_detection/traj3d/last.ckpt` |
| `event_3d.skip` | 3Dイベント検出スキップ | `true` |

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
    ball_uv: (T, 2)              # ボール2D位置（正規化）
    ball_uv_pred: (T, 2)         # 軌道補完モデルの生予測
    ball_uv_completed: (T, 2)    # 観測値マージ後の補完UV
    ball_visibility: (T,)        # ボール可視性
    ball_3d: (T, 3)              # ボール3D位置（メートル）
    event_uv_probs: (T, E)       # UVイベント確率
    event_uv_peak_mask: (T, E)   # UVイベントピーク位置
    event_uv_names: list[str]    # UVイベント名
    event_3d_probs: (T, E)       # 3Dイベント確率
    event_3d_peak_mask: (T, E)   # 3Dイベントピーク位置
    event_3d_names: list[str]    # 3Dイベント名
    human_kp_2d: (T, 17, 2)      # 人物2DKP（正規化）
    human_kp_vis: (T, 17)        # 人物KP可視性
```

## 座標変換

SMPLメッシュの座標変換は以下の手順で行われます：

1. `smpl_vertices_local` から root を差し引き、メッシュを root 基準に中心化
2. `smpl_global_orient` の逆回転で GVHMR 由来の向きを打ち消し
3. PLCS の yaw を Z 軸回転として適用
4. PLCS の 3D 位置を加算

## モジュール構成

```
src/tennis_scene/
├── __init__.py              # パッケージ定義
├── io.py                    # SceneResultデータ構造
├── utils/
│   ├── __init__.py
│   └── transforms.py        # 座標変換ユーティリティ
├── pipeline/                # モジュラーパイプライン
│   ├── __init__.py          # エクスポート
│   ├── dependency_graph.py  # ステージ依存の保持/解決/検証
│   ├── orchestrator.py      # TennisSceneOrchestrator
│   └── components/
│       ├── __init__.py
│       ├── base.py          # BasePipelineModule基底クラス
│       ├── court_kp.py      # CourtKPModule
│       ├── gvhmr.py         # GVHMRModule + GVHMRConfig
│       ├── wasb.py          # WASBModule + WASBConfig
│       ├── trajectory.py    # TrajectoryModule
│       ├── event_uv.py      # EventUVModule
│       ├── plcs.py          # PLCSModule
│       ├── blcs.py          # BLCSModule
│       └── event_3d.py      # Event3DModule
├── configs/
│   └── pipeline.yaml        # Hydra設定
├── scripts/
│   └── run_pipeline.py      # エントリポイント
└── README.md                # このファイル
```
