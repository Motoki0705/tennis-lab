# Tennis Scene 3D Reconstruction

単眼動画からテニスシーンを3D再構築するための統合パイプライン。

## 概要

このモジュールは以下のコンポーネントを統合します：

- **Court KP Detection**: コートキーポイント検出（固定カメラ前提で1フレームのみ）
- **GVHMR**: 3D人物メッシュ（ローカルSMPL）+ 2Dスケルトン
- **Player Association**: カメラ間プレーヤー対応付け（手動UI）
- **Ball Detection**: `src.tasks.ball_detection` によるボール2D検出
- **PLCS**: プレーヤー3D位置 + yaw推定
- **BLCS**: ボール3D軌道推定

## アーキテクチャ

オーケストレーション型のモジュラー設計を採用しています：

```
TennisSceneOrchestrator (オーケストレーター)
├── CourtKPModule          # コートKP検出
├── GVHMRModule            # 3D人物メッシュ推定（CLIサブプロセス経由）
├── PlayerAssociationModule # カメラ間プレーヤー対応付け
├── BallDetectionModule    # ボール検出
├── PLCSModule             # プレーヤー3D位置推定
└── BLCSModule             # ボール3D軌道推定
```

各モジュールは独立して設定・ロード可能で、`BasePipelineModule` を継承しています。パイプラインは合計 **6コンポーネントモジュール**を持ちます。

### ステージ依存関係

- `PLAYER_ASSOCIATION <- GVHMR`
- `PLCS <- COURT_KP, GVHMR, PLAYER_ASSOCIATION`
- `BLCS <- COURT_KP, BALL_DETECTION`

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
    court_kp.annotation_frame_index=0 \
    output_dir=outputs/tennis_scene
```

### Court KPをUIで入力

手動入力UIを使う場合は `manual_ui` を指定します。結果JSONは `court_kp.output_path` に保存されます。
手動入力された1フレーム分のCourt KPは、対象フレーム列へ展開されます。

```bash
python -m src.tennis_scene.scripts.run_pipeline \
    video_path=inputs/demo/match.mp4 \
    court_kp.mode=manual_ui \
    court_kp.output_path=outputs/tennis_scene/court_kp_sequence_result.json
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
| `court_kp.annotation_frame_index` | Court KP推論/手動入力フレーム番号 | `0` |
| `court_kp.mode` | Court KPの入力モード（`model`/`manual_ui`） | `manual_ui` |

### GVHMR設定

| キー | 説明 | デフォルト |
|------|------|----------|
| `gvhmr.checkpoint` | GVHMRモデル | `third_party/GVHMR/inputs/checkpoints/gvhmr/...` |
| `gvhmr.yolo_checkpoint` | YOLOトラッカー | `third_party/GVHMR/inputs/checkpoints/yolo/yolov8x.pt` |
| `gvhmr.vitpose_checkpoint` | ViTPose | `third_party/GVHMR/inputs/checkpoints/vitpose/...` |
| `gvhmr.hmr2_checkpoint` | HMR2特徴抽出 | `third_party/GVHMR/inputs/checkpoints/hmr2/...` |
| `gvhmr.skip` | GVHMRスキップ | `false` |
| `gvhmr.load_path` | キャッシュ済み結果のロードパス（推論スキップ） | `null` |

### Player Association設定

| キー | 説明 | デフォルト |
|------|------|----------|
| `player_association.mode` | 対応付けモード（`manual_ui`） | `manual_ui` |
| `player_association.initial_frame_index` | UI起動時の初期フレーム | `0` |
| `player_association.reference_camera` | SMPL配列の基準カメラID | `cam0` |
| `player_association.load_path` | キャッシュ済み結果のロードパス（UI スキップ） | `null` |

### Ball Detection/PLCS/BLCS設定

| キー | 説明 | デフォルト |
|------|------|----------|
| `ball_detection.checkpoint` | `src.tasks.ball_detection` のLightning checkpoint | `outputs/ball_detection/...` |
| `ball_detection.batch_size` | ボール検出の推論バッチサイズ | `4` |
| `ball_detection.image_size` | ボール検出モデル入力サイズ `[height, width]` | `[288, 512]` |
| `ball_detection.normalize_imagenet` | ImageNet正規化を適用 | `true` |
| `ball_detection.score_threshold` | 可視判定に使うピークスコア閾値 | `0.5` |
| `ball_detection.prefetch_batches` | 推論前にCPU側で準備しておくバッチ数 | `2` |
| `ball_detection.window_stride` | 時系列windowのstride。`null`ならモデルの`num_frames` | `null` |
| `ball_detection.tail_policy` | 末尾windowの扱い。`backfill`は最後のフレームで終わるwindowを作る | `backfill` |
| `ball_detection.overlap_aggregation` | 重複推論されたフレームの集約方法 | `last_window_wins` |
| `ball_detection.pin_memory` | 推論前バッチをpin memory化 | `true` |
| `ball_detection.skip` | ボール検出スキップ | `false` |
| `plcs.checkpoint` | PLCSモデル | `checkpoints/plcs/frame/last.ckpt` |
| `plcs.load_path` | キャッシュ済み結果のロードパス（推論スキップ） | `null` |
| `blcs.checkpoint` | BLCSモデル | `checkpoints/blcs/single/last.ckpt` |
| `blcs.skip` | BLCSスキップ | `false` |
| `blcs.load_path` | キャッシュ済み結果のロードパス（推論スキップ） | `null` |

## 出力

出力ファイル: `{output_dir}/{video_name}.npz`

`SceneResult` データ構造：

```python
SceneResult:
    num_frames: int              # フレーム数
    fps: float                   # FPS
    width: int                   # 動画幅
    height: int                  # 動画高さ
    court_kp: (N, T, K, 2)      # コートKP（正規化）  N=カメラ数, K=キーポイント数
    court_vis: (N, T, K)        # コートKP可視性
    player_position: (P, T, 3)  # プレーヤー3D位置（メートル）  P=プレーヤー数
    player_yaw: (P, T)          # プレーヤーyaw（ラジアン）
    smpl_body_pose: (P, T, 63)  # SMPLボディポーズ
    smpl_global_orient: (P, T, 3)  # SMPLグローバル向き
    smpl_betas: (P, 10)         # SMPL形状パラメータ
    smpl_vertices_local: (P, T, V, 3)   # ローカルSMPL頂点（optional）
    ball_uv: (N, T, 2)          # ボール2D位置（正規化）（optional）
    ball_visibility: (N, T)     # ボール可視性（optional）
    ball_3d: (T, 3)             # ボール3D位置（メートル）（optional）
    human_kp_2d: (P, N, T, 17, 2)  # 人物2DKP（正規化）（optional）
    human_kp_vis: (P, N, T, 17)    # 人物KP可視性（optional）
```

## 座標変換

### `tennis_scene/utils/transforms.py::apply_plcs_transform`（SMPL頂点の変換ユーティリティ）

`apply_plcs_transform` / `apply_plcs_transform_batch` は GVHMR/SMPL ローカル座標系での変換です：

1. PLCS の yaw を **Y軸回転**（`rotation_matrix_y`）として適用
2. PLCS の 3D 位置を加算

これは GVHMR が出力するローカルSMPLフレーム（Y上方向の人物中心座標系）での変換を想定しています。

### `tennis_scene/rendering/tennis_scene_renderer.py::_build_players_smpl_vertices_court`（レンダラーでのSMPL頂点復元）

レンダラーは SceneResult から頂点をコート座標系に復元するため、独立した変換パイプラインを持ちます：

1. `smpl_vertices_local` から root を差し引き、メッシュを root 基準に中心化
2. `smpl_global_orient` の逆回転で GVHMR 由来の向きを打ち消し（axis-angle → matrix）
3. PLCS の yaw を **Z軸回転**（`_rotation_matrix_z`）として適用
4. PLCS の 3D 位置を加算

レンダラーはコート平面が XY 平面（Z上方向）の座標系を前提とし、yaw をZ軸周りの回転として適用しています。

> **既知の不整合 (known inconsistency)**: `apply_plcs_transform` は Y軸回転、レンダラーは Z軸回転を使用しており、回転軸が一致しません。前者は GVHMR/SMPL ローカルフレーム（Y上方向）、後者はコートフレーム（Z上方向）の慣習に基づくと考えられますが、コードレベルでの整合性確認は未完了です。`apply_plcs_transform` はレンダラーからは呼ばれておらず、両者は独立したパスです。

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
│       ├── player_association.py  # PlayerAssociationModule + PlayerAssociationConfig
│       ├── ball_detection.py # BallDetectionModule + BallDetectionConfig
│       ├── plcs.py          # PLCSModule
│       └── blcs.py          # BLCSModule
├── rendering/
│   └── tennis_scene_renderer.py  # TennisSceneRenderer
├── configs/
│   └── pipeline.yaml        # Hydra設定
├── scripts/
│   └── run_pipeline.py      # エントリポイント
└── README.md                # このファイル
```
