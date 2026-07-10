# `src/tennis_scene`

`src/tasks/ball_detection`, `src/tasks/court_detection`, `src/submodules`（GVHMR）, `src/tasks/plcs`, `src/tasks/blcs` をつないで、同期済みマルチカメラ動画から 1 つの `SceneResult` を組み立てる統合パイプラインです。カメラは固定（静止カメラ、カメラ回転推定なし）を前提とします。

## Modules

### io.py
- **`SceneResult`**: パイプライン共有スキーマ(`court_kp`/`player_position`/`player_yaw`/`smpl_*`/`ball_*`等)。
- **`save()`/`load()`**: `.npz`+`*.metadata.json`サイドカーで保存、pickle混入legacy形式は警告付きフォールバック。

### pipeline/
- **`orchestrator.py`**: `TennisSceneOrchestrator`。全stageの構築・同期検証・実行・`SceneResult`組み立てを統括。
- **`dependency_graph.py`**: `PipelineDependencyGraph`。stage依存(`PLCS<-COURT_KP,GVHMR`等)の解決・循環検出。
- **`components/court_kp.py`**: `CourtKPModule`。手動UIまたはモデル推論でコートkeypointを取得。
- **`components/gvhmr.py`**: `GVHMRModule`。`src/submodules/models` の GVHMR チェーン（tracking→ViTPose→HMR2→GVHMR）をサブプロセスまたは直接実行しSMPL/2D poseを取得。重みは `ckpt/` symlink 経由。
- **`components/player_association.py`**: `PlayerAssociationModule`。カメラ間player対応付け(手動UI)を正準player軸へ整列。
- **`components/plcs.py`**: `PLCSModule`。`court_kp`をplayer数へexpandしPLCS predictorへ渡す。
- **`components/ball_detection.py`**: `BallDetectionModule`。スライディングウィンドウ推論とオーバーラップ集約。
- **`components/blcs.py`**: `BLCSModule`。複数カメラのball観測から3D軌道を推定。

### rendering/
- **`tennis_scene_renderer.py`**: `TennisSceneRenderer`。SMPL/skeleton表示によるコート上3D可視化・動画保存。3D表示範囲はコート座標系に固定する。カメラ・テーマ・レイヤ規約・HUD・ミニマップなどの描画プリミティブは `src.utils.rendering`(`camera_view`/`theme`/`layers`/`hud`/`minimap`/`effects`)を直接利用し、ここには `SceneResult` 固有の変換(SMPL→コート座標、HUD行の選択、ミニマップ配列抽出)だけを持つ。

### utils/
- **`transforms.py`**: `src.utils.geometry` からのthin re-export(このパイプライン内からは未使用)。

### scripts/
- **`run_pipeline.py`**: パイプライン実行エントリポイント。結果を `.npz` に保存。
- **`visualization.py`**: 保存済み `SceneResult` の可視化エントリポイント。

### configs/
- **`pipeline.yaml`**: stage別(`court_kp`/`gvhmr`/`player_association`/`ball_detection`/`plcs`/`blcs`)の実行設定。
- **`visualization.yaml`**: 可視化スタイル・出力設定。`style`(テーマ・影・トレイル・HUD・ミニマップ)と `camera`(プリセット・mode・keyframes)を含む。

## 座標系メモ

- `player_position` / `ball_3d`: コート座標系。XY平面が地面、+Zが上。
- `smpl_vertices_local` / `smpl_global_orient` / `smpl_body_pose`: GVHMR/SMPL由来の人体座標系。人体のup軸はY。
- 可視化時は、SMPL頂点をroot中心化した後に `src.utils.geometry.matrices.smpl_y_up_to_court_z_up` でY-upからコートZ-upへ明示変換し、その後 `player_yaw` をコート+Z軸まわりに適用する。
