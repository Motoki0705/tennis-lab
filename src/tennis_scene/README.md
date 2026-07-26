# `src/tennis_scene`

`src/tasks/ball_detection`, `src/tasks/court_detection`, `src/submodules`（GVHMR）, `src/tasks/plcs`, `src/tasks/blcs` をつないで、同期済みマルチカメラ動画から 1 つの `SceneResult` を組み立てる統合パイプラインです。カメラは固定（静止カメラ、カメラ回転推定なし）を前提とします。

再構成済み3D sceneを使った学習データ生成は、責務を分離した[`src/synthetic_data_generation`](../synthetic_data_generation/README.md)が担当します。

## Modules

### clip_studio/
長時間・非同期のマルチカメラ動画を同期してラリークリップを切り出し、追記可能な構造化データセットへエクスポートするGUI。詳細は `clip_studio/README.md`。

### generate_dataset/
構造化クリップのうち未処理分へパイプラインを適用し、BLCS/PLCS用観測と3D出力を含む `SceneResult` を監査可能な疑似アノテーションとして追加する。詳細は `generate_dataset/README.md`。

### io.py
- **`SceneResult`**: パイプライン共有スキーマ(`court_kp`/`player_position`/`player_yaw`/`smpl_*`/`ball_*`等)。
- **`save()`/`load()`**: `.npz`+`*.metadata.json`サイドカーで保存、pickle混入legacy形式は警告付きフォールバック。

### pipeline/
- **`orchestrator.py`**: `TennisSceneOrchestrator`。全stageの構築・同期検証・実行・`SceneResult`組み立てを統括。
- **`dependency_graph.py`**: `PipelineDependencyGraph`。stage依存(`PLCS<-COURT_KP,GVHMR`等)の解決・循環検出。
- **`components/court_kp.py`**: `CourtKPModule`。手動UIまたはモデル推論でコートkeypointを取得。
- **`components/gvhmr.py`**: `GVHMRModule`。`src/submodules/models` の GVHMR チェーン（YOLOまたはDINO person検出→BoT-SORT→ViTPose→HMR2→GVHMR）を同一プロセスで実行しSMPL/2D poseを取得。`gvhmr.detector` で `yolo` / `dino` を明示選択する。
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
- **`clip_studio.py`**: クリップスタジオGUIの起動エントリポイント。
- **`export_clips.py`**: プロジェクトJSONからのヘッドレスクリップエクスポート。
- **`generate_dataset.py`**: 構造化データセットへの増分疑似アノテーション生成。

### configs/
- **`pipeline.yaml`**: stage別(`court_kp`/`gvhmr`/`player_association`/`ball_detection`/`plcs`/`blcs`)の実行設定。
- **`visualization.yaml`**: 可視化スタイル・出力設定。`style`(テーマ・影・トレイル・HUD・ミニマップ)と `camera`(プリセット・mode・keyframes)を含む。
- **`clip_studio.yaml` / `export_clips.yaml` / `generate_dataset.yaml`**: クリップ編集・エクスポート・疑似アノテーション生成の設定。

## 座標系メモ

- `player_position` / `ball_3d`: コート座標系。XY平面が地面、+Zが上。
- `smpl_vertices_local` / `smpl_global_orient` / `smpl_body_pose`: GVHMR/SMPL由来の人体座標系。人体のup軸はY。
- 可視化時は、SMPL頂点をroot中心化した後に `src.utils.geometry.matrices.smpl_y_up_to_court_z_up` でY-upからコートZ-upへ明示変換し、その後 `player_yaw` をコート+Z軸まわりに適用する。
