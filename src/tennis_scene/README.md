# `src/tennis_scene`

`src/tasks/ball_detection`, `src/tasks/court_detection`, `src/submodules`（GVHMR）, `src/tasks/plcs`, `src/tasks/blcs` をつないで、同期済みマルチカメラ動画から 1 つの `SceneResult` を組み立てる統合パイプラインです。カメラは固定（静止カメラ、カメラ回転推定なし）を前提とします。

再構成済み3D sceneを使った学習データ生成は、責務を分離した[`src/synthetic_data_generation`](../synthetic_data_generation/README.md)が担当します。

## Referenceモデルによる実クリップ検証

DINO → ViTPose → PLCSと外部ボール観測 → BLCSを接続する経路は
[`reference_pipeline/README.md`](reference_pipeline/README.md)を参照してください。
SMPLを実行せず、canonical poseと3D関節を保存できます。

PLCSのコート軌道とGVHMRのincam SMPLを従来フィールドへ保存し、それらを使った
GVHMRワールドモーションの整合もパイプライン内で常に実行します。整合結果は
`gvhmr_aligned_*`フィールドへ追加し、PLCS配置を上書きしません。下流は両方から用途に
合う表現を選べます。フィールド契約、設定値、残差診断は
[`motion_alignment/README.md`](motion_alignment/README.md)を参照してください。

## Modules

### dataset_pipeline/
実RGB clipから品質重み付き3D教師・DINOv3特徴・固定splitを作る経路。
1コマンド生成、入力と重み、品質判定、再開と再学習は
[`dataset_pipeline/README.md`](dataset_pipeline/README.md)を参照。

### clip_studio/
長時間・非同期のマルチカメラ動画を同期してラリークリップを切り出し、追記可能な構造化データセットへエクスポートするGUI。詳細は `clip_studio/README.md`。

### generate_dataset/
構造化クリップのうち未処理分へパイプラインを適用し、BLCS/PLCS用観測と3D出力を含む `SceneResult` を監査可能な疑似アノテーションとして追加する。詳細は `generate_dataset/README.md`。

### schema.py / archive.py
- **`schema.SceneResult`**: パイプライン共有スキーマ(`court_kp`/`player_position`/`player_yaw`/`smpl_*`/`ball_*`等)の唯一の定義。
- **`archive.save_scene_result()` / `load_scene_result()`**: `.npz` と必須 `*.metadata.json` サイドカーを明示的に保存・読込する唯一のarchive I/O。sidecar欠落・非object metadataはエラーにし、旧module/methodへ転送しない。

### pipeline/
- **`orchestrator.py`**: `TennisSceneOrchestrator`。全stageの構築・同期検証・実行・`SceneResult`組み立てを統括。
- **`dependency_graph.py`**: `PipelineDependencyGraph`。stage依存(`PLCS<-COURT_KP,GVHMR`等)の解決・循環検出。
- **`model_io/gvhmr.py`**: GVHMR chainの型付きrequest/result、検証adapter、composition factoryの唯一の定義。factoryがDINO/YOLOを一度だけ選択してsubmodule chainを構築し、adapterがvideo metadata・track・keypoints・boxes・features・SMPL keysを各model境界の前で検証する。
- **`components/court_kp.py`**: `CourtKPModule`。手動UIまたはモデル推論でコートkeypointを取得。
- **`components/gvhmr.py`**: `GVHMRModule`。composition rootから解決済みのGVHMR chainを受け取り、typed requestを渡すか保存済みresultを読む。model class、detector variant、tensor layout、raw output keyを認識しない。
- **`components/player_association.py`**: `PlayerAssociationModule`。カメラ間player対応付け(手動UI)を正準player軸へ整列。
- **`components/plcs.py`**: `PLCSModule`。task-owned multiview I/O adapterを持つpredictorへ観測を渡し、typed predictionをwindow集約する。
- **`components/ball_detection.py`**: `BallDetectionModule`。スライディングウィンドウ推論とオーバーラップ集約。
- **`components/blcs.py`**: `BLCSModule`。task-owned multiview I/O adapterを持つpredictorへ観測を渡し、typed predictionから3D軌道を集約する。

### rendering/
- **`tennis_scene_renderer.py`**: `TennisSceneRenderer`。SMPL/skeleton表示によるコート上3D可視化・動画保存。3D表示範囲はコート座標系に固定する。カメラ・テーマ・レイヤ規約・HUD・ミニマップなどの描画プリミティブは `src.utils.rendering`(`camera_view`/`theme`/`layers`/`hud`/`minimap`/`effects`)を直接利用し、ここには `SceneResult` 固有の変換(SMPL→コート座標、HUD行の選択、ミニマップ配列抽出)だけを持つ。

### scripts/
- **`run_pipeline.py`**: パイプライン実行エントリポイント。結果を `.npz` に保存。
- **`visualization.py`**: 保存済み `SceneResult` の3D可視化エントリポイント。
- **`visualize_tasks.py`**: stage別タスク動画(`plcs`/`gvhmr_alignment`/`blcs`等)を保存済み `SceneResult` から書き出すエントリポイント。`gvhmr_alignment`は`gvhmr_aligned_*`とPLCS配置を重ねる。
- **`clip_studio.py`**: クリップスタジオGUIの起動エントリポイント。
- **`export_clips.py`**: プロジェクトJSONからのヘッドレスクリップエクスポート。
- **`generate_dataset.py`**: 構造化データセットへの増分疑似アノテーション生成。

### configs/
- **`pipeline.yaml`**: stage別(`court_kp`/`gvhmr`/`player_association`/`player_motion`/`ball_detection`/`plcs`/`blcs`)の実行設定。整列は常時実行し、`player_motion.scale_mode`・`alignment`が推定方法を制御する。`court_keypoints.selector`と`court_reference`はPLCS/BLCSが共有するreference-frame設定であり、camera-view checkpointではcamera IDと各viewの半回転を明示する。
- **`visualization.yaml`**: 可視化スタイル・出力設定。`style`(テーマ・影・トレイル・HUD・ミニマップ)と `camera`(プリセット・mode・keyframes)を含む。
- **`clip_studio.yaml` / `export_clips.yaml` / `generate_dataset.yaml`**: クリップ編集・エクスポート・疑似アノテーション生成の設定。

## 座標系メモ

- `player_position` / `gvhmr_aligned_player_position` / `ball_3d`: コート座標系。XY平面が地面、+Zが上。
- `smpl_vertices_local` / `smpl_global_orient` / `smpl_body_pose`: GVHMR/SMPL由来の人体座標系。人体のup軸はY。
- 可視化時は、SMPL頂点をroot中心化した後に `src.utils.geometry.matrices.smpl_y_up_to_court_z_up` でY-upからコートZ-upへ明示変換し、その後 `player_yaw` をコート+Z軸まわりに適用する。
- `gvhmr_aligned_*`も既存レンダラーと同じ配置規則を使う。整列済みの4フィールドがworld頂点の直接相似変換を再現することの契約は[`motion_alignment/README.md`](motion_alignment/README.md)を参照。

## Courtモデル推論のKP・LINE共同推定

モデル実行は共通`CourtPredictor`のhybrid結果を使います。旧KP-only再推定・座標ごとのtemporal medianは適用しません。raw KP/scoreは診断へ残し、下流の`court_kp`はHによる再投影14点です。画像外座標をclipせず不可視とし、H失敗はゼロ座標＋全不可視にします。完全な14点を必要とするreference calibrationやfootpoint filterの条件は維持します。

既定のCourt・PLCS・BLCSを`camera_view_v2`へ統一しました。PLCSは`real-rgb-meiji-foot-e60-v1.ckpt`、BLCSは`real-rgb-meiji-e60-v1.ckpt`を使い、windowは128フレームです。いずれもMeiji実画像でfine-tuneした重みであり、他会場への精度を保証する評価ではありません。BLCSは3〜4台の同期カメラを要求します。

`pipeline.yaml`の動画パスは3台の例です。実動画と`camera_ids`を指定し、`court_reference.reference_camera`と`view_half_turns`を必ず設定してください。`view_half_turns`はcamera_ids順で、referenceは`false`、反対側のbaselineに向いたviewは`true`です。例えば向きが確認できた3台なら`court_reference.view_half_turns=[false,false,true]`と指定します。未設定・カメラ数不一致ではモデルロード前に停止します。共通predictorは各画像のcamera-view順を保ち、`court_reference`が一度だけreference-camera順へ変換します。

手動入力と`source=load`はモデル補正を通りません。旧physical順の入力・保存結果には`court_keypoints.selector=physical_v1`、対応する旧PLCS/BLCS重み、`court_reference.reference_camera=null`・`view_half_turns=null`を明示してください。新Court checkpointをphysical順として使うことは拒否します。 契約情報のない旧artifactをcamera-view順として読む場合だけ、内容の順序を確認したうえで`court_kp.load_keypoint_contract=camera_view_v2`を明示します。保存済み契約の上書きや元artifactの書換えは行いません。

新規Court結果と`SceneResult.metadata.court_detection`にはcheckpoint識別情報、後処理設定、入力KP schema、採用点、H生成可否を保存します。採用点のmask（最大8点）を再投影14点のvisibilityとして使うことはありません。
