# 選手視点の保存済みscene bundle

[English](README.md) · [简体中文](README.zh-CN.md)

保存済みSceneResult v2をCPUで読み、Blenderと3DGSの選手視点描画で共通利用する
bundleを書き出します。既存のmesh配置・PC2・コート幾何を再利用します。
後述の頭部カメラexporterはこのbundleだけを読み、描画は後続で行います。

```bash
.venv/bin/python -m src.tennis_scene.visualization.player_view \
  --input /absolute/path/to/store \
  --output outputs/tennis_scene/visualize/player_view/example/bundle \
  --smpl-faces /absolute/path/to/body_models/smplh/neutral/model.npz \
  --smpl-joint-regressor /absolute/path/to/body_models/gvhmr/smpl_neutral_J_regressor.pt \
  --smplx-model /absolute/path/to/body_models/smplx/SMPLX_NEUTRAL.npz \
  --frames 10 110 --player-ids 7 42
```

入力はNPZ＋必須metadata sidecar、store directory、`scene.json`です。
storeは`indexed_scene_path`だけで採用済みexportを解決し、checksumと現採用の依存鎖を検証します。
未完成・古いexportは拒否します。元動画・球再投影・推論は不要です。
[既存reveal](../blender/export/README.md)のplate・再投影検査は維持します。

資産は明示pathで渡します。faces、保存meshと一致する24行のSMPL regressor、
21 body jointの骨格を持つneutral SMPL-X NPZが必要です。数値fieldだけを読み、
モデルを初期化しません。派生した数値資産をbundleへ同梱するため、後続は元資産やstoreを必要としません。

区間は元frameの半開区間`[start,stop)`で、省略時は全区間です。人物指定はIDで行い、
指定順を保持します。省略時は全IDを元の順で保持します。人数は1人以上の任意数で、
未知・重複IDは拒否します。既存出力は拒否し、検証後に一時directoryへ生成してから完成版を公開します。
失敗時の一時出力は除去し、入力は読み取りのみです。

## スキーマ

`manifest.json`のschemaは`tennis_scene_player_view_bundle_v1`です。
`players`はbundle `index`・元`source_index`・`player_id`・PC2名・有効数・人物別`scale`を保持します。
`frames`はstart・排他的stop・元frame総数、`num_frames`・`fps`は元時間軸、
`arrays`は各配列の相対path・shape・dtypeを記録します。座標・SMPL・PC2・mask規約、
コート幾何、指定した選択設定、入力archive/metadataのdigestとstore採用export参照も保存します。
`config`内のnullは全入力を意味し、解決した区間・人物は`frames`・`players`に保存します。
`source_metadata.json`は配置設定・診断を含む元metadataです。

P=選択人数、F=stop-start、V=頂点数、N=面数。配列keyと`.npy` basenameは一致します。

| key/file | shape | 型・単位・意味 |
|---|---|---|
| `source_frames`, `source_time_seconds` | F / F | int64の元0始まりframe / float64のframe÷元FPS（秒） |
| `player_<index>.pc2` | F,V,3 | float32、配置済みmesh、m |
| `player_observed`, `player_valid`, `player_heading_valid`, `player_smpl_valid` | P,F | bool、元v2 mask |
| `player_kp_3d_vis` | P,F,17 | bool、元関節mask |
| `player_rejection_code`, `player_kp_3d_rejection_code` | P,F / P,F,17 | uint8、元理由 |
| `ball_positions`, `ball_3d_valid`, `ball_rejection_code` | F,3 / F / F | float32のm / bool / uint8 |
| `smpl_body_pose`, `smpl_betas` | P,F,63 / P,10 | float32、親関節相対axis-angle（rad）/ 無次元shape |
| `smpl_global_orient`, `player_yaw` | P,F,3 / P,F | float32、補償回転axis-angle / court +Z yaw、rad |
| `player_position`, `smpl_root_local` | P,F,3 | float32、court joint0並進 / local meshのrow0回帰root、m |
| `body_to_court_rotation` | P,F,3,3 | float32、復号した回転、列vector |
| `smpl_faces`, `smpl_joint_regressor` | N,3 / 24,V | int32面index / float32回帰重み |
| `smplx_parents` | 22 | int32、root親=-1 |
| `smplx_rest_joints`, `smplx_joint_shapedirs` | 22,3 / 22,3,10 | float32、unit scaleのSMPL-X rest joint（m）とshape基底 |

PC2はcache frame 0・sample rate 1で、sample iは`source_frames[i]`です。
再サンプル・欠測補間は行いません。mesh/球は対応maskがfalseなら非表示にし、座標0から有効性を推定しません。
無効PC2・球は0、mesh無効時の回転/rootも0です。他のscene配列は固有のmaskを保持します。
有効rootでもmesh有効とは限りません。

## 配置と頭部姿勢の境界

worldは右手系court座標（m、XY地面、+Z上、X幅・Y長さ方向、原点はネット中央直下の地面）。
local meshには人物scaleが適用済みです。
既存rendererと同じ式で、列vectorに次を適用します。

```text
A = [[1,0,0], [0,0,-1], [0,1,0]]
R = Rz(player_yaw) @ A @ Rodrigues(smpl_global_orient).T
world_vertex = R @ (saved_local_vertex - smpl_root_local) + player_position
```

`body_to_court_rotation`がRです。`smpl_global_orient`は生GVHMR incam回転ではなく
renderer用補償回転、`smpl.parameters.transl`は`player_position`を指すcourt joint0並進です。
元のincam translはv2に存在せず、捏造しません。

頭部回転はRと骨格に沿ったbody poseの親相対回転から求められます。
`smpl.body_joint_names`はroot＋21関節の名前で、joint jのposeはtriple j-1です。
骨格は共通の名前付きbody骨格と検証します。
shape付きrest jointは`smplx_rest_joints + einsum('jck,k->jc', smplx_joint_shapedirs, betas)`です。
頭部位置は配置済みmeshへ同梱SMPL regressorを適用できます。
SMPL-X FK関節と変換後SMPL meshはtopologyが違うため、位置を同一視しません。
unit scaleの骨格・offsetには保存scaleを使い、PC2へ二重適用しないでください。
有効meshを持つ人物は正の保存scale必須で、全frame無効の人物だけ`scale: null`を許します。

## 頭部カメラ

カメラ列は一度生成し、Blenderと3DGSへ同じ保存済み姿勢を渡します。
CPUでscene bundle v1だけを読み、元store・動画・body model・推論は不要です。
全人物の元ID・時間軸を保持します。出力先は入力bundle外の新しいdirectoryにします。
公開はatomicで、失敗時は一時出力を除去します。

```bash
.venv/bin/python -m src.tennis_scene.scripts.export_player_view_cameras \
  --bundle outputs/tennis_scene/visualize/player_view/example/bundle \
  --output outputs/tennis_scene/visualize/player_view/example/cameras \
  --width 1280 --height 720 --horizontal-fov-degrees 90 \
  --face-offset-head-m 0 0.08 0.12 --near-m 0.05 --far-m 100
```

### 設定と向き

上のコマンドが仮の既定値です。顔offsetは、meshから回帰したhead関節を原点とする
**unit scaleのSMPL頭部軸**でのm値です。人物scaleを掛ける前の値として指定します。
near/farは人物scaleに依存しない、courtのm単位のcamera +Z depthです。
全設定をmanifestへ保存します。正方画素・skew 0・主点`(width/2,height/2)`、
`fx=fy=width/(2*tan(horizontal_fov/2))`とし、レンズ歪みはありません。

頭部を選ぶ前に名前付き骨格とmesh関節順を検証します。
joint jはpose triple j-1に対応し、回転の順は
`spine1 → spine2 → spine3 → neck → head`です。列vectorで次の式を使います。

```text
H = body_to_court_rotation @ R_spine1 @ R_spine2 @ R_spine3 @ R_neck @ R_head
head_position = smpl_joint_regressor[mesh_head_row] @ placed_PC2_vertices
camera_center = head_position + H @ (player_scale * face_offset_head_m)
camera_to_court[:3,:3] = H @ diag(-1,-1,1)
camera_to_court[:3,3] = camera_center
```

SMPL/SMPL-Xのneutral頭部は+Yが上、+Zが顔正面、+Xが本人の左です。
OpenCV cameraの+X右・+Y下・+Z前は、頭部の−X・−Y・+Zに対応します。
正面軸は上流の[vertex ID](https://github.com/vchoutas/smplx/blob/main/smplx/vertex_ids.py)に従う
neutralモデルの鼻・眼・耳の位置で確認しました。保存済み63値のposeは、線形な
SMPL-X→SMPL topology変換前に使われたSMPL-Xのbody関節回転であり、SMPLの69値poseではありません。
回転FKはその骨格、位置は**変換・配置済みSMPL mesh**を使います。
SMPL-XのFK位置による代用、PC2の再中心化・二重scale、encoded global orientationを
頭部回転として使うことはしません。

元frameごとの静止ピンホールでrollを保持します。追加の平滑化・補間・安定化・
ボール注視は行いません。頭部姿勢によってコート外を向いたり、相手や球が画面外になることはあります。

### カメラ出力schema

`manifest.json`のschemaは`tennis_scene_player_view_head_cameras_v1`です。
入力bundle参照・使用ファイルのdigest、元frame区間、FPS、ID/index対応、人物scale、
有効数・無効理由の件数、全設定、軸・投影、頭部の方式、欠測方針を記録します。
各`players[index].path`のNPZは単体で読めます（F frame）。

| Key | Shape | 型・意味 |
|---|---|---|
| `K` | 3,3 | float64、固定内部行列、pixel |
| `camera_to_court` | F,4,4 | float64、OpenCV camera→court同次変換、並進m |
| `camera_valid` | F | bool、姿勢を使う前に必ず確認 |
| `camera_rejection_code` | F | uint8、下記の理由 |
| `head_position` | F,3 | float64、配置済みmeshから回帰したhead位置、m |
| `source_frames` | F | int64、元の0始まりframe番号 |
| `source_time_seconds` | F | float64、元frame/FPS |
| `source_player_rejection_code` | F | uint8、入力の人物理由を保持、camera理由とは独立 |

有効なTでは`X_camera=T[:3,:3].T @ (X_court-T[:3,3])`をKで投影します。
+Zがdepthです。画面・near/farのclipは保存設定による描画処理で、姿勢を変えません。
無効な変換とhead位置は**全要素NaN**で、単位行列や0姿勢ではありません。Kは常に定義します。

理由は優先順に`0 VALID`、`1 MESH_INVALID`、`2 HEAD_POSE_NONFINITE`、
`3 PLACEMENT_ROTATION_INVALID`、`4 HEAD_POSITION_NONFINITE`です。
mesh欠測maskは0埋め値の検査より優先し、有限な正規回転と頭部chainのposeが必要です。
骨格・regressor・時間軸の不整合はexport全体を拒否します。
全frame欠測の人物だけscale nullを許し、全cameraを無効にします。
身体yaw、前後frame、固定カメラへのfallbackはありません。

### カメラの読み方

```python
from pathlib import Path
from src.tennis_scene.visualization.player_view.camera_bundle import load_head_cameras
from src.tennis_scene.visualization.blender.export.camera import to_blender_camera

saved = load_head_cameras(Path("outputs/tennis_scene/visualize/player_view/example/cameras"), player_id=7)
index = 0  # cache index。元frameはsaved.source_frames[index]
if saved.series.camera_valid[index]:
    camera = saved.series.pinhole(index)
    blender_pose = to_blender_camera(camera, sensor_width_mm=36.0)
```

readerは保存済み変換・maskを検証し、入力bundleを必要としません。
`pinhole()`は無効sampleを拒否するため、描画側は時間枠を保持して欠測表示frameにします。
Blenderのcamera回転は`T[:3,:3] @ diag(1,-1,-1)`、位置は`T[:3,3]`です。
保存した解像度・内部行列・clip値を使い、両backendで頭部運動を再計算しません。
## Blenderの選手視点動画

対応する保存済みscene/camera bundleからportableなshot入力を作ります。cameraの入力digest・人物ID・元frame・FPS・maskを照合し、必要な数値資産をコピーします。Blender側は同梱PythonとNumPyだけで動き、camera再推定やrepoのimportを行いません。

```bash
.venv/bin/python -m src.tennis_scene.scripts.prepare_player_view_blender \
  --bundle outputs/tennis_scene/visualize/player_view/example/bundle \
  --cameras outputs/tennis_scene/visualize/player_view/example/cameras \
  --output outputs/tennis_scene/visualize/player_view/example/shot

blender --background --factory-startup --python-exit-code 1 --threads 4 \
  --python src/tennis_scene/visualization/blender/shots/player_view.py -- \
  --package outputs/tennis_scene/visualize/player_view/example/shot \
  --output outputs/tennis_scene/visualize/player_view/example/renders/player_7 \
  --player-id 7 --gpu-backend OPTIX

.venv/bin/python -m src.tennis_scene.scripts.publish_player_views \
  --package outputs/tennis_scene/visualize/player_view/example/shot \
  --renders outputs/tennis_scene/visualize/player_view/example/renders \
  --output outputs/tennis_scene/visualize/player_view/example/videos
```

Blenderコマンドは元の人物IDごとに実行し、出力名を`renders/player_<ID>`とします。解像度・内部行列・clip・元FPSは保存cameraを使います。仮の外観はmanifestへ記録します。本人のmesh全体を隠し、相手は灰色、球は緑色・半径0.05m、alpha=1・発光0、簡素なコート/ネット、Cycles 16sampleです。OPTIX/CUDA GPUを明示使用し、CPUへのfallbackは拒否します。実行資源はプロジェクトの運用手順に従います。

Blender/cache frame `i`は`source_frames[i]`と厳密に対応します。PC2は明示sampleで補間なし、camera/球はframeごとに直接設定し、animation curveとmotion blurを使いません。物体欠測は非表示、camera欠測は同じ長さの空白frameとして描画し、公開時に欠測表示を入れます。`render.json`は完成PNGごとのPC2評価誤差・camera姿勢・物体表示・球位置・GPU・材質を記録します。

初回の少数frame確認はBlenderコマンドへ`--source-frames <最大12個の元frame番号>`を明示し、動画化に`--preview`を付けます。最終動画化では入力時間軸の切り詰めを拒否します。描画記録を照合し、各画面へ人物ID・元frame・camera有効性を表示して、人物別MP4と左右同期比較を生成します。全frameをdecodeして解像度・件数・FPS・長さを確認し、manifestへ入力・設定とcamera欠測frameを保存します。shot入力と描画記録も動画と一緒に保持してください。既存出力は上書きしません。

描画・検証・CIが成功しても`human_acceptance: pending`のままです。第2フェーズの統合前に、人間の定性受入を別途記録します。
