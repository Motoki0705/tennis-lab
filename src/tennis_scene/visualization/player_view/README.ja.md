# 選手視点の保存済みscene bundle

[English](README.md) · [简体中文](README.zh-CN.md)

保存済みSceneResult v2をCPUで読み、Blenderと3DGSの選手視点描画で共通利用する
bundleを書き出します。既存のmesh配置・PC2・コート幾何を再利用します。
頭部カメラの計算・描画は後続の責務です。

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

頭部回転・meshからの頭部位置に必要な情報を保持します。
頭部カメラ、正面軸、顔offset、カメラ欠測規約の決定は後続の担当です。
