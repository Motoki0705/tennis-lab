# blender_export

保存済みSceneResult v2と元動画から、Blenderで「入力映像 → グリッド → コート → 選手・ボール」へ
切り替わり、最後にカメラが移動するアニメーションを生成します。

- 書き出し: `python -m src.tennis_scene.scripts.export_blender`（設定 `configs/blender_export.yaml`）
- 撮影: `scripts/blender/tennis_reveal.py`（Blender 5.2同梱Python。repoをimportしない）
- Blender上の検証: `scripts/blender/validate_reveal.py`

## 入力と書き出し

`input`はARTIFACT root相対のscene archive、component storeのdirectory、または`scene.json`です。
標準のarchive readerで検証したv2 sceneの`player_smpl_valid`・`ball_3d_valid`を使います。
有効性maskを持たない旧sceneを推測で補完しません。

```bash
.venv/bin/python -m src.tennis_scene.scripts.export_blender \
  input=tennis_scene/evaluate/<experiment>/<run>/scene.npz camera_id=cam1
```

worktreeから共有データを使う場合は`paths.project_root=<main repoの絶対パス>`を明示します。
SMPL facesはCHECKPOINT root、regressorはPROJECT rootから解決します。
元動画はscene metadataの`camera_ids`と`video_paths`を対応させて読み、解像度・FPS・フレーム数の一致を要求します。

カメラは`metadata.court_reference.camera_fits`のpinhole fit（歪みなし、主点=画像中心、fx=fy）を使います。
書き出し前に有効な3D球と2D観測の再投影誤差中央値を`max_ball_reprojection_median_px`と比較し、
カメラとsceneの不一致を拒否します。検証できる球観測がないsceneも拒否します。

## bundle

```text
<OUTPUT>/tennis_scene/visualize/blender_export/<run-id>/bundle/
├── manifest.json        # schema、カメラ、FPS、補間frame、コート形状、出典
├── smpl_faces.npy
├── player_<p>.pc2       # コート座標のSMPL頂点（Mesh Cache用）
├── player_valid.npy
├── ball_positions.npy   # 無効frameはNaN
├── ball_valid.npy
└── plate/%05d.jpg       # 元動画のframe 0からの連番
```

world座標はコート座標（m、XY地面、+Z上）です。rendererと同じSMPL配置変換を使います。
`player_max_gap_frames` / `ball_max_gap_frames`以下の内側の欠損だけを線形補間し、
補間frameと残った欠損をmanifestに記録します。残った欠損はBlenderで非表示にし、球の軌跡も欠損を跨ぎません。
コート線・ネット形状の正本は`src/utils/schema/court.py`で、bundleに書き出します。
既存bundleは上書きせず、途中失敗の出力は削除します。

## 撮影・動画化

GPUを使うレンダリングは共有training queue経由で実行します。
Linux版Blender、またはWindows版BlenderにWSLのパスを渡す場合は`\\wsl.localhost\Ubuntu\...`形式を使います。
Windowsで作成した`.blend`の外部参照はWindows形式になるため、同じOS側のBlenderで開いてください。

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender/tennis_reveal.py -- \
  --bundle <bundle> --output <shot_dir> --render
# fpsはmanifest.jsonの値に合わせる。例は59.94fpsの入力。
ffmpeg -framerate 60000/1001 -start_number 0 -i <shot_dir>/frames/%05d.png \
  -c:v libx264 -crf 16 -pix_fmt yuv420p -movflags +faststart <shot_dir>/reveal.mp4
```

`tennis_reveal.blend`には出力先と描画範囲も保存します。`.blend`はbundleのPC2とplate連番を参照するため、
bundleも一緒に保持してください。`--render`を外すとシーン作成・保存だけを実行します。
`--frames 100:100`で部分描画、`--samples 8 --resolution-percentage 50`で軽量なプレビューができます。
部分描画の動画化では`-start_number`も開始frameに合わせます。

色・切り替えのタイミング・カメラ終点は撮影スクリプトの`Shot`にまとまっています。
タイミングは入力のframe単位で、既定では150から切り替わり、420の後にカメラが移動します。
短いclipではこの範囲に合わせて`Shot`を調整してください。スクリプトを再実行するとシーンを作り直します。

## 検証

```bash
blender --background <shot_dir>/tennis_reveal.blend --python-exit-code 1 \
  --python scripts/blender/validate_reveal.py -- \
  --bundle <bundle> --report <shot_dir>/validation.json
```

Blenderで評価したmesh頂点とPC2、球の位置、全frameの有効性と非表示設定、入力カメラとの投影一致、
FPSとカメラ移動を検査します。不一致は非ゼロ終了し、成功時にJSONの検証結果を保存します。
映像側は切り替え前のframeと元動画、切り替え中、自由視点の各場面も確認してください。
