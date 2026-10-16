# blender_export

保存済みsceneを1カメラ視点で書き出し、Blenderで「入力動画 → グリッド → コート → 選手」へ
切り替わる解説動画用ショットを撮影するための入口です。

- 書き出し: `python -m src.tennis_scene.scripts.export_blender`（設定 `configs/blender_export.yaml`）
- 撮影: `scripts/blender/tennis_reveal.py`（Blender 5.2同梱Pythonで実行。repoをimportしない）

## 入力の前提

- sceneは `player_smpl_valid`・`ball_3d_valid` を持つこと。無効frameは0埋めなのでmaskが唯一の区別手段であり、
  無ければ `KeyError` で停止する。
- カメラは `metadata.court_reference.camera_fits` のpinhole fit（歪み補正なし、主点=画像中心、fx=fy）を
  そのまま使う。3D復元はこのカメラ系と整合しているため、歪みを追加推定すると選手・ボールの重なりは悪化する
  （meiji cam1: ボール再投影中央値 2.7px → 歪みモデルでは 5.3px）。
- 書き出し前にボール再投影誤差の中央値を `max_ball_reprojection_median_px` と比較し、
  カメラとsceneの座標系不一致を拒否する。

## bundle

```text
<OUTPUT>/tennis_scene/visualize/blender_export/<run-id>/bundle/
├── manifest.json        # schema, カメラ(OpenCV/Blender), fps, 補間frame, コート形状
├── smpl_faces.npy
├── player_<p>.pc2       # court座標のSMPL頂点 (Mesh Cache modifier用)
├── player_valid.npy
├── ball_positions.npy   # 無効frameはNaN
├── ball_valid.npy
└── plate/%05d.jpg       # 選択カメラの元frame
```

Blenderのworld座標はcourt座標（m、XY地面、+Z上）そのもの。短い欠損は `player_max_gap_frames` /
`ball_max_gap_frames` 以下のみ線形補間し、補間したframeと残った無効frameをmanifestに列挙する
（無効frameは消灯する）。コート線・ネット形状は `src/utils/schema/court.py` から書き出し、
Blender側では寸法を持たない。

## 撮影

WSLからWindows版Blenderを使う場合、パスは `\\wsl.localhost\Ubuntu\...` 形式で渡す。

```bash
blender.exe --background --factory-startup --python scripts/blender/tennis_reveal.py -- \
  --bundle <bundle> --output <shot_dir> --render
ffmpeg -framerate 60000/1001 -i <shot_dir>/frames/%05d.png -c:v libx264 -crf 16 -pix_fmt yuv420p shot.mp4
```

`<shot_dir>/tennis_reveal.blend` を常に保存するので、見た目の調整はBlender GUIでも行える。
演出の仕組みとタイミング（`Shot` dataclass）はスクリプト冒頭のdocstringを参照。

検証済みの不変条件（meiji cam1で確認）:

- 切り替わり前のframeは元動画と一致する（frame 100: PSNR 55 dB）。plate連番のframe対応は
  `image_user.frame_offset = -1`。
- Blenderカメラの投影はOpenCV投影と0.0001px未満で一致する。
- Mesh Cacheの評価頂点はPC2と一致する（誤差0）。
