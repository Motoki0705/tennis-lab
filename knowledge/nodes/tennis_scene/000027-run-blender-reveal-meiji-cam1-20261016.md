---
id: run-blender-reveal-meiji-cam1-20261016
type: run
task: tennis_scene
sequence: 27
recorded_at: '2026-10-16'
title: Meiji clip_000/cam1のBlender再構成アニメーション再生成・整合性検証
provider: codex
session: 01a14329-6c48-74f2-ad36-640189a81fcb
date: '2026-10-16'
status: done
config:
  scene_schema_version: 2
  clip: video_000/clip_000
  camera_id: cam1
  blender_version: 5.2.1 LTS
  renderer: llvmpipe (LLVM 20.1.2, 256 bits)
  threads: 4
  render_samples: 4
  resolution_percentage: 50
  frame_step: 4
  player_max_gap_frames: 12
  ball_max_gap_frames: 3
metrics:
  source_frames: 1010
  rendered_frames: 253
  video_width: 960
  video_height: 540
  video_fps: 14.985014985014985
  video_duration_s: 16.883533
  mesh_frames_checked: 9
  max_mesh_error_m: 0.0
  max_ball_error_m: 0.0
  max_camera_projection_error_px: 0.00013076003665446478
  visibility_matches_masks: true
  source_plate_psnr_db: 43.13495005467417
repro:
  commit: 8549c66c05abd9e2b3f73b785cc6382603cfb7b5
  branch: codex/blender-scene-reveal
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 LIBGL_ALWAYS_SOFTWARE=1
    GALLIUM_DRIVER=llvmpipe /home/kamimura/.cache/tennis-lab/blender/blender-5.2.1-linux-x64/blender
    --background --factory-startup --python-exit-code 1 --threads 4 --python scripts/blender/tennis_reveal.py
    -- --bundle /home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/blender_export/pr1203-meiji-cam1/bundle
    --output /home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/hook_reveal/pr1203-meiji-cam1-preview
    --samples 4 --resolution-percentage 50 --frame-step 4 --render && env OMP_NUM_THREADS=4
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
    /home/kamimura/.cache/tennis-lab/blender/blender-5.2.1-linux-x64/blender --background
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/hook_reveal/pr1203-meiji-cam1-preview/tennis_reveal.blend
    --python-exit-code 1 --threads 4 --python scripts/blender/validate_reveal.py --
    --bundle /home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/blender_export/pr1203-meiji-cam1/bundle
    --report /home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/hook_reveal/pr1203-meiji-cam1-preview/validation.json
    && ffmpeg -v error -framerate 15000/1001 -pattern_type glob -i '/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/hook_reveal/pr1203-meiji-cam1-preview/frames/*.png'
    -c:v libx264 -crf 16 -pix_fmt yuv420p -movflags +faststart /home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/hook_reveal/pr1203-meiji-cam1-preview/reveal_preview.mp4
    && ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,duration:format=duration,size
    -of json /home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/hook_reveal/pr1203-meiji-cam1-preview/reveal_preview.mp4
    > /home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/hook_reveal/pr1203-meiji-cam1-preview/video_metadata.json
artifacts:
  run_dir: knowledge/runs/run-blender-reveal-meiji-cam1-20261016
  log: knowledge/runs/run-blender-reveal-meiji-cam1-20261016/render.log
  output_dir: outputs/tennis_scene/visualize/hook_reveal/pr1203-meiji-cam1-preview
  validation: knowledge/runs/run-blender-reveal-meiji-cam1-20261016/validation.json
  input_provenance: knowledge/runs/run-blender-reveal-meiji-cam1-20261016/input_provenance.json
  preview_gif: assets/tennis_scene/blender_reveal_meiji_cam1.gif
  video: outputs/tennis_scene/visualize/hook_reveal/pr1203-meiji-cam1-preview/reveal_preview.mp4
  blend: outputs/tennis_scene/visualize/hook_reveal/pr1203-meiji-cam1-preview/tennis_reveal.blend
  contact_sheet: {status: deleted, record: knowledge/runs/run-blender-reveal-meiji-cam1-20261016/artifact-manifest.json#/files/0}
parents:
- run-i931-default-meiji-clip000-20260927
relations: []
papers: []
tags:
- blender
- rendering
- integration_validation
- pr1203
---

## 観測と受入

[既存のMeiji scene](000022-run-i931-default-meiji-clip000-20260927.md)を再利用し、SceneResult v2 → Blender bundle → `.blend` → PNG → MP4の経路を完走した。3D推論は再実行していない。
元の1010frameを4frameごとに描画した253枚（source frame 0〜1008）から、960×540・15000/1001fps・16.883533秒のMP4を生成した。全編をffmpegの`-xerror`付きでデコードできた。
[PR用GIF](../../../assets/tennis_scene/blender_reveal_meiji_cam1.gif)は同じMP4から400×225・8fpsへ縮小した。

保存した`.blend`を別プロセスで開き、`scripts/blender/validate_reveal.py`で1010frameの球位置・選手/球/軌跡の表示maskを検査した。選手meshは0,126,252,378,504,630,756,882,1009の9frameでBlender評価後の頂点をPC2と比較した。mesh・球とも座標差は0 m、移動前カメラの最大投影差は0.000131 pxだった。この差は**書き出したデータとの一致**であり、真の3D位置との誤差ではない。

元動画のframe 100をOpenCV INTER_AREAで960×540へ縮小し、描画frame 100と照合したPSNRは43.135 dB。前後の99/101frameでは37.744/37.499 dBであり、この時点のフレーム対応も確認した。
[代表6frame（削除済み）](../../runs/run-blender-reveal-meiji-cam1-20261016/artifact-manifest.json#/files/0)で、元映像、グリッド/コートの切り替え、選手と球軌跡、自由視点への移動を目視確認した。

## 入力と再現

scene・動画・SMPL faces資産・regressor・bundle manifestのSHA-256は`input_provenance.json`に記録した。元sceneのSMPL有効数は1999/2020 player-frame、球は987/1010frame。既定の内側欠損補間で選手21frame、球17frameを補完し、球の末尾1004〜1009は非表示のまま保持した。補間したframeは`bundle_manifest.json`に列挙している。
カメラの入力整合性検査は973球観測の再投影誤差中央値2.673 pxで通過した。これは人手3D基準での精度ではない。

bundleを生成したコマンド（作業checkoutのrootから実行）:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m src.tennis_scene.scripts.export_blender \
  paths.project_root=/home/kamimura/projects/tennis-lab \
  input=tennis_scene/evaluate/i931-default-meiji-clip000-20260927/store/exports/ce2672d89d42ab612f3943934308ba0d469a4249533065d9b328ec26a368486f/scene.npz \
  camera_id=cam1 output=tennis_scene/visualize/blender_export/pr1203-meiji-cam1/bundle
```

描画・Blender内検証・MP4化の完全な実行コマンドとcommitは`runs/`の`run.json`/`repro.sh`に保持した。実行時はcleanなcommit `8549c66c05abd9e2b3f73b785cc6382603cfb7b5`で、差分patchは空。bundle・入力・SMPL資産とLinux版Blenderが別途必要で、再実行には空の出力先を用意する。
Linux Blender 5.2.1 LTSは[公式配布](https://download.blender.org/release/Blender5.2/blender-5.2.1-linux-x64.tar.xz)のSHA-256 `a31f524fa99a527d3d52b7f5aaa68c34e1a19d5a1c9473f79c5cc610fd5b10e9`を照合した。
学習を行わない描画検証なのでTensorBoard曲線はない。

## 制約と次の確認

Linuxの描画adapterはllvmpipeで、GPU描画を確認できなかった。最終runは`LIBGL_ALWAYS_SOFTWARE=1`と`GALLIUM_DRIVER=llvmpipe`を明示し、4thread・4sampleのEEVEE描画を使った。低sampleのためネットや影にノイズがあり、フル解像度/59.94fpsの品質評価ではない。
先行するadapter計測の初回は、background modeで`gpu.init()`より先にadapterを取得したため失敗した。計測用コードに初期化を追加して得た結果を`graphics_probe.json`に残した。描画や再構成処理の失敗とは区別する。

入力は旧pipelineの1clipであり、注釈ballと確認済み人物対応を含む。現行検出・対応モデルの実動画精度や独立3D精度の合格は主張しない。既存のモデル採用判断は変更しない。
今回のデータ変換と切り替え演出は検証できた。配布用の高品質動画が必要な場合は、GPU adapterを確認できる環境で解像度・sample数を上げ、同じ保存sceneの検査を併用する。
