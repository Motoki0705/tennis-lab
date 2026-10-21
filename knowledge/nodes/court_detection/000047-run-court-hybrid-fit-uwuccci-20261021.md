---
id: run-court-hybrid-fit-uwuccci-20261021
type: run
task: court_detection
sequence: 47
recorded_at: '2026-10-21'
title: Court 2D候補のGPU描画：UwuccciC0Jg_100
issue: 1207
provider: codex
session: 01a15e3f-f433-79b2-8cef-f72bfe4e4cda
date: '2026-10-21'
status: done
config:
  checkpoint: court_detection/multiscale_depth3/b863df1f01f0.ckpt
  device: cuda
  sample_id: UwuccciC0Jg_100
  output:
    width: 960
    height: 540
    fps: 30
    gif_width: 960
    gif_fps: 15
metrics:
  search_rank: 2
  kp_only_error_px: 10.625403295558375
  hybrid_error_px: 2.8945211888325653
  improvement_px: 7.73088210672581
  mp4_decoded_frames: 420
  gif_decoded_frames: 70
  duration_seconds: 14.0
  stills_count: 6
repro:
  commit: 12efecfab4aad0565a949e09b835055efa15d010
  branch: codex/court-2d-hybrid-fit
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.tasks.court_detection.scripts.visualize_hybrid_fit
    mode=render sample_id=UwuccciC0Jg_100 paths.data_root=/home/kamimura/projects/tennis-lab/data paths.checkpoint_root=/home/kamimura/projects/tennis-lab/ckpt
    paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party paths.output_root=/home/kamimura/projects/tennis-lab/outputs
    output.width=960 output.height=540
artifacts:
  run_dir: knowledge/runs/run-court-hybrid-fit-uwuccci-20261021
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/0-8207416308018680032_2026-10-21T11-54-51-981319967Z_3062568_court-hybrid-fit-render-UwuccciC0Jg_100.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/court_detection/visualize/hybrid_fit/20261021T121934.595304Z-fc1a36d2/presentation
  verification: knowledge/runs/run-court-hybrid-fit-uwuccci-20261021/artifact-verification.json
  manifest: knowledge/runs/run-court-hybrid-fit-uwuccci-20261021/manifest.json
parents:
- run-court-hybrid-fit-search-20261021
relations: []
papers: []
tags:
- presentation
- hybrid-homography
- gpu-validation
- candidate-for-user-review
---

## 観察

TennisCourtDetector valの`UwuccciC0Jg_100`を、探索と同じcheckpoint・CUDAで再推論し、
14秒の段階描画を生成した。探索順位は2位で、採用は未確定である。
GPUのKPだけの誤差は10.625403295558375px、KP＋LINEは2.8945211888325653px、
改善量は7.73088210672581px。いずれも同じ可視GT点への原画像pixelの平均距離であり、
GPU探索の値と完全一致した。

MP4はH.264・960×540・30fpsで全420 frameを復号し、PTSの連続性と14秒の尺を確認した。
GIFは全70 frameをRGBへ復号し、幅960・高さ540・総尺14秒を確認した。
元の210時刻のうち隣接する同一画像はGIF writerで統合されるが、表示時間は合算されている。
6段階のPNGも全画素を読み、RGB・960×540を確認した。原画像の静止画はdatasetから読んだ画像、
最後の静止画はmanifestに保存したHから描いた結果と画素が完全一致した。
MP4・GIF・PNG6枚・manifestのSHA-256は
[検証記録](../../runs/run-court-hybrid-fit-uwuccci-20261021/artifact-verification.json)へ保存した。

## CPU参考版との比較と限界

CPU参考版はKPだけ10.62306757744298px、KP＋LINE2.863671552131612pxだった。
GPU−CPUの差はそれぞれ0.0023357181153951245pxと0.030849636700953198px。
原因の切り分けは行っていない。GPU版の素材にはGPU版の誤差を添え、CPU値を流用しない。
画像・予測・seed・fitは実測で、色と区間長は説明用の演出である。
この画像の成功を全体精度や下流3Dの改善へ一般化しない。
動画・GIF・元サイズの静止画はローカルOUTPUTに保持し、Gitへは小さなmanifest・検証記録・
queueの再現bundleだけを残した。学習をしていないので学習曲線はない。
