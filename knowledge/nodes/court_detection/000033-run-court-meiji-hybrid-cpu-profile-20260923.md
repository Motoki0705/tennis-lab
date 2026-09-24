---
id: run-court-meiji-hybrid-cpu-profile-20260923
type: run
task: court_detection
sequence: 33
recorded_at: '2026-09-23'
title: 'Meiji Court: 全frame処理の時間とCPU後処理の内訳'
provider: codex
date: '2026-09-23'
status: done
config:
  checkpoint_sha256: b863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383
  device: cpu
  torch_threads: 1
  clip: video_000/clip_000
  cameras: [cam0, cam1, cam2]
  frame: 0
metrics:
  previous_full_court_seconds: 6956.117
  previous_full_court_frames: 3030
  hybrid_geometry_seconds_unprofiled: [1.962866799, 1.766706007, 2.581103646]
  cam0_profiled_geometry_seconds: 2.443
  cam0_least_squares_calls: 17
  cam0_least_squares_cumulative_seconds: 1.869
artifacts:
  run_dir: knowledge/runs/run-court-meiji-hybrid-cpu-profile-20260923
parents:
- run-tennis-scene-meiji-auto-person-mismatch-20260922
relations: []
papers: []
tags: [real_clip, cpu_profile, diagnostic]
session: 01a0c8c3-d190-7c51-b671-ded223340a2f
---

## 観測と時間の意味

前回のGPU実行ログではcam0領域決定04:20:57.020からCourt JSON保存06:16:53.137まで115分56.117秒だった（JST、2026-09-23）。3030frame当たり約2.30秒であり、モデルのGPU forwardだけの時間ではない。領域候補探索、動画読込、CPU後処理、終端保存も含む。

GPU実行を継続したまま、別途CPUだけで各cameraのframe 0を同じcheckpoint・選択済みROI・hybrid設定で推論し、NumPy/SciPyのgeometry呼出しを計測した。未profile時のgeometry単体はcam0/1/2それぞれ1.963/1.767/2.581秒。全CPU推論時間6.120/5.311/6.120秒はGPU forward時間として扱わない。

同じ入力によるcProfileではcam0 geometry全体2.443秒中、17回のleast_squaresが累積1.869秒、line_residualsが累積1.688秒だった。数値Jacobianのための反復評価が多い。累積時間は呼出し階層で重なるため合算しない。稼働中の本番プロセスはCPU約1coreを消費し、約30秒のGPU観測では大半3%、一部21%だった。これらはCPU hybrid後処理が主要な遅延要因であることを支持する。

現実装はcamera・frameとも直列で、GPU forwardとCPU geometryをframeごとに完了する。court_kp.frame_indexはROI選択と校正referenceの指定であり、単一frameだけにCourt処理を制限する設定ではない。

## 解釈の限界と次の確認

計測は3cameraの各1frameのみで、3030frameのCPU/GPU分解profileではない。CPU診断は実行中のGPUジョブと同じホスト上で行い、CPU負荷による時間変動があり得る。学習ではないためTensorBoard曲線はない。新しいsceneや教師の生成にはこの診断予測を使っていない。

結果契約を保つ高速化候補はframe独立のCPU geometry並列化とGPU batch化だが、この診断では未実装・未測定。固定frameの反復、時間間引き、補間はper-frameの予測・可視性・失敗記録を変えるため同一結果の高速化とは区別する。下流は全TのCourt配列を要求する。
