---
id: run-ball-colab-checkpoint-resume-20261024
type: run
task: ball_detection
sequence: 67
recorded_at: '2026-10-24'
title: 720p Colab学習のcheckpoint逐次保存と新VM再開の検証
issue: 1375
provider: codex
date: '2026-10-24'
status: running
config:
  architecture: hybrid
  frames: 32
  native_image_size: [1280, 720]
  frame_chunk_size: 4
  activation_checkpointing: true
  updates: 12
  interrupted_after_update: 5
  input_frames_per_update: 128
  seed: 42
metrics:
  first_environment_setup_seconds: 41.91178623799999
artifacts:
  run_dir: knowledge/runs/run-ball-colab-checkpoint-resume-20261024
parents:
- run-ball-colab-persistence-preflight-20261023
- run-ball-portable-distribution-20261023
- run-ball-temporal-heatmap-gpu-diagnostic-20261023
relations: []
papers: []
tags: []
---

720p・本番モデル幅の短い学習で、checkpoint完成直後のDrive保存と、新VMへの同run復元を検証する。
先行の注釈path問題は解消し、移動済み配布版と破損注釈を含む6テストをこのcheckoutでも確認した。
最初のL4 VMではPython 3.11・`uv sync --locked --no-cache`の環境構築が41.91秒で成功した。
全train/validation配布storeの新規取得・照合を開始した。既存の他者所有の可能性がある割当は操作していない。

実験controllerはこのrunの`resume_probe.py`に置き、常設runnerには同期保存callbackだけを接続した。
汎用の`RcloneSnapshotStore`はrun treeを世代ごとに保存し、全fileのサイズとMD5、markerの内容を照合して
receiptを返す。復元はreceiptに固定したmarkerのSHA-256、各fileのSHA-256、相対pathと集合を照合する。
保存失敗を成功へ読み替えず、同名世代や部分世代を上書きしない。
CPUでは実モデルの中断・snapshot復元・resumeと、通信断後に更新が進まないことを確認した。

主比較や本学習の予算決定は行わない。全量storeの取得・検証とT=32 manifestの再生成を行い、
保存復旧だけの短い更新・評価にはそこから作る独立した4 clipの診断storeを使う。
trainはTrackNet・Meiji・Chat、validationは位置正例と確認済み不在を持つChatを選ぶ。
すべて1280×720のnative画像とし、幅・深さ・5ch・全frame教師・128frame/更新を維持する。
12更新・5更新目で中断、損失許容差は絶対1e-6・相対1e-4を実行前に固定した。
この診断の精度や時間を本学習の収束・予算・長系列上限の根拠には使わない。
Colabでの保存・中断・新VM再開の結果は実測後に追記する。
