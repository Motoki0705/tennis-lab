---
id: run-i1757-dino-gpu-health
type: run
task: tennis_scene
sequence: 73
recorded_at: '2026-11-19'
title: DINO fp16の停止とGPU定期照会の切り分け
issue: 1757
provider: codex
status: failed
summary_impact: none
config:
  clip: video_000/clip_027
  camera: cam0
  frames: 235
  precision: fp16
  batch_size: 1
  heartbeat_timeout_seconds: 60
  condition_limit_seconds: 480
  gpu_query_interval_seconds_in_b: 1
  gpu_query_timeout_seconds_in_b: 10
metrics:
  condition_a:
    status: complete
    elapsed_seconds: 133.18842032499379
    frames_completed: 235
    gpu_queries: 0
    load_1m: 1.6044921875
    other_worktree_pytest: 0
    pytorch_peak_allocated_bytes: 1767208448
  condition_b:
    status: nvidia_smi_unresponsive
    elapsed_seconds: 84.0375918019854
    frames_completed: 52
    last_started_frame_index: 52
    last_completed_frame_index: 51
    gpu_queries_successful: 73
    load_1m: 1.90869140625
    other_worktree_pytest: 0
    pytorch_peak_allocated_bytes_before_stop: 1767208448
    sampled_peak_device_mib: 3563
repro:
  commit: 429065101fd7ddbef62b891048281daaa012d382
  branch: codex/video-batch-benchmark
  command: .venv/bin/python -B -u knowledge/runs/run-i1757-dino-gpu-health/run.py
    run --root outputs/tennis_scene/evaluate/dino-gpu-health-20261119
    --expect-commit 429065101fd7ddbef62b891048281daaa012d382
artifacts:
  run_dir: knowledge/runs/run-i1757-dino-gpu-health
  output_dir: outputs/tennis_scene/evaluate/dino-gpu-health-20261119
  metrics: knowledge/runs/run-i1757-dino-gpu-health/evidence.json
  queue_job: 0-8204931363351882979_2026-11-19T06-10-36-648117020Z_1140385_i1757-dino-gpu-health
parents:
- run-i1757-video-batch-meiji
relations: []
papers: []
tags:
- inference
- diagnosis
---

## 考察 / Findings

逐次probeの2回の停止を受け、DINO人物検出だけを同じ `clip_027/cam0` の全235フレームで実行した。前回probeと動画のSHA-256、DINO fp16の設定、重み・ソース・拡張を含む56件のファイルが一致した。条件ごとに新規プロセスを使い、RTX 5060 Tiのall枠へ1ジョブを投入した。[再現手順](../../runs/run-i1757-dino-gpu-health/README.md)

| 条件 | 開始 JST | 結果 | 所要秒 | 完了フレーム | 最後のheartbeat | 開始時の1分load / 他worktree pytest |
| --- | --- | --- | ---: | ---: | --- | --- |
| A・GPU照会なし | 15:10:37 | 完走 | 133.19 | 235/235 | `complete`、最終完了index 234 | 1.60 / 0 |
| B・1秒ごとにGPU照会 | 15:14:21 | 照会タイムアウトで中止 | 84.04 | 52/235 | `dino_inference`、処理中index 52（53番目） | 1.91 / 0 |

Bの開始前にCPU条件を満たすまで90.09秒待った。Aは `[1, 3, 750, 1333]` のforwardが235回、DINO load/unloadは各1回だった。PyTorch割当ピークはA完走時、B停止直前とも1,767,208,448 bytes。Bの照会で取得できたGPU全体の使用量の最大は3,563 MiB。Bは照会73回に成功し、15:15:34.360 JSTに開始した74回目が10秒以内に返らなかった。照会プロセスの終了待ちを含め、15:15:45.380 JSTにworkerを停止し、15:15:49 JSTにqueueが `failed`（終了コード1）になった。[測定値とイベントのJSON](../../runs/run-i1757-dino-gpu-health/evidence.json)

Bの最後のheartbeatは15:15:45.212 JSTで、応答しなかった照会の開始から10.85秒後、監視の停止判定の0.17秒前まで推論処理が進んでいた。WSLには15:15:44 JSTに `dxgvmb_send_sync_msg: wait_for_completion failed: fffffe00` が1件あり、Bの照会中には `dxgkio_escape: Ioctl failed: -22` が219件あった。Aでは該当するdxgログが0件。WindowsのSystemログから取得した `nvlddmkm` は両条件とも0件だった。生の `heartbeat.json`、`stop.json`、`gpu-queries.jsonl`、`kernel-dxg.log`、`windows-nvlddmkm.log` は上記出力先の各条件にある。

定期照会が計測の停止に関与する仮説を支持する。DINO単独でも照会タイムアウトで監視が終了することが分かり、今回のBでは直前までDINOのheartbeatが続いた。各条件1回でAからBの固定順なので、GPU処理自体が停止した過去の事例への一般化は保留する。標準pipeline一括推論の性能・品質の判断は既存の要約に従い、この記録はGPU照会と停止の調査結果として残す。
