---
id: run-video-batch-input-qualification-20261116
type: run
task: tennis_scene
sequence: 53
recorded_at: '2026-11-16'
date: '2026-11-16'
title: meiji_4camの追加8clipもCourt cam3で停止
issue: 1748
provider: codex
session: 01a1dfbd-594e-7f72-9f3a-53d3f7f3a340
status: failed
summary_impact: none
config:
  purpose: input qualification only; not performance baseline
  candidate_order:
  - video_000/clip_003
  - video_000/clip_004
  - video_000/clip_005
  - video_000/clip_006
  - video_000/clip_007
  - video_000/clip_008
  - video_000/clip_009
  - video_000/clip_010
  camera_order:
  - cam3
  - cam1
  - cam0
  - cam2
  allocator_limit_gib: 10
  stop: first fully accepted clip or 8 candidates
metrics:
  candidates_screened: 8
  all_cameras_accepted: 0
repro:
  commit: ec4c192060d4394d7e2a486bf2536a8ec8cd0337
  command: TENNIS_LAB_DINO_EXTENSION=/home/kamimura/projects/tennis-lab/outputs/player_detection/evaluate/coco_fullframe/i964-r5-20260929/dino_extension/lib/MultiScaleDeformableAttention.so
    .venv/bin/python knowledge/runs/run-video-batch-baseline-20261116/qualify_inputs.py
artifacts:
  run_dir: knowledge/runs/run-video-batch-input-qualification-20261116
  output_dir: outputs/tennis_scene/video_batch_baseline/20261116/qualification
  inputs: knowledge/runs/run-video-batch-input-qualification-20261116/input-files.json
  queue_job: 0-8205243388358865349_2026-11-15T15-30-11-641134650Z_336854_video-batch-input-qualification
parents:
- run-video-batch-baseline-20261116
relations: []
papers: []
tags:
- inference
- input-qualification
- negative-result
---

既定のCourt検出で全cameraを通過できる入力を選ぶため、meiji_4camで書き出し・登録済みのclip_003〜010を順に調べた。候補8件すべてがcam3でCourtRegionUnavailableとなり、完走候補は得られなかった。これは[固定3clipでの停止](000052-run-video-batch-baseline-20261116.md)と同じ未解決点を補強する結果であり、summaryの判断は変わらないためsummary_impactはnoneとする。

カメラ順はcam3→cam1→cam0→cam2。1cameraで不成立なら残りを実行せず次clipへ進み、最初の全4camera成立例、または8候補で終了する計画を実行前に子issueへ記録した。全例が最初のcam3で止まったので、ほかのcameraの成立可否はこのrunでは未測定である。既定モデル・閾値・精度・clip境界は変えていない。Court predictorを保持してframe0を確認する入力選定であり、このrunのtimingやload回数を現行clip逐次実行の基準値へ混ぜない。

| clip | source frames | 停止camera | 最もinliersの多い候補 | その候補のinside_content |
|---|---:|---|---:|---|
| video_000/clip_003 | 369 | cam3 | 9 | false |
| video_000/clip_004 | 586 | cam3 | 7 | false |
| video_000/clip_005 | 314 | cam3 | 7 | true |
| video_000/clip_006 | 867 | cam3 | 8 | false |
| video_000/clip_007 | 401 | cam3 | 5 | true |
| video_000/clip_008 | 626 | cam3 | 7 | false |
| video_000/clip_009 | 974 | cam3 | 7 | false |
| video_000/clip_010 | 545 | cam3 | 7 | false |

clip_003と006では8点以上の支持がある候補も存在したが、投影がcontent内に入る条件を満たさず採用されなかった。したがって全例を単に「点数不足」と説明しない。[全候補の診断](../../runs/run-video-batch-input-qualification-20261116/results.json)に各regionのstatus・inliers・area・inside_contentを保持する。clip_011以降や全69clipが失敗すると推定した結果ではない。

入力16file（実際に読んだcam3動画8本とclip.json8本）のSHA-256は終了後に採取し、ctimeがscreening開始より前であり、hash前後でctime/sizeが変わらないことを確認した。[input-files.json](../../runs/run-video-batch-input-qualification-20261116/input-files.json)に保存し、比較基準値の入力を事前freezeした方法とは区別する。モデル資産SHAは[recorder JSON](../../runs/run-video-batch-input-qualification-20261116/recording.json)、実際のcommit/command・未コミット差分はrun.jsonとuncommitted.patchで確認できる。scriptは前提run bundleのqualify_inputs.pyである。

GPU実行は共有training queueのall枠。queueとrecorderは正常終了したが、全cameraを通過する入力選定という目的は未達なのでstatusはfailedとする。source dataset、clip_studio、同期スクリプトへは書き込まず、processも止めていない。C1を完了するには、Court側の対処の先行か、比較入力の見直しについて親判断が必要である。
