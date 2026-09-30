---
id: run-i935-pipeline-candidate-r24-20260930
type: run
task: ball_refiner
sequence: 22
recorded_at: '2026-09-30'
title: 'e9 anchored seed42明示option: 元動画3camera execute/load（投入準備）'
provider: codex
status: planned
config:
  ball_path: e9_anchored_s42_covariance
  clip_id: meiji/video_000/clip_010
  cameras:
  - cam0
  - cam1
  - cam2
  frames_per_camera: 270
metrics: {}
artifacts:
  run_dir: knowledge/runs/run-i935-pipeline-candidate-r24-20260930
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/generate/detector_only_pipeline/i935-e9-anchored-s42-r24-20260930
parents:
- run-i935-covariance-loco-s42-r23-20260930
relations: []
papers: []
tags: []
issue: 935
date: '2026-09-30'
session: 01a0f1dc-49b9-7c82-bcfb-fe29dda6b9e1
repro:
  branch: campaign930/i935-14-pipeline-candidate
  command: timeout --signal=TERM --kill-after=15s 1185s env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
    MAX_JOBS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_CACHE_DIR=/home/kamimura/projects/tennis-lab/outputs/ball_refiner/generate/detector_only_pipeline/i935-e9-anchored-s42-r24-20260930/compiler
    /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/.venv/bin/python
    -u /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/knowledge/runs/run-i935-pipeline-candidate-r24-20260930/check_pipeline.py
    --plan /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/knowledge/runs/run-i935-pipeline-candidate-r24-20260930/plan.json
---

[ユーザー判断B](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5908081470)の既定切替条件に備え、名前付きe9_anchored_s42_covarianceを追加した。既定のft-e13/現refinerは維持。schema v2は補正済み全GMM・存在logit・元frame/PTS/採用窓・倍率・元checkpoint/hashを保存し、fresh loadでは補正を再適用しない。資産が欠落/不一致ならexecute/loadとも停止する。通常検証は[pipeline/configuration 410件](../../runs/run-i935-pipeline-candidate-r24-20260930/pipeline-tests.log)、[全要素比較4件](../../runs/run-i935-pipeline-candidate-r24-20260930/parity-tests.log)が成功。

seed42のselected epoch41 checkpointからCPUでbundleを書き出し、[manifest](../../runs/run-i935-pipeline-candidate-r24-20260930/bundle_manifest.json)へmodel/input/weight/provenanceを固定した。runtimeではこのmanifestのSHA256とe9/checkpoint-bound covariance artifactのSHA256を要求する。元checkpoint SHA985308b0…・artifact SHA197f9e64…を勝手に再選択しない。

[事前許容値・資源見積](../../runs/run-i935-pipeline-candidate-r24-20260930/predeclared-check.md)をGPU出力生成前にissueへ投稿した。[plan](../../runs/run-i935-pipeline-candidate-r24-20260930/plan.json)のvideo_000/clip_010 cam0,1,2全270frameを元mp4から直列executeし、各cameraを新しいプロセスでloadする。保存/loadの全配列・artifact参照はbit一致、source/cacheの全GMMは事前atol・rtol0で比較する。JPEG cacheと元mp4の媒体差で不一致となる可能性を明記し、失敗後に許容値を緩めない。全cameraの最大差・超過数を残し、1つでも未通過ならBを止める。checkpointの再fitや3D精度評価ではない。

[CPU preflight](../../runs/run-i935-pipeline-candidate-r24-20260930/preflight.json)でsource寸法/270frame/cachedNPZ/frame軸/hashと固定optionを照合した。見積3–8分/VRAM2–6GB/約1.5GB、timeout1185秒+kill15秒、device監視7.5GB/allocator6GiB/disk3GB、前jobと同じRAM guard。seed44の後へ1件のみqueue投入する。結果の評価・既定切替は次run。person/pose/courtやvideo_001は使用しない。

最終通常検証は[guard/parity 19件](../../runs/run-i935-pipeline-candidate-r24-20260930/guard-parity-tests.log)、ruff/mypy（変更src/testsの8ファイル）成功。knowledge validatorは0error、repro path検査は0missing。独立validatorは指定がなく0回。
