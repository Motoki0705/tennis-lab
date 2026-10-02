---
id: run-i964-coco-person-features-r7-20260930
type: run
task: person_tracking
sequence: 7
recorded_at: '2026-09-30'
title: COCO全人物のViTPose・CLIP・SOLIDER特徴回収と全24 NPZ照合
issue: 964
provider: codex
session: 01a0ef7d-19ed-7173-994c-c83934301564
date: '2026-09-30'
status: done
config:
  source: COCO full-frame .30, 800/1333
  pose: ViTPose-H float32 flip-test
  encoders:
  - CLIP-ReID
  - SOLIDER
metrics:
  detection_rows_per_encoder: 40531
  camera_frames: 10491
  archives_verified: 24
  elapsed_seconds: 2946.546712269017
  peak_allocated_bytes: 3021378048
  peak_reserved_bytes: 3074424832
  output_bytes: 365635201
repro:
  commit: 62033eda245c42d70ba0956585274bf1a82f4ff7
  branch: campaign930/i964-2-tracking
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: timeout --signal=TERM --kill-after=15s 5385s env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i964-2-tracking
    /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i964-2-tracking/.venv/bin/python
    -u /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i964-2-tracking/tests/benchmarks/person_tracking_dev_features.py
    --phase extract --repo /home/kamimura/projects/tennis-lab --report /home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/dev_features/i964-features-r7-20260930
artifacts:
  run_dir: knowledge/runs/run-i964-coco-person-features-r7-20260930
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790728513515020890_17831_i964-coco-person-features-r7-20260930.log
parents:
- run-i964-default-solider-cpu-r7-20260930
relations: []
papers: []
tags: []
---

## 回収結果

queue job `1790728513515020890_17831_i964-coco-person-features-r7-20260930` はdone。
workerの2026-09-30 10:23:43 JST done記録とterminal stateを確認した。成功logはexit_codeを
出力しない実装だが、done遷移はrc=0の場合だけである。実行commit `62033eda`、cleanなreproと一致。
[全検査結果](../../runs/run-i964-coco-person-features-r7-20260930/collection.json)と
[成功manifest](../../runs/run-i964-coco-person-features-r7-20260930/features.json)を保存した。

全24 NPZをallow_pickle=Falseで読み直し、hash/bytes、元検出rowの全順序・frame offsets・box・score、
provenance、有限float32値・embedding norm/maskを検証した。両encoder各40,531row、10,491frameで欠落なし。
CLIP/SOLIDERのposeは全rowでbit一致。入力planのhashも成功manifestと一致し、特徴を比較に使用できる。
抽出時にも全値の保存読戻しを行っている。検査の再現はbundleの`collect.py`。

所要2,946.55秒（49.1分）、torch peak allocated 3,021,378,048 bytes、reserved 3,074,424,832 bytes。
出力365,635,201 bytes（348.7 MiB）。見積45–75分/6–9 GB/0.4–0.8 GBに対し、時間内でVRAM・diskは小さかった。
VRAMはtorch allocatorのpeakであり、device全体のtelemetryやCPU RSS peakを測ったとは扱わない。
推論のみで学習曲線はない。

## 次の比較と限界

この回収は整合性・資源の確認であり、tracker/encoderの精度の証拠ではない。
[事前評価プロトコル](../../runs/run-i964-tracker-matrix-r8-20260930/protocol.md)を固定し、
同じ検出・poseに2方式×2encoderと両baselineを並べる。既定tracker・選別規則・予約未見clipは変更しない。
