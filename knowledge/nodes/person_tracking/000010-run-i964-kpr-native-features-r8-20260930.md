---
id: run-i964-kpr-native-features-r8-20260930
type: run
task: person_tracking
sequence: 10
recorded_at: '2026-09-30'
title: KPR native全12 NPZの回収・元40,531検出とposeの照合
issue: 964
provider: codex
session: 01a0efea-3d6e-72a2-9b24-fda1932992d5
date: '2026-09-30'
status: done
config:
  encoder: KPR Market/SOLIDER
  batch_size: 2
  negative_prompts: all other same-frame detections
  parts: 6
  dimension_per_part: 512
metrics:
  rows: 40531
  camera_frames: 10491
  archives_verified: 12
  elapsed_seconds: 1103.9553178070346
  peak_allocated_bytes: 573095424
  peak_reserved_bytes: 731906048
  output_bytes: 322955301
repro:
  commit: 935cb0d55175214abef807d44d3f654ebb743038
  branch: campaign930/i964-2-tracking
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: timeout --signal=TERM --kill-after=15s 5385s env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i964-2-tracking
    /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i964-2-tracking/.venv/bin/python
    -u /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i964-2-tracking/tests/benchmarks/person_kpr_features.py
    --phase extract --repo /home/kamimura/projects/tennis-lab --report /home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/dev_features/i964-kpr-r8-20260930
artifacts:
  run_dir: knowledge/runs/run-i964-kpr-native-features-r8-20260930
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790735986295857060_1473474_i964-kpr-native-features-r8-20260930.log
parents:
- run-i964-kpr-cpu-parity-r8-20260930
- run-i964-coco-person-features-r7-20260930
relations: []
papers: []
tags: []
---

## 回収と整合性

queue `1790735986295857060_1473474_i964-kpr-native-features-r8-20260930` のdone/clean reproを確認。
実行commitは`935cb0d55175214abef807d44d3f654ebb743038`。
[再検査結果](../../runs/run-i964-kpr-native-features-r8-20260930/collection.json)と
[検査script](../../runs/run-i964-kpr-native-features-r8-20260930/collect.py)を保存した。
全12 NPZ、40,531row・10,491camera-frameで、元CLIP archiveのrow/box/score/pose/offsetがbit一致し、
検出artifactのbox/score/offsetにも直接一致した。全入力・出力hash、plan・重み・code・source動画の出自も一致。
partsは有限float32 `(N,6,512)`、visibilityはbool `(N,6)`。可視partはunit normで、不整合は無かった。
flattenしたcosineへ変換していない。HL3の[notice](../../../src/tasks/player_association/appearance/kpr_vendor/NOTICE.md)を維持。
重みsha256は`e29bacd699a15d1d069c9d19c8804b7d35baea81e713b55e632ee546e1733a1b`。

## 資源と解釈

所要1,103.96秒（18.40分）。torch peak allocated 573,095,424 bytes、reserved 731,906,048 bytes。
出力322,955,301 bytes（308.00 MiB）。前runの見積20–45分、VRAM4–7GB、disk0.5–0.8GBを下回った。
VRAMはtorch allocatorのpeakで、device全体telemetry/CPU peak RSSは未測定。
学習ではないため曲線はない。回収runはCPU照合のみでGPU再推論はしていない。

この記録は特徴の整合性を確かめたもので、追跡精度の比較ではない。次はrun 9の事前addendumに沿い、
共通可視partだけの距離をtracker/camera間対応へ接続する。未見clipは予約を維持する。
