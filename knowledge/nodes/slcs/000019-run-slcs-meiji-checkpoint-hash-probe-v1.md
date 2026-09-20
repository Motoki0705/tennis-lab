---
task: slcs
sequence: 19
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-meiji-checkpoint-hash-probe-v1
type: run
title: 'Meiji checksum診断: CUDA pose後のPythonハッシュ不一致を再現'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: failed
config:
  model: ViTPose-H BF16; flip_test; batch8
  data: video_000/clip_007 cam0 P0;652frames
  diagnosis: hashlib.file_digest / hashlib manual / sha256sum; before/after/unload
metrics:
  files: 4
  phases: 3
  file_phase_comparisons: 12
  mismatching_file_phase_comparisons: 1
  detection_identity_errors: 1
  stat_changed_comparisons: 0
repro:
  commit: 88dca25f954178bb3668768c8418332bd0a98749
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    .venv/bin/python knowledge/runs/run-slcs-meiji-checkpoint-hash-probe-v1/probe.py
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-checkpoint-hash-probe-v1
  log: .training_queue/logs/1789733184078608758_800353_slcs-meiji-checkpoint-hash-probe-v1.log
  output_dir: outputs/tennis_scene/analyze/meiji_hash_probe/s42-001
  diagnostics: knowledge/runs/run-slcs-meiji-checkpoint-hash-probe-v1/results.json
parents:
- run-slcs-meiji-temporal-probe-v1
relations: []
tags:
- slcs
- meiji
- integrity
- diagnosis
---

## 考察 / Findings

### 要約
CUDA ViTPose推論の前後で同じ4ファイルを3方式で読み、推論後にPython内のSHA256不一致を再現した。診断assertが失敗したためqueue statusはfailed。ファイル改変やハードウェア故障を断定する根拠はない。

### アーキテクチャ詳細
DINO・ViTPose・DINOv3 checkpointと動画を、hashlib.file_digest、1MiB逐次update、別processのsha256sumで比較した。実ViTPoseを652frameに適用し、推論前・推論後・unload後に同じ検査を行った。各phaseで実際の_detections cache照合も実行した。

### メトリクスの解釈
12件のfile×phase比較のうち、推論後ViTPoseの手動Python hashだけ64115ceb...となり、既知の50e33f40...と不一致。file_digestとsha256sumは一致した。同phaseの_detectionsはDINOをe1dfce85...と計算し、既知のe61688af...を拒否した。inode/size/mtimeは全比較で不変。推論前とunload後は全方式一致した。学習ではなく収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察
推論後に不一致が生じたという時間的関係は観測できたが、CUDA起因とはまだ確定しない。ファイル読み出し、Python/OpenSSLの計算、process内のメモリ状態、環境側の不安定性を分離していない。過去の異なるreceipt値や意図しない再起動との因果関係も未確定。

### 既存実験との比較
CPUのみの先行反復読出しは一致し、今回実推論を挟んで不一致を観測した。単なるstale cacheとしてreceiptを書き換えると、生成時の信頼性問題を隠すため行わない。

### 次に有効な実験
同じ不変bytesをPython/OpenSSLと_sha256 software実装で計算し、既知bytes・file同一chunk・再読出しの差を分離する。GPU診断も共有queueで1件ずつ実行し、原因を確認するまで全体生成を開始しない。
