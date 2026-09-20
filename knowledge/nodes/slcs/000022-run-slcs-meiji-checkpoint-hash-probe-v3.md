---
task: slcs
sequence: 22
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-meiji-checkpoint-hash-probe-v3
type: run
title: 'Meiji高速checksum診断: 別processのsha256sumでも不一致'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: failed
config:
  model: ViTPose-H BF16 batch8 flip_test; one652frame inference
  diagnosis: 6post-predict diskpasses; two OpenSSL contexts; per-block software baseline
metrics:
  comparison_rows: 14
  mismatch_rows: 1
  python_disk_passes: 8
  python_memory_passes: 2
  mismatched_blocks: 0
  system_sha256sum_calls: 3
  system_sha256sum_mismatches: 1
  elapsed_seconds: 110.17556197499835
repro:
  commit: 89daad4e7cf501ff4c63f5016d9aa25ae159ac52
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: timeout --kill-after=5s 590s env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -B knowledge/runs/run-slcs-meiji-checkpoint-hash-probe-v3/probe.py
    --output-dir outputs/tennis_scene/analyze/meiji_hash_probe_v3/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-checkpoint-hash-probe-v3
  log: .training_queue/logs/1789734467299446249_848899_slcs-meiji-checkpoint-hash-probe-v3.log
  output_dir: outputs/tennis_scene/analyze/meiji_hash_probe_v3/s42-001
  diagnostics: knowledge/runs/run-slcs-meiji-checkpoint-hash-probe-v3/results.json
parents:
- run-slcs-meiji-checkpoint-hash-probe-v1
- run-slcs-meiji-checkpoint-hash-probe-v2b
relations: []
tags:
- slcs
- meiji
- integrity
- diagnosis
---

## 考察 / Findings

### 要約
高速なdisk反復はPython内で一致したが、明示同期後に起動した別processのsha256sumがf09d7371...を返し、既知50e33f40...と不一致。問題をPythonのprocess内状態だけに限定できなくなった。全体生成は未再開。

### アーキテクチャ詳細
ViTPose checkpointの不変bytesと1MiB単位のsoftware SHAをCPUで事前取得。実推論直後に明示同期を挟まずdiskを6回読み、各chunkを独立した2つのOpenSSL contextへ渡し、block hashと保持bytes完全一致も確認した。その後memory1回・明示同期後disk1回・unload後disk1回、適所でsystem hashを記録した。libs maps・thread数・statを保存。model内部のCPU転送は暗黙同期を含む。

### メトリクスの解釈
Python disk8回・memory2回は全digest一致、block異常0。system hashは3回中1回不一致、その前後は一致。全stat検査でinode/size/mtime/ctimeは不変。総110.2秒。14行はload/statを含む比較で独立14回の再現試験ではない。学習・収束曲線なし。

### アーキテクチャ⇄メトリクスの因果考察
不一致はCUDA明示同期の後、別processのsystem toolでも発生したため、Python固有の入力buffer破壊だけでは十分に説明できない。ただしfile読出し・libcrypto計算・OS/CPU/メモリのどこで異常が生じたか未確定。system sha256sumはlibcrypto.so.3に依存し、Pythonはbuiltin OpenSSL3.5.5なので、完全に無関係なSHA実装同士とは言えない。chronologicalなphaseを同期有無のclean A/Bとみなさない。

### 既存実験との比較
v1はPython hashが不一致、v2bは全検査一致だった。今回system側の不一致を確認し、system hashへの置換だけでは解決にならないと分かった。旧receiptの由来や意図しない再起動との因果関係は未確定。

### 次に有効な実験
CUDA/modelを使用しないfresh processでsystem hashを反復し、既知bytesの計算も比較する。OpenSSLのSHA命令経路をprocess-local環境変数で切り替えて診断するが、productionの検査を無効化・緩和しない。
