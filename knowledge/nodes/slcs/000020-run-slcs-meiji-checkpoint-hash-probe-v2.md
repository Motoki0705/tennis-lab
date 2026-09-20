---
task: slcs
sequence: 20
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-meiji-checkpoint-hash-probe-v2
type: run
title: 'Meiji checksum差分診断: provenance取得のbuiltin module対応で起動失敗'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: failed
config:
  diagnosis: same-buffer SHA differential; before/after CUDA
metrics:
  completed_hash_comparisons: 0
  cuda_inferences: 0
repro:
  commit: 9450fff0345ef5ebe5c166df6e61672a3742d0aa
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: timeout 10m env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    .venv/bin/python knowledge/runs/run-slcs-meiji-checkpoint-hash-probe-v2/probe.py
    --output-dir outputs/tennis_scene/analyze/meiji_hash_probe_v2/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-checkpoint-hash-probe-v2
  log: .training_queue/logs/1789733935690362918_818639_slcs-meiji-checkpoint-hash-probe-v2.log
parents:
- run-slcs-meiji-checkpoint-hash-probe-v1
relations: []
tags:
- slcs
- meiji
- integrity
- startup-failure
---

## 考察 / Findings

### 要約
provenance取得で組込み_hashlibに存在しない__file__を参照して起動失敗。hash比較もCUDA推論も未実行であり、checksum原因に関する結果はない。

### アーキテクチャ詳細
同じ不変bytesをOpenSSL・_sha256・sha256sumで比較する追加診断。Python3.11.15のこのbuildでは_hashlibが組込みmoduleとなる。

### メトリクスの解釈
完了比較0、CUDA推論0。収束曲線なし。

### アーキテクチャ⇄メトリクスの因果考察
環境メタデータの取得方法に問題があり、既観測のhash不一致との関係はない。

### 既存実験との比較
v1の実推論後hash不一致は引き続き未解決。今回の起動失敗を不一致の再現または解消として数えない。

### 次に有効な実験
__file__がない組込みmoduleを__spec__.originで記録し、CPUのみの初期化確認後、新runとして診断を実行する。
