---
task: slcs
sequence: 16
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-known-memory-cpu-audit-v1
type: run
title: '既知メモリのみのCPU checksum診断: 6process×64回すべて一致'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: done
config:
  diagnosis: CPU-only immutable 64 MiB known pattern; fixed 64 passes; affinity-pinned
    6 processes
  sha_implementations:
  - hashlib OpenSSL
  - CPython _sha256 software
  gpu_used: false
metrics:
  workers: 6
  iterations_per_worker: 64
  comparisons: 384
  mismatches: 0
  elapsed_seconds: 24.86841116999858
repro:
  commit: 892b02c481e368d3357c1d77a446ea368126f4a0
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: timeout --kill-after=5s 300s env PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    .venv/bin/python -B knowledge/runs/run-slcs-known-memory-cpu-audit-v1/probe.py
    --output-dir outputs/tennis_scene/analyze/known_memory_cpu_audit/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-known-memory-cpu-audit-v1
  output_dir: outputs/tennis_scene/analyze/known_memory_cpu_audit/s42-001
  diagnostics: knowledge/runs/run-slcs-known-memory-cpu-audit-v1/results.json
  log: knowledge/runs/run-slcs-known-memory-cpu-audit-v1/queue.log
parents:
- run-slcs-meiji-checkpoint-hash-cpu-v1
- run-slcs-meiji-checkpoint-hash-gpu-control-v1
relations: []
tags:
- slcs
- meiji
- integrity
- diagnosis
---

## 考察 / Findings

### 要約
ファイルを読まずに作った既知不変bytesを、6つの仮想CPUへ固定したprocessで各64回検査し、OpenSSL・独立software SHAの計384比較はすべて固定期待値に一致した。過去のcheckpoint不一致の原因や環境の健全性全般は確定しない。

### アーキテクチャ詳細
各子processはbytes(range(256))を64MiBへ反復した不変bytesを保持し、同じbufferをhashlib.sha256とCPythonの_sha256.sha256へ渡す。既知期待SHAは281e519d...。CPU affinityは0〜5、GPU/model/checkpointは未使用。反復は固定64回で成功まで再試行しない。240秒の内部期限、外側300秒のtimeoutと共有queueのprocess groupを使用した。

### メトリクスの解釈
6processすべてが64回を完了し、返り値0、計384比較で不一致0。総24.868秒。各計算値・反復番号・CPU affinityはcpu-*.jsonlへ保存。入力生成後のhash計算にファイルI/Oはないが、ログと環境記録にはI/Oを使う。学習・収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察
今回の既知メモリ・CPUのみの条件では問題を再現できなかった。checkpointのfile読出し、過去のGPU負荷、命令dispatch、OS/ハードウェアのどれかを原因と断定できない。仮想CPU固定は物理CPU固定を保証せず、CPU6〜11も未試験である。

### 既存実験との比較
CPU対照・GPU後対照に加え、今回は入力ファイルを除外して比較回数を増やした。v1/v3の不一致をこの成功によって取り消さない。productionのhash処理やcache受領書、OS設定は変更していない。

### 次に有効な実験
同じ条件の盲目的な反復は避け、ユーザーによるPCの動作設定の確認、または異なる実行環境との比較を行う。追加SLCS学習・全体生成を再開できる条件はまだ確認できていない。既存教師のCPU品質分析は別途進める。
