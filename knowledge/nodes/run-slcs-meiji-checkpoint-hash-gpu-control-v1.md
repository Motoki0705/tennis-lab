---
id: run-slcs-meiji-checkpoint-hash-gpu-control-v1
type: run
title: 'Meiji推論後のchecksum対照: 通常・命令制限とも不一致なし、原因未確定'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: done
config:
  model: ViTPose-H BF16 batch8 flip_test;652frames; explicit synchronize
  diagnosis: reuse CPU control;16post-predict filechildren;2known-memory children;2after-unload
    filechildren
  openssl_child_mask: :~0x20000000
metrics:
  post_predict_default_file_runs: 8
  post_predict_masked_file_runs: 8
  after_unload_file_runs: 2
  file_mismatches: 0
  known_memory_children: 2
  memory_iterations_per_child: 4
  memory_mismatches: 0
  elapsed_seconds: 100.99369124000077
repro:
  commit: c0ee3ecaa7be5bccbd58995008775e201a3f995f
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: timeout --kill-after=5s 590s env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -B knowledge/runs/run-slcs-meiji-checkpoint-hash-gpu-control-v1/probe.py
    --output-dir outputs/tennis_scene/analyze/meiji_hash_gpu_control_v1/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-checkpoint-hash-gpu-control-v1
  log: .training_queue/logs/1789735247330776197_877072_slcs-meiji-checkpoint-hash-gpu-control-v1.log
  output_dir: outputs/tennis_scene/analyze/meiji_hash_gpu_control_v1/s42-001
  diagnostics: knowledge/runs/run-slcs-meiji-checkpoint-hash-gpu-control-v1/results.json
parents:
- run-slcs-meiji-checkpoint-hash-cpu-v1
- run-slcs-meiji-checkpoint-hash-probe-v3
relations: []
tags:
- slcs
- meiji
- integrity
- diagnosis
---

## 考察 / Findings

### 要約
GPU推論・明示同期後にモデルを保持した状態でも、通常・OpenSSL命令制限のfresh-process対照は今回すべて一致した。発生は間欠的であり、原因も命令制限の改善効果も確定しない。全体教師生成と追加SLCS学習はまだ再開しない。

### アーキテクチャ詳細
CPU対照のProbeを再利用し、同じ652frameのViTPose推論後にfile比較16回と既知memory子process2件を実施した。unload後は通常・命令制限を各1回確認し、phase別に集計する。親processだけがCUDAモデルを保持し、hash子processはtorchをimportしない。maskは子の環境だけで、OS・production設定は不変。

### メトリクスの解釈
推論後のfile各8回とunload後2回は不一致0、既知memory各4回も固定期待値と一致。総101.0秒。GPUを使用しない対照と両条件が成功したため、maskの優位性やGPU起因を立証しない。学習・収束曲線なし。

### アーキテクチャ⇄メトリクスの因果考察
同じモデルを用いたv3ではsystem側不一致を観測したが、本runでは発生しなかった。再現に必要なCPU/メモリ/OS/ライブラリの条件が残っている。成功するまでの再試行で過去の失敗を無効化せず、各runを独立して保存した。

### 既存実験との比較
v1のPython不一致、v3のsystem不一致に対し、v2b・CPU対照・本GPU対照は成功。一定した失敗方式や単純なCUDA有無の説明に絞れていない。別processのhashへ置き換えるだけの修正は採用しない。

### 次に有効な実験
PCのオーバークロック・低電圧化・メモリ設定の確認を踏まえ、別の安定した環境または実行条件で同じ診断を比較する。問題をcheckpoint受領書の手修正やchecksum検査の緩和で隠さず、全体生成の再開条件を確認する。
