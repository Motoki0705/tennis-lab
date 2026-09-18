---
id: run-slcs-meiji-checkpoint-hash-cpu-v1
type: run
title: 'Meiji CPUのみのchecksum対照: 通常・命令制限とも不一致なし'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: done
config:
  diagnosis: no CUDA/model imports;16fresh sha256sum processes;2fresh Python memory
    processes
  openssl_child_mask: :~0x20000000
metrics:
  default_file_runs: 8
  masked_file_runs: 8
  file_mismatches: 0
  known_memory_children: 2
  memory_iterations_per_child: 4
  memory_mismatches: 0
  elapsed_seconds: 63.003466768999715
repro:
  commit: d4e6e65244b3eabc530c57e5899f117168785a62
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: timeout --kill-after=5s 590s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python
    -B knowledge/runs/run-slcs-meiji-checkpoint-hash-cpu-v1/probe.py --output-dir
    outputs/tennis_scene/analyze/meiji_hash_cpu_v1/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-checkpoint-hash-cpu-v1
  log: .training_queue/logs/1789734915571163396_866836_slcs-meiji-checkpoint-hash-cpu-v1.log
  output_dir: outputs/tennis_scene/analyze/meiji_hash_cpu_v1/s42-001
  diagnostics: knowledge/runs/run-slcs-meiji-checkpoint-hash-cpu-v1/results.json
parents:
- run-slcs-meiji-checkpoint-hash-probe-v3
relations: []
tags:
- slcs
- meiji
- integrity
- diagnosis
- cpu-only
---

## 考察 / Findings

### 要約
GPU/modelを使用せずに実行したfresh processの対照はすべて一致。通常のsha256sum8回、OpenSSL命令制限8回、既知64MiBの2条件×4回の計算が既知SHAに一致した。GPU推論後の不一致は未解明のまま。

### アーキテクチャ詳細
2.55GBの同一ViTPoseファイルを別processで反復し、通常環境と子processだけのOPENSSL_ia32cap=:~0x20000000を交互に使用した。既知bytesは2つのfresh Python子processでhashlibと_sha256を比較した。maskはSHA拡張bitを落とすが、OpenSSL3.5では後続vectorも無効となりSHA単独介入ではない。OSやproduction設定は変更していない。

### メトリクスの解釈
16回のfile hash不一致0、既知memory不一致0、stat変化0。総63.0秒。通常も対照も成功したため、このrunから命令制限の改善効果を主張できない。学習ではなく収束曲線なし。

### アーキテクチャ⇄メトリクスの因果考察
GPUを使わないこの条件で問題を再現しなかったという観測に限定する。CPU・RAM・OSの健全性全般やCUDA起因は立証しない。既知bufferも固定期待hashに照合しており、2実装が同じ誤値を返しても成功扱いしない。

### 既存実験との比較
実推論を含むv1ではPython、v3ではsystem hashが不一致になった。v2bは実推論を含んでも全一致だったため、発生は単純なCUDA有無だけで説明できていない。

### 次に有効な実験
推論後のfresh childでも通常・命令制限を比較すれば経路の関与を検証できる。PC設定の確認と合わせ、計算の信頼性を確認するまで全体生成・追加学習を保留する。
