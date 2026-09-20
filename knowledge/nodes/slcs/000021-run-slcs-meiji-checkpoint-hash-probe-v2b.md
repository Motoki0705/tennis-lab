---
task: slcs
sequence: 21
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-meiji-checkpoint-hash-probe-v2b
type: run
title: 'Meiji checksum差分診断: 同一bytes比較は一致、原因未確定'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: done
config:
  model: ViTPose-H BF16 batch8 flip_test; one652frame inference
  diagnosis: same bytes hashlib/_sha256/sha256sum; explicit CUDA synchronize
metrics:
  comparison_rows: 29
  mismatch_rows: 0
  disk_blocks_per_phase: 2431
  elapsed_seconds: 207.3279926059986
  inference_seconds: 25.147090390000812
repro:
  commit: bbe6a703022c6b44a765b6f570e6fe7b85eceb7b
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: timeout 10m env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    .venv/bin/python knowledge/runs/run-slcs-meiji-checkpoint-hash-probe-v2/probe.py
    --output-dir outputs/tennis_scene/analyze/meiji_hash_probe_v2/s42-002
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-checkpoint-hash-probe-v2b
  log: .training_queue/logs/1789734039602475863_820662_slcs-meiji-checkpoint-hash-probe-v2b.log
  output_dir: outputs/tennis_scene/analyze/meiji_hash_probe_v2/s42-002
  diagnostics: knowledge/runs/run-slcs-meiji-checkpoint-hash-probe-v2b/results.json
parents:
- run-slcs-meiji-checkpoint-hash-probe-v1
- run-slcs-meiji-checkpoint-hash-probe-v2
relations: []
tags:
- slcs
- meiji
- integrity
- diagnosis
---

## 考察 / Findings

### 要約
同じ不変bytesを保持したOpenSSL・_sha256・sha256sum比較は全29行で一致した。v1の不一致を今回は再現しなかったが、原因の解明・修復を意味しない。

### アーキテクチャ詳細
2.55GB checkpointを1MiB bytes列として一度保持し、推論前・明示CUDA同期後・unload後に各2回のメモリ内比較を行った。同一chunkを2つの実装へ入力するdisk hash、chunkごとのhash、保持bytesとの完全一致、別processのstdin/file hashを比較した。既知64MiB patternも使用し、torch等のimport前とimport後を分離した。Python/OpenSSL版と実装module originを保存した。

### メトリクスの解釈
29行は各method比較とstat検査を含む。diskは各phase2431blockで不一致0。総207.3秒、そのうち652frameのViTPose推論25.1秒。推論以外の検査に時間を要し、v1より検査順とタイミングが異なる。学習ではなく収束曲線なし。

### アーキテクチャ⇄メトリクスの因果考察
同一bytesの不一致やdisk再読取り差をこのrunでは観測しなかった。v1にない明示CUDA同期、保持buffer、software hashの追加は処理タイミング・メモリ状態を変えるため、同期による修復や特定実装の無罪を断定しない。過去24時間のWindows WHEA-Logger取得は0件だったが、ハードウェアの健全性を保証しない。

### 既存実験との比較
v1では推論後の手動hashとcache照合で不一致を観測した。v2は起動失敗で未測定。本runは診断として完走しただけで、全体データ生成の再開条件を満たしたとはしない。

### 次に有効な実験
v1に近い推論直後の高速読取りを繰り返し、同じchunkの二重OpenSSL contextと既知block hash・bytes比較で不一致位置を局所化する。検査を緩めたりreceiptを手修正したりしない。
