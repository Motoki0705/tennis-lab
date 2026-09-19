---
id: run-slcs-meiji-read-modes-v1
type: run
title: 1bit差の位置を通常読取とdirect IOで限定比較
provider: codex
date: '2026-09-19'
status: done
config:
  device: cpu
  offset: 1896501248
  length: 4096
  schedule:
  - saved_good
  - saved_bad
  - live_buffered_before
  - live_direct
  - live_buffered_after
  kernel: 6.6.87.2-microsoft-standard-WSL2
  direct_supported: true
  statx_dioalign:
    statx_mask: 14335
    memory_bytes: 4
    offset_bytes: 512
  global_cache_eviction: false
  retries: 0
metrics:
  scheduled_reads: 5
  completed_reads: 5
  live_reads: 3
  live_reads_matching_good: 3
  live_reads_matching_bad: 0
  saved_block_differences: 1
  bytes_per_read: 4096
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-read-modes-v1
  output_dir: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_read_modes/s42-001
  ledger: knowledge/runs/run-slcs-meiji-read-modes-v1/ledger.json
parents:
- run-slcs-meiji-stream-byte-diff-v1
relations: []
tags:
- slcs
- integrity
- direct-io
- limited-control
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: f87e0bf95c36554d59eede0b9d8f7f68bb6ddf31
  branch: codex/slcs-real-rgb
  command: env CUDA_VISIBLE_DEVICES= .venv/bin/python -B knowledge/runs/run-slcs-meiji-read-modes-v1/probe.py
    --output /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_read_modes/s42-001
---

## 考察 / Findings

### 要約
保存済み1bit差の位置を含む4KiBだけを、固定5回の順序で比較した。good/bad保存blockは既知の1bit差を再現し、liveの通常read→O_DIRECT→通常readはすべてgood blockと完全一致した。この時点・この区間で持続する異常は観測しなかった。過去の異常を取り消す結果や全体の安定性確認ではない。

### アーキテクチャ詳細
新規outputへgood/bad各1回、live各3回のblockを保存。各readで同じbytesのhashlib/_sha256と前後statを検査した。O_DIRECTはlibc statxのSTATX_DIOALIGNによりmemory4bytes・offset512bytesを取得し、匿名MAP_SHARED bufferのアドレスと4096byteのoffset/lengthが要件を満たすことを確認した。accepted_flagsも検査し、未対応・short read時は停止する設計。cache eviction、元fileへの書込、再試行、GPU、model loadは無い。7 fixture testsと対象ruff/mypy通過。

### メトリクスの解釈
5read×4096bytesが完了。live3readの対象byteはすべて157（0x9D）でgood block SHA93a99071...に一致し、bad保存blockは189（0xBD）でSHAabed82d2...。全体2.55GBを検査した数値ではなく、offset1896501248から4096bytesの限定比較である。取得時刻は2026-09-18 19:57:32 UTC。学習曲線は無い。

### アーキテクチャ⇄メトリクスの因果考察
通常/directの同位置の違いはこのrunで再現せず、持続するpage cacheと媒体の差を示す証拠は得られなかった。[Linux open(2)の仕様](https://man7.org/linux/man-pages/man2/open.2.html)に従いalignmentを検査したが、O_DIRECTはWSL hostやdevice側cacheをすべて排除するものではない。過去の一時的なread buffer、memory、I/O等の変化の場所は特定できない。file全体や実モデル処理の安全性は本runだけでは保証しない。

### 既存実験との比較
親runは保存snapshotの全byte比較で1bit差を捕捉した。本runはその根拠を保持し、同じ位置が現在も異なるかを通常/directの異なる経路で1回比較した。成功までfull hashを繰り返す方法でも、壊れたsnapshotや元receiptを書き換える方法でもない。

### 次に有効な実験
元の不一致は未解決として維持する。Windowsの既知NVMe resetと実行環境の動作設定を確認し、host側の診断・環境変更後に固定pinを維持して生成を再開する条件を決める。現時点で同じ対照の盲目的な反復や全体教師の採用は行わない。
