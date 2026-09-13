---
id: run-court-storage-audit
type: run
title: Scenes全体の容量・完全重複監査
provider: codex
date: '2026-09-13'
status: done
config:
  data: B00-B03 scenes, all regular files
  method: inode accounting; size/prefix/suffix candidates then complete SHA256
metrics:
  files: 58374
  logical_bytes: 102052875844
  unique_inode_bytes: 101309706540
  allocated_unique_inode_bytes: 101414285312
  already_shared_bytes: 743169304
  exact_duplicate_reclaimable_bytes: 2632986999
artifacts:
  run_dir: knowledge/runs/run-court-storage-audit
  metrics: knowledge/runs/run-court-storage-audit/metrics.json
parents: []
relations: []
tags:
- court
- storage
session: 01a0985c-eb87-7310-9657-5411e2818d4e
---

## 考察 / Findings

### 要約
容量の中心はfloat32配列。完全に同じ未共有ファイルは約2.45 GiBであり、重複排除だけでは主因を解消しない。

### アーキテクチャ詳細
全ファイルの論理サイズとinode単位の割当量を別集計した。既存hardlinkを二重に削減候補へ数えない。候補の完全SHA256一致のみ重複に数える。

### メトリクスの解釈
58,374ファイル、論理95.044 GiB。主要容量はCourt RGB/alpha/depth。詳細内訳と重複パスはmetrics.jsonが正本。

### アーキテクチャ⇄メトリクスの因果考察
浮動小数点の数値精度とbyte-planeの冗長性を残したままNPYへ保存しているため、一般的なファイル重複とは別の可逆圧縮が有効。

### 既存実験との比較
先行storage監査ノードなし。

### 次に有効な実験
float32完全一致の圧縮と、immutableなscene asset共有。既存の可変ファイルを無条件hardlinkする移行は行わない。
