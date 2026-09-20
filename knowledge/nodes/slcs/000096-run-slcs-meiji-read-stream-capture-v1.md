---
task: slcs
sequence: 96
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-read-stream-capture-v1
type: run
title: ViTPoseの読取byteを同時保存した固定3pass対照
provider: codex
date: '2026-09-19'
status: done
config:
  device: cpu
  model: none; no numpy/torch imports
  planned_reads: 3
  chunk_size_bytes: 1048576
  checkpoint_sha256: 50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc
  capture: same immutable read bytes hashed then written, not a later live copy
metrics:
  completed_passes: 3
  bytes_per_snapshot: 2549075546
  pairwise_comparisons: 3
  differing_snapshot_bytes: 0
  cancelled_passes: 0
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-read-stream-capture-v1
  output_dir: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_read_stream_capture/s42-001
  audit: knowledge/runs/run-slcs-meiji-read-stream-capture-v1/audit.json
parents:
- run-slcs-meiji-v9-observation-reuse-v1
relations: []
tags:
- slcs
- integrity
- cpu
- snapshot
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: 6b3756ec
  branch: codex/slcs-real-rgb
  command: env CUDA_VISIBLE_DEVICES= .venv/bin/python -B knowledge/runs/run-slcs-meiji-read-stream-capture-v1/capture.py
    --checkpoint /home/kamimura/projects/tennis-lab/third_party/GVHMR/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth
    --output-dir /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_read_stream_capture/s42-001
    --passes 3
---

## 考察 / Findings

### 要約
ViTPose固定重みを事前指定3passで読み、hash計算と同じbyte列をsnapshotへ保存した。3passとも固定pinと一致し、保存snapshotを別processで検証しても一致、snapshot同士のbyte差は0だった。この軽量対照では間欠的不一致を再現せず、元の失敗や環境問題を解決済みとは扱わない。

### アーキテクチャ詳細
標準ライブラリのみ、CPU・CUDA_VISIBLE_DEVICES空。各live readの1MiB immutable bytesをhashlib/_sha256全体hash・chunk hashへ渡してから同じbytesをxb snapshotへ書きfsyncする。各snapshotはisolated CPython(-I -S)とsha256sumで各1回検証し、3組のsnapshot間byte比較を行う。異常時は残り予定passを中止し、liveの回復再読は行わない。checkpoint本体3本はoutputsのみで、gitにはコードとJSONを保存する。

### メトリクスの解釈
各snapshotは2,549,075,546bytes、全3passの2providerと保存後検証が固定pin50e33f...に一致。statとbyte countは通過。3組のsnapshot差分は全て0byte・最初の差位置なし、chunk digest列も一致。学習曲線は無い。

### アーキテクチャ⇄メトリクスの因果考察
失敗後の別readによるcopyでは失敗時の内容を保持できないため、hashへ渡した同じread結果をその場で保存した。hash・write間のメモリ異常や後のI/O異常を完全に排除するものではない。本対照はnumpy/torchや人物選択処理を含まないため、その処理後に生じた前runの不一致の原因を切り分け切っていない。

### 既存実験との比較
親runは公開前のViTPose digestだけ固定pinと不一致となったが、読取bytesは保存していなかった。本runはその観測不足を補う手段を検証した。scoutが提案した失敗後copyは別readになるため採らず、同一read streamの保存へ変更した。成功を理由に親runの停止や失敗記録を取り消さない。

### 次に有効な実験
実際の人物観測再利用処理の初回・公開前・公開後の3読取へ同じ記録を組み込み、1回だけ観測付き実行する。全ての既存採用ゲートを維持し、不一致時は残り読取を停止してsnapshotを解析する。
