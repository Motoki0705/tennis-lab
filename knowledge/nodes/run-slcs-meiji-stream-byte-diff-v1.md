---
id: run-slcs-meiji-stream-byte-diff-v1
type: run
title: 正常時と異常時の保存ViTPose streamに1bit差を確認
provider: codex
date: '2026-09-19'
status: done
config:
  device: cpu
  live_checkpoint_reads: 0
  saved_snapshot_full_scans: 1
  chunk_bytes: 1048576
  zip_inspection: central/local headers only; no pickle/model load
  script_sha256: ddf71d8e109416bbf93b0f034d08e825181c0d981404a0ea87127a340f2eaac1
metrics:
  bytes_per_snapshot: 2549075546
  different_bytes: 1
  different_bits: 1
  length_only_bytes: 0
  different_chunks: 1
  first_difference_zero_based: 1896501665
  last_difference_zero_based: 1896501665
  stream_hash_stat_gates_passed: true
  metadata_before_after_equal: true
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-stream-byte-diff-v1
  output_dir: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_stream_byte_diff/s42-001
  comparison: knowledge/runs/run-slcs-meiji-stream-byte-diff-v1/comparison.json
parents:
- run-slcs-meiji-v9-observation-reuse-v2
- run-slcs-meiji-read-stream-capture-v1
relations: []
tags:
- slcs
- integrity
- snapshots
- bit-difference
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: e8483b69a6f1c1c41b818b9ca87f56fc2d4ed4d4
  branch: codex/slcs-real-rgb
  command: env PYTHONPATH=. CUDA_VISIBLE_DEVICES= .venv/bin/python knowledge/runs/run-slcs-meiji-stream-byte-diff-v1/compare.py
    --good-metadata /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_read_stream_capture/s42-001/audit.json
    --bad-metadata /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_v9_read_witness/s42-001/ledger.json
    --output /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_stream_byte_diff/s42-001
---

## 考察 / Findings

### 要約
正常時と異常時の保存snapshotを1回だけ全byte比較した。2549075546 bytes中、offset1896501665の1byte・1bitだけが異なり、0x9Dから0xBD（XOR0x20）となっていた。両snapshotの同一stream二実装hash・期待digest・サイズ・stat検査は通過した。実際に異なるbyte列を捕捉した根拠であり、SSD・RAM等の根本原因を特定する結果ではない。

### アーキテクチャ詳細
live checkpointを開かず、既存good/bad snapshotを同時に順次読み、各bytesをhashlibと_sha256へ渡す。その同じbytesで全差分を比較し、総byte/bit数、offset range、詳細上限4096件、XOR分布を保存した。ZIP central/local headerだけを読み、差分が属するstored memberを特定した。pickle decode・モデルload・GPU処理は行っていない。6 fixture testsと対象ruff/mypyも通過した。

実行時のbase commitは63a2ca9fで、driverは未追跡だった。保存済みscript SHAとの一致を確認して、そのままrepro.commitに固定した。再現commitは当時のbaseとdriver以外の本処理sourceを変更していない。

### メトリクスの解釈
差は0始まりoffset1896501665のみ。good=157、bad=189、LSB0でbit5が1つ異なる。chunk1808のoffset676257、4096byte単位でpage463013のoffset417に相当する（物理memory pageの特定ではない）。差分はZIP entry archive/data/94533706120768のpayload [1886831744,1906492544) 内にある。metadataの前後SHAも一致。detail truncation無しで全差分を保持した。学習曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察
read_streamで得た不一致hashが、別processの検証に加えて今回の全byte比較でも同じ差を示した。ハッシュ文字列の表示誤りだけでは説明できない。一方、snapshotが異なることだけで、元媒体・page cache・RAM・CPU・softwareのどの段階で変化したかは区別できない。1bit差をハードウェア故障の確定診断や特定部品交換の根拠にはしない。

### 既存実験との比較
先行のstdlib対照は期待pinの3snapshotが完全一致し、人物観測driverでは初回ViTPose読取が異なった。本runはそれらの保存内容だけを照合し、異常後のlive再読を成功まで繰り返していない。以前のhashだけの記録から、位置と内容を持つbyte差分の証拠へ進んだ。

### 次に有効な実験
保存した証拠を基に、入力を変更しない方法でstorage/cache/memory経路を切り分ける。Windows側の既知NVMe resetは独立した観測として扱い、因果を決めつけない。安定性を判断する前にMeiji教師を完成扱いせず、採用pinと不一致時停止を維持する。
