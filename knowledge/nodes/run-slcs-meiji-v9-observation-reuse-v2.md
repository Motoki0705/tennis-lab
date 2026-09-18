---
id: run-slcs-meiji-v9-observation-reuse-v2
type: run
title: 人物観測の初回ViTPose読取で不一致byte列を捕捉し公開前停止
provider: codex
date: '2026-09-19'
status: failed
config:
  device: cpu
  cuda_visible_devices: ''
  omp_mkl_threads: 4
  source_observations: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004
  target_observations: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-005
  max_live_checkpoint_reads: 3
  same_stream_snapshot: true
  automatic_retry: false
metrics:
  captured_live_reads: 1
  camera_plans: 0
  published_cameras: 0
  captured_bytes: 2549075546
  expected_pin_matches: false
  same_stream_hash_implementations_agree: true
  saved_snapshot_matches_capture: true
  stat_unchanged: true
  completed_prior_input_files: 928
  prior_inputs_before_after_equal: true
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-observation-reuse-v2
  output_dir: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_v9_observation_reuse/s42-002
  read_witness_directory: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_v9_read_witness/s42-001
  ledger: knowledge/runs/run-slcs-meiji-v9-observation-reuse-v2/ledger.json
  failure: knowledge/runs/run-slcs-meiji-v9-observation-reuse-v2/failure.json
parents:
- run-slcs-meiji-v9-observation-reuse-v1
- run-slcs-meiji-read-stream-capture-v1
relations: []
tags:
- slcs
- meiji
- integrity
- captured-stream
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: 78ac29cb3976110db8c6597f326951f99b27b302
  branch: codex/slcs-real-rgb
  command: env PYTHONPATH=. CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    .venv/bin/python -B knowledge/runs/run-slcs-meiji-v9-observation-reuse-v2/witness.py
    --source /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004
    --target /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-005
    --output /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_v9_observation_reuse/s42-002
    --evidence /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_v9_read_witness/s42-001
---

## 考察 / Findings

### 要約
旧観測再利用driverのViTPose読取だけに同一streamの保存を加え、1回実行した。初回pin照合で異なるSHAを検出し、人物選択前・公開0で停止した。不一致の読取byte列をsnapshotに保存し、独立Pythonとsha256sumでも同じ不一致を確認した。原因は未確定だが、今回の差をhash表示だけの問題として扱うことはできない。

### アーキテクチャ詳細
元driver・pins・選択・公開条件を維持し、dual_sha256 bindingのViTPoseだけを置き換えた。最大3回のlive read（初回・公開前・公開後）の各1MiB immutable bytesを、同じstream内で2実装hashとsnapshotへ渡す。snapshotをisolated Pythonと外部sha256sumで照合し、各段階のledgerをatomic保存する。異常後の再読と4回目は拒否する。今回は初回digestの登録前に停止したため、旧例外handlerはViTPoseを再読していない。wrapperの8 CPU fixture testsも通過した。

### メトリクスの解釈
2549075546 bytesを読み・保存し、dev/inode/size/mtime/ctimeは前後一致。2実装ともa8e786c151eedd1dca57ea5b108b901b6a6cb8731a3990b213bdffa418751905となり、固定期待値50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bcと不一致。保存snapshotの独立Python2実装とsha256sumもa8e786...で、chunk hashと長さもcaptureに一致した。先行して登録済みの928入力は前後一致。targetのpeople/detections NPZはともに0で、元観測を変更していない。新規学習はなく曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察
モデルforwardや人物選択より前、CUDA無効の初回読取で発生した。保存後の独立照合でも同じbyte列が観測されているため、元hash計算だけの一過性状態に原因を限定できない。live file更新の有無をstatだけで完全に証明できず、storage経路・page cache・memory・software等の原因もまだ分離できない。ハードウェア故障を断定しない。

初回とは本processのViTPose読取を指し、その前に928入力のhashとDINO pin照合は成功している。今回のdriverはtorch・numpy等をimport済みで、成功したstdlibのみの対照とはprocess状態が異なる。これは交絡要因の候補であり、importやmemory pressureが原因という証拠ではない。

### 既存実験との比較
先行の3回capture対照は期待pinと一致した。今回の保存chunk記録との比較では1MiB chunk index1808（offset1895825408）だけ異なった。全byteの差分位置・bit数は別の保存snapshot比較runで確認する。以前の失敗後にlive fileを読み直して一致した事実と、今回保存した不一致内容を区別する。

### 次に有効な実験
正常時と異常時の保存snapshotだけを比較し、差分offset・byte・bitとarchive内の位置を記録する。live checkpointを再読して成功まで繰り返さず、原因の切り分け後にMeiji人物観測・3D教師生成へ戻る。
