---
id: run-slcs-meiji-inference-repeatability-v1
type: run
title: 'Meiji同一652frameのViTPose反復: 全出力配列一致'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: done
config:
  model: ViTPose-H BF16 batch8 flip_test
  input: video_000/clip_007 cam0 player0; saved temporal tracks
  seed: 42
  predictions: 2
  determinism_intervention: false
metrics:
  frames: 652
  joints: 17
  compared_scalar_elements: 33252
  differing_elements: 0
  coordinate_max_absolute_difference_px: 0.0
  confidence_max_absolute_difference: 0.0
  hash_rows: 8
  hash_mismatch_rows: 0
  elapsed_seconds: 106.98457846199926
repro:
  commit: 777d339101e162739c12e37cf52dbb4fe75cce6a
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: timeout --kill-after=5s 590s env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -B knowledge/runs/run-slcs-meiji-inference-repeatability-v1/probe.py
    --output-dir outputs/tennis_scene/analyze/meiji_inference_repeatability/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-inference-repeatability-v1
  output_dir: outputs/tennis_scene/analyze/meiji_inference_repeatability/s42-001
  diagnostics: knowledge/runs/run-slcs-meiji-inference-repeatability-v1/results.json
  predictions: knowledge/runs/run-slcs-meiji-inference-repeatability-v1/prediction_1.npz
  log: knowledge/runs/run-slcs-meiji-inference-repeatability-v1/queue.log
parents:
- run-slcs-meiji-checkpoint-hash-gpu-control-v1
- run-slcs-known-memory-cpu-audit-v1
relations: []
tags:
- slcs
- meiji
- integrity
- repeatability
---

## 考察 / Findings

### 要約
同じ652frame・同じtracking request・同じViTPose instanceで2回推論し、float32出力(652,17,3)が全33,252要素で完全一致した。checkpointの4段階とvideo/trackの開始終了、計8行の独立2実装hash比較も一致した。間欠的不一致の原因解決や一般的なhardware正常性は証明しない。

### アーキテクチャ詳細
ViTPose-H、BF16、batch8、flip test、seed42。保存済みtemporal trackingのplayer0 boxesから同一requestを作り、推論間にモデルを再ロードせず各結果をCUDA明示同期後CPUへ保存した。入力requestの非破壊と有限性を確認。deterministic algorithmsやOpenSSL命令設定への介入なし。checkpointは既知SHA、videoは保存metadataのSHA、trackは開始時の二実装一致値を基準とする。

### メトリクスの解釈
座標・confidence最大差0、異なる要素0。約107秒、再試行なしで終了コード0。各hashは同じimmutable chunkをOpenSSLとCPython _sha256へ渡し、読み取り長とinode/size/mtime/ctimeも照合した。8行は同じ3種類のファイルを各phaseで測ったもので、8回の独立した推論実験ではない。学習を伴わず収束曲線なし。

### アーキテクチャ⇄メトリクスの因果考察
この入力・設定で推論配列の再現性が得られたため、checksumの不一致だけから推論数値も壊れているとは結論できない。一方、有限回の成功で過去の不一致を取り消せず、別入力・長時間・学習の再現性も未検証。

### 既存実験との比較
過去のhash診断は計算したdigest中心だった。本runは実推論配列を同時に保存し比較した。既知メモリ384比較と今回の結果はいずれも成功したが、異常を恒久修復した証拠ではない。

### 次に有効な実験
productionのファイル照合で独立二実装のdiscordanceを検出して即時停止させ、固定checkpointの期待SHAをrecipeへ明示する。品質閾値を維持して観測生成を再開し、結果をclipごとに検証する。以前のreceiptを新hashへ書き換えない。
