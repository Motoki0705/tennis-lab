---
task: slcs
sequence: 133
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-real-rgb-dino-crc-v1
type: run
title: '全体版のRGB NPZ後続CRC点検: 1回のreadで173/173通過'
provider: codex
date: '2026-09-19'
status: done
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  data: slcs/real_rgb_v1
  repeats: 1
  check: Python ZipFile.testzip
metrics:
  archives: 173
  crc_failed: 0
repro:
  commit: e454473c28e5a591dc892586e8575e1b29daa6cb
  command: >-
    env CUDA_VISIBLE_DEVICES= .venv/bin/python
    knowledge/runs/run-slcs-real-rgb-dino-crc-v1/probe.py
    --dataset-root data/slcs/real_rgb_v1
    --output-dir outputs/slcs/analyze/real_rgb_dino_crc/s42-takeover-001
artifacts:
  run_dir: knowledge/runs/run-slcs-real-rgb-dino-crc-v1
  output_dir: outputs/slcs/analyze/real_rgb_dino_crc/s42-takeover-001
parents: [run-slcs-full-real-rgb-no-ball-smooth-start-failure-v1]
tags: [slcs, rgb, crc, read-only, limited-check]
---

## 考察 / Findings

### 要約

学習開始時のCRC失敗後、全61clip・173cameraのDINO NPZを一度だけ点検し、全て通過した。
先の失敗を取り消す結果ではなく、後続のこの読込では永続破損を特定できなかったという限定結果。

### アーキテクチャ詳細

dataset.jsonのclip一覧・camera数と実archive inventoryを照合し、各ZIP memberのCRCを1回検証した。
データ書換え・重み読込・GPU使用・retryなし。結果を全path・byte数付きで保存した。
添付probe.pyが実行sourceである。

### メトリクスの解釈

173archive、失敗0。教師精度・DINO数値の意味・環境全体の安定性を保証する検査ではない。
学習runではなく収束曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

間欠的な読込異常の可能性は残るが根本原因は未確定。このpassを根拠にCRC検証を無効化しない。

### 既存実験との比較

先行CPU full smoke成功と今回の学習開始失敗の両方を保持する。破損fileを推測して差し替えていない。

### 次に有効な実験

読込errorへclip/camera/archive pathとcauseを付ける修正を12CPU testsで確認後、同条件を新runで再実行する。
hardware原因調査や無制限のCRC反復には進まない。
