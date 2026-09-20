---
id: run-court-vit-l4-b-smoke-20260920
type: run
title: Court ViT-B：Colab L4で512px・batch 8のGPU smoke成功
provider: codex
date: '2026-09-20'
status: done
parents:
- run-court-vit-l4-preflight-20260920
config:
  backbone: dinov3_vitb16
  encoder_train_mode: frozen
  targets:
  - kp
  - seg
  - line
  - semantic_line
  batch_size: 8
  train_scales:
  - 512
  precision: bf16-mixed
  max_steps: 1
  fast_dev_run: true
metrics:
  peak_allocated_bytes: 19865299456
repro:
  commit: ab824a635182d878c5523a564d833809bde2e124
  branch: codex/court-vit-size-colab
  command: bash scripts/colab/run.sh run court_vit_ablation --gpu L4 --ref ab824a63
    -- --size b --smoke
artifacts:
  run_dir: knowledge/runs/run-court-vit-l4-b-smoke-20260920
  log: /home/kamimura/.local/state/tennis-lab/colab-launches/court-vit-l4-20260920-r6.log
tags:
- court_detection
- dinov3
- colab
- l4
- smoke
---

## 考察 / Findings

### 要約
L4で実データ8枚・512pxのforward/backwardとvalidation 1 batchが完了した。OOMは発生せず、consoleのtrain loss 80.90、val loss 41.90は有限だった。これは使い捨ての動作確認であり、学習済みBの精度評価でも20 epochの本学習結果でもない。

### アーキテクチャ詳細
169M parameters、うち83.4Mが学習対象、85.7Mが凍結DINOv3。Transformer/DPT/pose/4 dense headsは論文baselineと同じ。Bにはfeature adapterを追加していない。epoch 17 checkpointは完全状態を検証した後、smoke内でresumeを解除して新規headを1 stepだけ更新した。

### メトリクスの解釈
peak_allocated_bytesはtorch.cuda.max_memory_allocatedの実値。約18.5 GiBであり、device全体の使用量ではない。lossはconsole表示の丸め値で、比較実験の精度指標には使わない。S/S+/LのGPU実行はこのrunでは未確認。

### アーキテクチャ⇄メトリクスの因果考察
BF16でBの512px/batch 8はL4へ収まった。起動時間の大半はSynthetic pose/targetの事前検証で、DataModule構築とDataset構築に同一検証の重複があった。後続コードでは検証済みDatasetそのものを再利用し、検証を残したまま重複を除いた。

### 既存実験との比較
誤って指定されていた旧LoRA checkpointを上書きせず、ユーザー承認したSHA b863df1f...の論文baselineをDriveへ別名追加した。全6archiveのSHA検証・正しいdata/配置も通過した。

### 次に有効な実験
本学習は別runでBの完全状態を復元し、global step増加とDrive checkpoint保存を確認する。S/S+/Lは4段の1×1 adapterで後段を768幅へ固定する。
