---
id: run-court-vit-l4-preflight-20260920
type: run
title: Court ViT L4事前検証：Driveの旧checkpoint不一致で学習前に停止
provider: codex
date: '2026-09-20'
status: failed
config:
  requested_backbones:
  - vits16
  - vits16plus
  - vitb16
  - vitl16
  downstream_width: 768
  batch_size: 8
  max_epochs: 20
  baseline: multiscale-depth3-local-rtx
  observed_drive_checkpoint: B00 / 3 heads / LoRA / 256px
metrics: {}
repro:
  commit: 1a983a939d13df529f5ef0e83d35ff1f376dbfea
  branch: codex/court-vit-size-colab
  command: .venv/bin/python -m scripts.colab.train.court_vit_ablation.launch --run-id
    court-vit-l4-20260920-r4
artifacts:
  run_dir: knowledge/runs/run-court-vit-l4-preflight-20260920
  log: /home/kamimura/.local/state/tennis-lab/colab-launches/court-vit-l4-20260920-r4.log
parents: []
tags:
- court_detection
- dinov3
- colab
- resume
- preflight
---

## 考察 / Findings

### 要約
NVIDIA L4で12入力の転送・6データarchiveのSHA検証・展開まで成功した。Driveのepoch 17 checkpointは論文baselineと別のモデル構成であり、厳密な再開検証が拒否した。学習batchは一度も実行していない。基準checkpointの選択をユーザーへ確認し、不要なGPU課金を避けるためruntimeを停止した。

### アーキテクチャ詳細
予定はS/S+/Lの4段特徴を独立1×1 Convで768次元へ写像し、Transformer・DPT・pose/dense headsをBと一致させる。Bにはadapterを追加しない。期待baselineは4シーン、4 dense heads、LoRA無効、head depth 3、256〜512px。実際のDrive checkpointはB00、KP/SEG/LINEの3 heads、LoRA有効、head depth 2、256px、旧target schemaだった。

### メトリクスの解釈
学習・評価メトリクスはない。GPUはNVIDIA L4、CUDA利用可能。実checkpoint SHA-256は dd3a396841097e60ff1bc0eabcf7b911e97685e251bf8cc441c100b17276e816。epoch番号・global stepだけではモデルの同一性を確認できない。

### アーキテクチャ⇄メトリクスの因果考察
モデル・教師schemaが異なるため、weight-only loadやstrict=Falseによる回避は完全状態resumeにならず、サイズ比較を壊す。設定検証の拒否は正しい。

### 既存実験との比較
今回の事前検証は新たな学習結果ではない。ローカルのmultiscale-depth3 checkpointとDriveの旧LoRA checkpointを同一baselineとして扱わない。

### 次に有効な実験
ユーザーが論文baselineを選ぶ場合は、対応するローカルepoch 17 checkpointを新しいDrive入力名で保存し、元ファイルを上書きせず再開する。旧Drive baselineを選ぶ場合は旧model/data/target契約を明示的に復元して比較条件を組み直す。
