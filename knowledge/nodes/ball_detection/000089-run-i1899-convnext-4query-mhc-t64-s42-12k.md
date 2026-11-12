---
id: run-i1899-convnext-4query-mhc-t64-s42-12k
type: run
task: ball_detection
sequence: 89
recorded_at: '2026-11-12'
title: 旧基準のCNNのままqueryを4本＋mHCにした条件（T32→T64、各6k）
issue:
- 1899
- 1913
provider: claude
date: '2026-11-11'
status: done
summary_impact: none
config:
  seed: 42
  dim: 256
  layers: 4
  heads: 8
  ffn_dim: 704
  stages: [T32 6000, T64 6000]
  input_frames_per_update: 128
  freeze_updates: 600
  encoder_lr_ratio: 0.1
  initialization: #1514の共通DPT T32 10kのCNN。queryは新規初期化
  gpu: NVIDIA L4
metrics:
  mean_error_px: 36.42065879366264
  median_error_px: 7.5588812828063965
  p95_error_px: 199.28723754882802
  within_20px_rate: 0.7529980227674237
  observed_frames: 85473
  t32_6k/mean_error_px: 42.31901594323226
  query_updates: 12000
repro:
  commit: 929ad65645c1c7522e8921f1b439619d636a0128
  command: .venv/bin/python knowledge/runs/run-ball-query-controls-20261111/control.py launch --session <担当session> --attempt <未使用の番号>
  note: 実行scriptとREADMEは固定commitのknowledge/runs/run-ball-query-controls-20261111/にあり、mainには未統合。
artifacts:
  run_dir: knowledge/runs/run-i1899-convnext-4query-mhc-t64-s42-12k
  results: knowledge/runs/run-i1899-convnext-4query-mhc-t64-s42-12k/stage2-validation-update-00006000.json
  drive_outputs: outputs/ball_detection/train/query_controls/s42-20261111-main
parents:
- run-ball-sequential-length-20261025
relations:
- to: run-ball-sequential-length-20261025
  rel: compares
  note: 旧#1514のquery T64と同じvalidation・採点frame・教師identityで比較
- to: run-i1899-x3d-1query-t64-s42-12k
  rel: compares
  note: 同じvalidationで、queryだけ替えた条件とCNNだけ替えた条件を比較
papers: []
tags:
- ablation
- validation
- single-seed
---

## 何を変えた条件か

#1899は、旧#1514のquery T64（ConvNeXt V2＋1本query、T32 6k→T64 6k、mean 28.20px）と、#1675のmHC query（X3D/CSN＋4本query＋mHC、T64のみ）の差を、1要素ずつ戻して切り分ける。本条件はCNNを旧基準と同じConvNeXt V2のままにし、queryだけを#1675と同じ4本＋mHC（4 streams）に替えた。他の条件（dim256・4層・8 heads・FFN704、seed42、RGB＋MDD 5ch・720p、128 frames/update、T32 6k→T64 6k、各段600更新のCNN凍結とCNN LR比0.1、12k cosineの段順offset 0→6000）は旧基準にそろえた。後半は全重み・named AdamW・sample位置を継承した。条件と手順の正本は[固定版の実行README](https://github.com/Motoki0705/tennis-lab/blob/929ad65645c1c7522e8921f1b439619d636a0128/knowledge/runs/run-ball-query-controls-20261111/README.md)。

## 結果（固定validation、seed42、observed 85,473 frames、source px）

| 条件 | query累積 | mean | median | P95 | 20px以内 |
|---|---:|---:|---:|---:|---:|
| 旧#1514 query T64（基準） | 12,000 | 28.20 | 7.66 | 116.89 | 76.29% |
| 本条件 T32（前半） | 6,000 | 42.32 | 12.70 | 209.35 | 68.68% |
| 本条件 T64（後半） | 12,000 | 36.42 | 7.56 | 199.29 | 75.30% |

基準に比べ、meanは+8.22px、P95は+82.40pxと悪い。一方medianは-0.10pxとわずかに良く、20px以内率は-0.99ポイント。典型的な誤差は変わらず、外れの大きいframeが増えた形である。query数とmHCを同時に変えているので、mHC単独の効果には分けられない。

## 照合と限界

- 基準と本条件のvalidationは、採点frame（`scored_frame_sha256` 2501f696…）・target・教師のidentity hashが完全に一致した。件数が同じというだけの確認ではない。
- 両段の評価JSONのSHA-256はDriveの`_SNAPSHOT.json`の記載と一致した（T32: `64bc7d16efb74df907d97907fb2762a45d352cbfad7a623e46cfdc37b9467682`、T64: `81a96901b0506bf6e89c9d19ccf27b83f35ed2e0ec1a8f1441bb9411e26b095b`）。両段とも再開はない。評価JSON・継続性・完了印と、2026-11-11時点の比較表（`complete: false`）を[run_dir](../../runs/run-i1899-convnext-4query-mhc-t64-s42-12k/)に置いた。checkpoint本体（T64のDrive ID `1QyxrJrI4zB1OZP1bhUdq8YkZ4IeRbaun`）はDriveに保持し、再loadはしていない。
- seed42の1回・validationだけの比較で、seed間のばらつきと独立testは未評価。
- #1899の3条件目（ConvNeXt V2＋1本query、T64→T64。前半の系列長だけを変える）は、前半T64の5,500/6,000更新のcheckpointで止まっている（`outputs/ball_detection/train/query_controls/s42-20261111-main/length-s1-T64/checkpoint-00005500-0281b1ba/`、2026-11-11）。3条件そろった比較は未完了で、#1899の結論はまだ出ていない。
- sceneの採用モデル・checkpointは変更しない。このノード単独では現在の判断を変えないため、summary_impactはnoneとする（summaryの記述は[12kのgroup](000088-group-i1675-capacity-12k-s42.md)の変更で更新した）。
