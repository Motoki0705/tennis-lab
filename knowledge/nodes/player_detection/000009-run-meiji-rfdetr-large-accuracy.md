---
id: run-meiji-rfdetr-large-accuracy
type: run
task: player_detection
sequence: 9
recorded_at: '2026-11-24'
title: Meiji 4cam 固定画像での RF-DETR-L 精度と半精度出力の照合
issue: 2487
provider: codex
date: '2026-11-24'
status: done
summary_impact: none
config:
  model: rfdetr-l
  checkpoint: rf-detr-large-2026.pth
  benchmark_id: meiji_4cam/multiview-player-evaluation-v1
  bbox_margin_per_side: 0.10
  min_person_score: 0.25
  batch_size: 1
  input_height: 704
  input_width: 704
  gpu: NVIDIA GeForce RTX 5060 Ti
  tf32: false
metrics:
  evaluation_images: 6844
  eligible_player_boxes: 26276
  recall_iou_050_pt32: 0.9680697214
  recall_iou_050_pt16: 0.9681838940
  recall_iou_050_trt16: 0.9681458365
  recall_iou_075_pt32: 0.4206119653
  recall_iou_075_pt16: 0.4203075049
  recall_iou_075_trt16: 0.4208783681
  pt16_vs_pt32_xyxy_abs_error_p95_px: 0.03857421875
  trt16_vs_pt32_xyxy_abs_error_p95_px: 0.4041748046875
  onnx_export_seconds: 13.679644171
  trt_engine_build_seconds: 96.430636659
repro:
  commit: 64fde00dca465a7c2bb27881805f971d0bbaaee4
  branch: codex/parent-2492-issue-2487-rfdetr
artifacts:
  run_dir: knowledge/runs/run-meiji-rfdetr-large-accuracy
  results: knowledge/runs/run-meiji-rfdetr-large-accuracy/results.json
  output_dir: .worktree-outputs/parent-2492-issue-2487-rfdetr/player-model-comparison/rfdetr-l
parents:
- run-meiji-multiview-player-evaluation
relations: []
papers: []
tags:
- meiji-4cam
- rf-detr
- fp16
- tensorrt
---

## 固定画像での人物検出

#2495 の代表6 clip × 4カメラから、保存済みGVHMR推論フレームの6,844画像で RF-DETR-L を比較した。採点対象は配置済みSMPLの再投影に基づく26,276件の選手BBox。各辺10%の余白と品質マスクを固定し、人物score ≥0.25で一対一IoU対応を採点した。他の人物のラベル状態は不明で、PrecisionとAPはN/A。再構成が現行のDINO・ViTPose観測に依存するため、数値はこの疑似GTに対する一致度を表す。 [参照の品質監査](../tennis_scene/000084-run-meiji-multiview-player-evaluation.md)

| backend | IoU 0.50 で対応した選手 | Recall@0.50 | Recall@0.75 | 対応IoU中央値 | 小人物の Recall@0.50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PT32 | 25,437 | 96.81% | 42.06% | 0.7339 | 95.27% |
| PT16 | 25,440 | 96.82% | 42.03% | 0.7339 | 95.29% |
| TRT16 | 25,439 | 96.81% | 42.09% | 0.7339 | 95.31% |

PT32との差は、PT16で元座標の絶対誤差P95が0.039 px、score差P95が0.00196、対応IoU P05が0.9963。TRT16ではそれぞれ0.404 px、0.00501、0.9620だった。両条件は双方向98%以上の対応率を含む事前の基準を満たした。疑似GTのRecall@0.50の差は各0.012ポイント以内だった。 [予測間の照合と採点指標](../../runs/run-meiji-rfdetr-large-accuracy/results.json)

同一予測と採点対象を保ち、疑似BBoxの各辺余白を5%・15%に変えて再採点した。PT32のRecall@0.75は5%で13.94%、10%で42.06%、15%で56.87%だった。余白による差がbackend間の差より大きい。比較表では同じ余白での値を使い、IoU 0.75の順位を解釈する際はこの感度を考慮する。手順は共通の `load_reference_frames(..., box_margin=margin)` で参照を読み、同じ `score_players` に渡す。 [各backendの感度](../../runs/run-meiji-rfdetr-large-accuracy/results.json)

## 実行と成果物

RTX 5060 Ti、PyTorch 2.13.0+cu130、TensorRT 10.14.1.48.post1を使用した。batch 1のFP16 ONNX構造検査が通り、TensorRTの静的planを96.43秒で構築した。全backendのqueueが `done` を記録し、実行HEADは `64fde00dca465a7c2bb27881805f971d0bbaaee4`。各JSONLの6,844件のキー、元画像の境界内の正面積BBox、score範囲、SHA-256を照合した。保存予測をCPUで再採点すると、全backendで採点JSONと一致した。統合担当のJSONL readerでも全件を読み戻した。 [正確な実行コマンドとファイルのSHA-256](../../runs/run-meiji-rfdetr-large-accuracy/results.json)・[準備と再現手順](../../runs/run-meiji-player-model-comparison/rfdetr.md)

速度・VRAMは #2496 の固定入力を待っている。TRT16の精度実行時に、既定CUDA streamでは追加同期が生じうる旨のTensorRTログが出たため、速度を比較する条件では共通harnessのstream設定を照合する。RF-DETR-2XLの3条件にはPML 1.0とローカル評価を許すRoboflowアカウント・プランの確認、および公式重みが必要。採用判断には共通条件の速度と下流での検証が必要な段階なので、この精度runの記録時点で `summary_impact` は `none`。 [#2496](https://github.com/Motoki0705/tennis-lab/issues/2496)・[2XLの取得条件](../../runs/run-meiji-player-model-comparison/rfdetr.md#checkpoints-and-licenses)
