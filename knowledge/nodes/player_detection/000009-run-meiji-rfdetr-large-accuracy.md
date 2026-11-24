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
  canonical_rescore: knowledge/runs/run-meiji-rfdetr-large-accuracy/canonical-rescore.json
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

2026-11-25に #2495 の正準順序対応版 `0d4bf0990f82ae0b8ebf2afe08d3209e583d2364` で、3条件の生予測をCPUから再採点した。各条件6,844画像・26,276選手を照合し、Recallを含む全採点値が保存済み結果と一致した。採点方針のメタデータにIoU同点時の順序が追加された。RF-DETR-Lの出力は人物BBoxとscoreで、姿勢推定の測定は対象外。 [再採点とSHA-256の照合記録](../../runs/run-meiji-rfdetr-large-accuracy/canonical-rescore.json)

同一予測と採点対象を保ち、疑似BBoxの各辺余白を5%・15%に変えて再採点した。PT32のRecall@0.75は5%で13.94%、10%で42.06%、15%で56.87%だった。同じ余白でのRecall@0.50は92.31%、96.81%、97.04%だった。Recall@0.75は疑似BBoxの余白規約に強く依存するため、検出器間の比較の主指標には固定余白でのRecall@0.50を使う。Recall@0.75は余白感度を添えて報告する。再採点は共通の `load_reference_frames(..., box_margin=margin)` で参照を読み、同じ `score_players` に渡す。 [各backendの感度](../../runs/run-meiji-rfdetr-large-accuracy/results.json)

## 実行と成果物

RTX 5060 Ti、PyTorch 2.13.0+cu130、TensorRT 10.14.1.48.post1を使用した。batch 1のFP16 ONNX構造検査が通り、TensorRTの静的planを96.43秒で構築した。全backendのqueueが `done` を記録し、実行HEADは `64fde00dca465a7c2bb27881805f971d0bbaaee4`。各JSONLの6,844件のキー、元画像の境界内の正面積BBox、score範囲、SHA-256を照合した。保存予測をCPUで再採点すると、全backendで採点JSONと一致した。統合担当のJSONL readerでも全件を読み戻した。 [正確な実行コマンドとファイルのSHA-256](../../runs/run-meiji-rfdetr-large-accuracy/results.json)・[準備と再現手順](../../runs/run-meiji-player-model-comparison/rfdetr.md)

RF-DETR-2XLは2026-11-24のユーザー判断により、Roboflow PML 1.0の契約・報告・商用自前運用の条件を理由に比較対象外となった。RF-DETR系列の比較対象はApache-2.0のLのみ。 [日付とライセンス上の理由](../../runs/run-meiji-player-model-comparison/rfdetr.md#checkpoints-and-licenses)

RF-DETR-Lの速度とVRAMは #2496 が公開した96画像と非既定CUDA streamで計測する。この精度runの `summary_impact` は `none`。採用判断には共通条件の速度と下流での検証を用いる。 [速度計測の再現手順](../../runs/run-meiji-player-model-comparison/rfdetr.md#fixed-speed-comparison)
