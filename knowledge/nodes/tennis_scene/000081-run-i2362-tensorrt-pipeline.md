---
id: run-i2362-tensorrt-pipeline
type: run
task: tennis_scene
sequence: 81
recorded_at: '2026-11-22'
title: 本番パイプラインのTensorRT FP16統合検証
issue: 2362
provider: codex
date: '2026-11-22'
status: failed
summary_impact: none
config:
  reference: PyTorch FP32 batch 1
  image_candidate_batch: 2
  gvhmr_candidate_batch: 1
  gvhmr_length_profile: [2, 145, 177]
  smoke_clip: video_000/clip_037
  smoke_candidate_batch: 1
  tensorrt_version: 10.14.1.48.post1
  gpu: NVIDIA GeForce RTX 5060 Ti
  compute_capability: [12, 0]
metrics:
  engine_builds_completed: 5
  models_meeting_phase1_criteria: 4
  dino_iou_p05: 0.9305638746573003
  dino_score_abs_p95: 0.01723533868789671
  gvhmr_incam_joint_l2_p95_mm: [8.354766560618225, 8.810336159216988]
  gvhmr_global_joint_l2_p95_mm: [22.632618174942746, 20.394282648732712]
  smoke_frames: 292
  smoke_smpl_valid_frames: [289, 289, 289, 289]
repro:
  commit: f5bbcb0c2f2953eb28e8f700b62f80081655aad8
  queue_job: 0-8204663892298719542_2026-11-22T08-28-27-701280457Z_2349504_i2362-phase2-rebuild.job
artifacts:
  run_dir: knowledge/runs/run-i2362-tensorrt-pipeline
  attempts: knowledge/runs/run-i2362-tensorrt-pipeline/attempts.json
  metrics: knowledge/runs/run-i2362-tensorrt-pipeline/metrics.json
  smoke: knowledge/runs/run-i2362-tensorrt-pipeline/smoke.json
  output_dir: outputs/tensorrt/qualification/phase2-b2-v2
parents:
- run-i2297-clipreid-tensorrt
- run-i2303-vit-tensorrt
- run-i2304-dino-tensorrt
- run-i2305-gvhmr-tensorrt
---

共有 `.venv` の本番export・buildコマンドで、5モデルのONNXとTensorRT FP16 engineを作成した。同じ固定入力に対するPyTorch FP32 B=1を参照とし、本番ラッパーの出力を照合した。画像モデルのB=2とGVHMRのB=1は検証条件。 [実行手順](../../runs/run-i2362-tensorrt-pipeline/README.md)

| モデル | 指標 | Phase 1 TRT FP16 B=1 | 本番ラッパー TRT FP16 | Phase 1の判定 |
|---|---|---:|---:|---|
| CLIP-ReID | cosine P05、FP32とのID一致 | 0.999987、236/236 | 0.999987、236/236 | 範囲内 |
| ViTPose-H | xy L2 P95 / 最大 (px)、confidence差 P95 | 0.02471 / 0.302、0.00159 | 0.02475 / 0.661、0.00154 | 範囲内。有効maskも一致 |
| HMR2 | cosine P05、相対L2 P95 | 0.999992、0.00401 | 0.999993、0.00370 | 範囲内 |
| GVHMR | カメラ座標のCOCO-17差 P95 (mm)、L=145 / 177 | 8.61 / 8.72 | 8.35 / 8.81 | 範囲内。回転P95は最大0.0774°、length・maskは全件一致 |
| DINO GridSample | IoU P05、score差 P95、欠落 / 追加 | 0.93252、0.01726、3 / 2 | 0.93056、0.01724、2 / 4 | IoU ≥ 0.98とscore差 ≤ 0.01を両計測で未達 |

DINOは64画像を照合した。欠落2/494と追加4/496は1%以下。xyxy絶対差P95はPhase 1の0.707 pxから今回の1.241 pxになった。IoUとscore差の単体基準未達は常設化前のPhase 1から観測されている。2026-11-22の判断により、現行GridSample版を明示選択の比較候補とし、採用は #2363 の全クリップE2Eで判断する。 [DINOのPhase 1記録](000078-run-i2304-dino-tensorrt.md)・[モデル別の実測値](../../runs/run-i2362-tensorrt-pipeline/metrics.json)・[全クリップE2E #2363](https://github.com/Motoki0705/tennis-lab/issues/2363)

GVHMRでは重力整列座標のCOCO-17差P95が22.63 / 20.39 mmだった。Phase 1でもこの座標で23.50 / 18.32 mmを記録している。今回の比較コードは両座標に10 mmを適用し、GVHMR比較を終了値1とした。Phase 1の判定対象であるカメラ座標と全1,288フレームのlength・maskを、保存済み指標から再照合した。 [GVHMRのPhase 1記録](000076-run-i2305-gvhmr-tensorrt.md)・[元の終了状態](../../runs/run-i2362-tensorrt-pipeline/attempts.json)

`meiji_4cam` の `video_000/clip_037` は5モデルすべてTensorRT B=1でsmokeを実施した。4カメラ・292フレームでシーン出力が `ok` になり、4選手の有効SMPLフレームは各289。publicationに記録した5モデルのengine digestはbuildのmanifestと一致し、5モデルのforwardを記録した。 [smokeの出力](../../runs/run-i2362-tensorrt-pipeline/smoke.json)

今回検証したTensorRT設定は明示選択用。標準構成の採用判断は維持した。
