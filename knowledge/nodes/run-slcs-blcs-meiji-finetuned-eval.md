---
id: run-slcs-blcs-meiji-finetuned-eval
type: run
title: Meiji BLCS改善重み・別収録test評価
provider: codex
date: '2026-09-18'
status: done
config:
  dataset: meiji_geometry_replay_v1
  split: test
  precision: 32-true
  checkpoint: blcs/real-rgb-meiji-e60-v1.ckpt
metrics:
  position_error_m: 0.4789881706237793
  position_accuracy_0.3m: 0.38943663239479065
  endpoint_error_m: 0.5972864031791687
artifacts:
  run_dir: knowledge/runs/run-slcs-blcs-meiji-finetuned-eval
  predictions: knowledge/runs/run-slcs-blcs-meiji-finetuned-eval/pred_test.npz
  output_dir: outputs/blcs/evaluate/meiji_finetuned/s42-001
parents:
- run-slcs-blcs-meiji-e60-v1
relations:
- to: run-slcs-blcs-meiji-baseline-eval
  rel: compares
tags:
- slcs
- blcs
- real-rgb
- evaluation
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  command: CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python
    -m scripts.analysis.evaluate_blcs_real --checkpoint blcs/real-rgb-meiji-e60-v1.ckpt
    --output blcs/evaluate/meiji_finetuned/s42-001
---

## 考察 / Findings

### 要約
旧checkpointと60epoch fine-tuningからvalidationで選んだcheckpointを、同じ別収録52シーンで比較する独立のCPU評価run。

### アーキテクチャ詳細
512幅・8層reference axial BLCS。両runでseed42、3視点、32–128フレーム、float32、augmentation無し。同じdata configでcheckpoint stateをstrict loadした。

### メトリクスの解釈
{"position_error_m": 0.4789881706237793, "position_accuracy_0.3m": 0.38943663239479065, "endpoint_error_m": 0.5972864031791687}。参照値はoutsource UVと学習Court校正から得た幾何擬似3D。実測3D正解の誤差ではない。

### アーキテクチャ⇄メトリクスの因果考察
比較の保存target_position、mask、scene_ids、target_frame_contractは完全一致を確認した。旧重み3.211951mから改善重み0.478988mへ低下し、同一観測条件への適応を示す。未知会場への一般化はこの評価だけでは判断できない。

### 既存実験との比較
学習runの最終epoch test=0.525525mと異なり、改善側はvalidation最良epoch58のcheckpointを評価した。testの良否でcheckpoint選択を変更していない。

### 次に有効な実験
実映像の再投影・軌道速度・支持率を確認し、支持された擬似ラベルでSLCSのRGB寄与と欠損耐性を評価する。
