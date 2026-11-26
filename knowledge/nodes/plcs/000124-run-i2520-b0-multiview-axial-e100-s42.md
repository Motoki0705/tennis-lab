---
id: run-i2520-b0-multiview-axial-e100-s42
type: run
task: plcs
sequence: 124
recorded_at: '2026-11-25'
title: i2520_b0_multiview_axial_e100_s42
issue: 2520
provider: gemini
session: ec99afa3-0ddf-4af9-b92f-9e3912cd1273
date: '2026-11-25'
status: done
summary_impact: none
config:
  model: multiview_axial_base
  loss: canonical
  data: multiview_sequence
metrics:
  position_error_m: 0.457989
  angular_error_deg: 41.75
  position_accuracy_0.5m: 0.630609
  angle_accuracy_15deg: 0.288328
  canonical_mpjpe_m: 0.248809
  canonical_pck_0.1m: 0.05642
repro:
  commit: d08f191cc66ac3a50980bcd2a19de4f7c40d2551
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.plcs.scripts.train
    model=multiview_axial_base data=multiview_sequence loss=canonical data.scene_dir=plcs/single_object_mixed
    training.trainer.max_epochs=100 training.warmup_steps=500 training.trainer.check_val_every_n_epoch=5
    training.trainer.accumulate_grad_batches=4 training.compile.enabled=false training.checkpoint.save_top_k=1
    training.checkpoint.save_last=true run.output_dir=plcs/i2520_b0_multiview_axial_e100_s42
    run.seed=42
artifacts:
  run_dir: knowledge/runs/run-i2520-b0-multiview-axial-e100-s42
  log: .training_queue/logs/0-8204421196433493597_2026-11-25T03-53-23-566506402Z_24640_i2520_b0_multiview_axial_e100_s42.log
  output_dir: outputs/plcs/i2520_b0_multiview_axial_e100_s42/logs/version_0
  curves: knowledge/runs/run-i2520-b0-multiview-axial-e100-s42/curves.png
  tb_logdir: outputs/plcs/i2520_b0_multiview_axial_e100_s42/logs/version_0
parents: []
relations:
- to: run-i2520-v2-m0-mono-gvhmr-base-e100-s42
  rel: compares
  note: 同一データ（single_object_mixed）・同一更新予算（100エポック・実効バッチ16・seed 42）における多視点参照基準（3〜5カメラ入力）と単眼GVHMR基本形の比較
papers: []
tags:
- plcs
- multiview-axial
- baseline
- i2475
- i2520
---

## 考察 / Findings

### 実験目的と設定
- 親Issue #2475（子Issue #2520）の単眼GVHMR・HTD-Refine比較実験における**既存多視点参照基準（B0）**として、`PLCSMultiViewAxialModel`（`multiview_axial_base`、52.2Mパラメータ、`loss=canonical`）を固定・ブロードキャスト混在合成データ `plcs/single_object_mixed` 上で学習した。
- 予算統一条件: `max_epochs=100`, `warmup_steps=500`, 実効バッチサイズ 16（`batch_size=4 × accumulate_grad_batches=4`）, `seed=42`, `bf16-mixed`, `compile=false`（RTX 5060 Ti 16GB・約55分）。入力は各シーン3〜5視点の2D COCO17＋コートKPである。

### 観測（test split / 学習完了後評価）
- `single_object_mixed` test split（Lightning `trainer.test`、best checkpoint `plcs-epoch=99.ckpt`）:
  - `position_error_m = 0.4580 m`（`position_accuracy_0.5m = 63.06%`）
  - `angular_error_deg = 41.75°`（`angle_accuracy_15deg = 28.83%`）
  - `canonical_mpjpe_m = 0.2488 m`（`canonical_pck_0.1m = 5.64%`）
- 共通オフライン評価スクリプトによる合成3 split横断比較（詳細は `run-i2520-cross-eval-and-htd-stage-b`）では、`fixed` で位置誤差 `0.361 m` / 姿勢角誤差 `45.14°`、`broadcast` で `0.558 m` / `57.41°`、`mixed` で `0.515 m` / `49.79°`、CPU推論遅延は `1.50〜3.22 ms/frame` であった。

### 考察と限界
- 多視点入力（3〜5カメラの幾何的手掛かり）によりコート絶対位置精度は `0.36〜0.56 m` と単眼モデル群（`0.84〜2.02 m`）を大きく上回る。
- 一方で姿勢方位角（`angular_error_deg`）は `41.8〜57.4°` に留まり、単眼GVHMR系統（`11.7〜34.1°`）より大幅に大きい。既知のPareto構造（`group-i545-loss-head-tuning`）と同様に、位置回帰と回転・局所姿勢の最適配分が競合している可能性がある（仮説）。
- 限界: 合成データ・単一seed（42）の結果であり、`PLCSMultiViewAxialModel` は固定コートKP入力契約のため実動画4カメラのキャリブレーション済み単眼経路とは直接同一条件ではない。
- `summary_impact: none` の理由: 本ノードは多視点参照基準の単体記録であり、PLCS全体の判断更新は横断評価 `run-i2520-cross-eval-and-htd-stage-b` および統括群 `group-i2520-monocular-gvhmr-htd` で実施する。
