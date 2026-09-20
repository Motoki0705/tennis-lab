---
task: slcs
sequence: 14
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-blcs-meiji-e60-v1
type: run
title: Meiji BLCSの実観測fine-tuning・60 epoch
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: done
config:
  model: multiview_axial_reference
  loss: default
  data: meiji_geometry_replay_v1
  epochs: 60
metrics:
  position_error_m: 0.525525
  position_accuracy_0.3m: 0.294014
  endpoint_error_m: 0.656212
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python
    -m src.tasks.blcs.scripts.train --config-name train_meiji_real_rgb run.output_dir=blcs/train/meiji_real_rgb/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-blcs-meiji-e60-v1
  predictions: knowledge/runs/run-slcs-blcs-meiji-e60-v1/pred_test.npz
  output_dir: /home/kamimura/projects/tennis-lab/outputs/blcs/train/meiji_real_rgb/s42-001/logs/version_0
  curves: knowledge/runs/run-slcs-blcs-meiji-e60-v1/curves.png
  tb_logdir: outputs/blcs/train/meiji_real_rgb/s42-001/logs/version_0
parents:
- run-slcs-real-infer-pilot-v1
- run-slcs-real-geometry-v1
relations: []
tags:
- slcs
- blcs
- real-rgb
- meiji
---

## 考察 / Findings

### 要約
Meijiの多視点幾何ラベルにfine-tuningし、最終epochの別収録test位置誤差は0.525525m、0.3m以内は29.4%。初回から200 epochにはせず60 epochで探索した。

### アーキテクチャ詳細
512幅・8層reference axial BLCSを既存ckptから初期化。outsourceボール観測と選定済みCourtから三角測量し、再投影・速度・高さで支持された連続32–128フレームを採用。trainはvideo_000由来49シーン+既存合成128シーン、valはvideo_001の67、testはvideo_002の52。59.94006fpsを2間引きして29.97003fpsを保持。lr=3e-5、batch=4、60epoch。

### メトリクスの解釈
frontmatterのtest値は最終epoch59。実データ生成用はvalidation位置誤差の最小0.487251m（epoch58）を選び、test選択はしない。3D参照は幾何学的擬似ラベルで、独立した実測3D正解ではない。学習曲線と最良重みは別々に保存した。

学習後にreview catalog用の `config.generation.mode=multi` をデータのメタ情報へ補記した。scene配列・split・教師値は変更していない。変更前JSONと前後SHA-256は `outputs/blcs/analyze/meiji_geometry_metadata/s42-001/` に保存し、生成コードも同じ形式に修正した。

### アーキテクチャ⇄メトリクスの因果考察
仮説: 実カメラ配置・検出誤差・outsource UVに合わせることで合成からのdomain gapを縮められる。合成replayは元の軌道範囲を保持する意図だが、replay無し対照はなく効果を分離できない。

### 既存実験との比較
run-slcs-real-infer-pilot-v1の再投影誤差は別尺度なので、このtest値と直接比較しない。同じ52シーンで旧checkpointと選定checkpointを別の評価runで比較する。

### 次に有効な実験
選定ckptの実映像再投影と旧ckptとの差を同一入力で測り、単眼SLCSの教師として支持されたフレームだけを利用する。
