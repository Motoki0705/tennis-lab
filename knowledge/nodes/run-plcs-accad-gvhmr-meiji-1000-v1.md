---
id: run-plcs-accad-gvhmr-meiji-1000-v1
type: run
title: GVHMR/ACCAD混合1000シーンの生成と分割検証
provider: codex
date: '2026-09-15'
status: done
config:
  motion_sources: accad_gvhmr
  simulation.num_scenes: 1000
  run.num_workers: 4
  split_group: source_motion_or_gvhmr_rally
metrics:
  generated_scenes: 1000
  failed_scenes: 0
  camera_views: 6000
  frames: 594460
  gvhmr_scenes: 485
  accad_scenes: 515
  gvhmr_motions_sampled: 112
  train_scenes: 800
  val_scenes: 100
  test_scenes: 100
  finite_arrays_checked: 40000
artifacts:
  output_dir: /home/kamimura/projects/tennis-lab/data/plcs/accad_gvhmr_meiji_1000_v1
  split_info: /home/kamimura/projects/tennis-lab/data/plcs/accad_gvhmr_meiji_1000_v1/split_info.json
repro:
  command: /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.plcs.scripts.generate_dataset motion_sources=accad_gvhmr simulation.num_scenes=1000 run.num_workers=4 +run.split_group=motion_source run.output_dir=plcs/accad_gvhmr_meiji_1000_v1 paths.project_root=/home/kamimura/projects/tennis-lab
parents:
- run-plcs-accad-gvhmr-dino-smoke-v1
relations: []
tags:
- plcs
- gvhmr
- dataset
---

## 考察 / Findings

### 要約

GVHMR114モーションとACCAD167モーションを候補に、混合1000シーンを生成した。全40000配列の有限性とsplit間のsource/rally重複なしを確認。52.2Mパラメータの標準PLCS multiview_axial_baseでCPU 1バッチのforward/backwardも成功した。

### アーキテクチャ詳細

共通COCO17から各シーン6仮想カメラへ投影。GVHMRのnative 59.94006fps、ACCADの120fpsを保存し、学習時だけ28/30/60Hzへ同期再サンプリングする。

生成後、各source_kind内で共有make_group_split_map（seed=42、val/test各0.1）を適用し直した。ACCADのgroup keyは完全なmotion_source path、GVHMRはmotion_source_idの最初のコロンより前のclip ID。同じラリーのcam1/cam2を同一splitへまとめた。正本のgroup_assignmentsは成果物のsplit_info.jsonに保存済みであり、再生成時はこの割当からtrain/val/test.txtを復元する。

### メトリクスの解釈

trainはGVHMR387/ACCAD413、valとtestは各GVHMR49/ACCAD51。ランダムサンプリングにより114件中112件のGVHMRモーションが実際に選ばれた。容量は約1.5 GiB。全114件を各1回必ず選ぶ割当ではない。

### アーキテクチャ⇄メトリクスの因果考察

候補の混合重み0.5に対し実際のGVHMR割合は48.5%。両選手を同一ラリー単位で分割することで、同時刻に対応する動作がtrainと評価へ分かれることを防いだ。ただし同一収録内の別ラリーは別groupであり、未見収録への汎化評価ではない。

### 既存実験との比較

親の4シーン試験から1000シーンへ拡張し、空の評価splitを避けて実際の学習入力まで確認した。新規混合val/testであり、旧ACCAD-only固定val/testとの直接比較ではない。生成runであり学習精度や収束曲線はない。

### 次に有効な実験

共有training queueへplcs_accad_gvhmr_meiji_1000_v1_trainを投入済み。標準モデルを新規学習（最大200epoch、bf16-mixed、batch4、compile無効、best1+last保存）し、完了時に別runとして評価を記録する。
