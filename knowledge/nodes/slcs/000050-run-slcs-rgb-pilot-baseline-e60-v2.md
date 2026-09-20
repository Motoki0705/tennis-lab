---
task: slcs
sequence: 50
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-rgb-pilot-baseline-e60-v2
type: run
title: 'SLCS実RGB pilot: baseline 60epoch完了（終端test）'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: done
config:
  model: SLCSFusionModel hidden128/shared4/DINO-downsample2
  loss: SLCSLoss; resolved_config.yaml参照
  data: slcs/real_rgb_pilot_v2, 7 clips
  augmentation_enabled: false
  max_epochs: 60
  seed: 42
metrics:
  player_position_error_m: 2.735102
  player_position_error_median_m: 1.727126
  player_angular_error_deg: 36.775043
  player_angular_error_median_deg: 34.798462
  player_position_accuracy_0.3m: 0.034247
  player_position_accuracy_0.5m: 0.109589
  player_position_accuracy_1.0m: 0.308219
  player_position_accuracy_2.0m: 0.568493
  player_angle_accuracy_10deg: 0.109589
  player_angle_accuracy_15deg: 0.262557
  player_angle_accuracy_30deg: 0.486301
  player_position_pred_b_m: 0.965827
  player_rotation_pred_b_deg: 15.845441
  player_position_conf_error_corr: 0.686756
  player_rotation_conf_error_corr: -0.471586
  ball_position_error_m: 5.591211
  ball_position_error_median_m: 4.965133
  ball_position_accuracy_0.3m: 0.011976
  ball_position_accuracy_0.5m: 0.023952
  ball_position_accuracy_1.0m: 0.035928
  ball_position_accuracy_2.0m: 0.143713
  ball_position_pred_b_m: 2.47156
  ball_position_conf_error_corr: 0.665461
  scene_position_error_m: 4.163157
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
    .venv/bin/python -m src.tasks.slcs.scripts.train --config-name train_real_rgb_pilot
    data.augmentation.enabled=false run.output_dir=slcs/train/real_rgb_pilot_baseline/s42-002
artifacts:
  run_dir: knowledge/runs/run-slcs-rgb-pilot-baseline-e60-v2
  predictions: knowledge/runs/run-slcs-rgb-pilot-baseline-e60-v2/pred_test.npz
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/train/real_rgb_pilot_baseline/s42-002/logs/version_0
  curves: knowledge/runs/run-slcs-rgb-pilot-baseline-e60-v2/curves.png
  tb_logdir: outputs/slcs/train/real_rgb_pilot_baseline/s42-002/logs/version_0
  log: knowledge/runs/run-slcs-rgb-pilot-baseline-e60-v2/queue.log
parents:
- run-slcs-rgb-pilot-baseline-e60-v1
relations: []
tags:
- slcs
- real-rgb
- pilot
- pseudo-teacher
---

## 考察 / Findings

### 要約
設定保存・confidence・checkpoint監視名の不整合を修正したprofileで60epoch完了。選手位置は学習されたがballはtrainでも大誤差が残る。完走を十分な3D精度と解釈しない。

### アーキテクチャ詳細
hidden128・共有4層のSLCSでDINO tokenと2D観測を融合。seed42、60epoch、augmentation.enabled=false。Meiji2 clip＋broadcast5 clipのpilot_v2、min_window_label_ratio=0.5。train: video_000/shanghai/washington、val: video_001/indoorhard、test: eastbourne。Meiji test収録はない。

### メトリクスの解釈
frontmatterとpred_test.npzは学習終端lastモデルのtestで、validation選定モデルの別評価ではない。終端test player=2.7351m、ball=5.5912m、scene=4.1632m。epoch誤差のtrain playerは13.905→1.773m、train ballは9.015→6.572m。終端val player=2.610m、ball=7.034m。curves.pngは選手改善とball頭打ちを示すが、教師は実測GTではない。

### アーキテクチャ⇄メトリクスの因果考察
ballはtrain誤差も約6.6mあり、主にheld-out汎化だけの問題とは考えにくい。後続CPU診断は低い予測分散・小さい入力条件差を確認した。loss/教師分布/入力融合のどれが原因かは未確定で、定数に近い解へ縮退した仮説として扱う。選手は明確に学習が進むため、全モデルが動作していないわけではない。

### 既存実験との比較
parent初回runは学習前にTensorBoard保存で失敗し精度比較は不能。本runは同じ目的の修正後完走。baseline/augmentationの比較では終端testと選定重みを混同せず、対応するselected-conditionsノードに同一target/maskのCPU比較を分離した。選定は保存top3内validation最小のepoch47（0-based）、monitor=4.800303m。testで選んでいない。

### 次に有効な実験
まずballのtrain小規模overfitと教師分布・loss寄与・入力依存を検証し、full Meijiデータ追加だけで問題が解消すると仮定しない。実測3D評価とMeiji test収録を別途確保し、単一seedのpilotから一般的効果を断定しない。
