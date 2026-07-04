---
id: run-i590-courtkp14
type: run
title: i590_courtkp14
issue: 590
provider: claude
session: e6b05f74-5cf0-4470-9219-ca1b8fb11eec
date: '2026-07-04'
status: done
config:
  model: multiview_axial_split
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.189139
  position_error_std_m: 0.143086
  position_error_median_m: 0.154564
  angular_error_deg: 6.28266
  angular_error_std_deg: 6.182175
  angular_error_median_deg: 4.424132
  x_error_m: 0.075996
  y_error_m: 0.147452
  z_error_m: 0.041714
  position_accuracy: 0.968301
  angle_accuracy: 0.910323
  position_accuracy_0.5m: 0.968301
  position_accuracy_1m: 0.995021
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.807096
  angle_accuracy_15deg: 0.910323
  angle_accuracy_30deg: 0.991744
repro:
  commit: 175045f8a9e7efce4585d658083defd895ad62f6
  branch: feat/tennis-scene-submodules-gvhmr
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=0
    model.num_task_layers=6 data=chunked_multiview_sequence_bs8 data.num_court_kp=14
    data.batch_size=8 training.trainer.accumulate_grad_batches=1 data.seq_len_range=[64,256]
    loss=canonical_rot loss.position_weight=8.0 loss.canonical_pose_weight=0.0 loss.joint_angle_weight=0.0
    loss.torsion_angle_weight=0.0 loss.torso_twist_weight=0.0 loss.bone_length_weight=0.0
    training.trainer.max_epochs=200 training.early_stopping.enabled=false run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i590-courtkp14
  predictions: knowledge/runs/run-i590-courtkp14/pred_test.npz
  log: .training_queue/logs/1783161398610537549_691530_i590_courtkp14.log
  output_dir: outputs/plcs/plcs_multiview_axial_split/logs/version_20
  curves: knowledge/runs/run-i590-courtkp14/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_20
parents:
- run-i545-s6-h0-auxoff-posw8
relations:
- to: run-i545-s6-h0-auxoff-posw8
  rel: compares
- to: run-i545-s6-h0
  rel: compares
- to: run-i560-nocanon-rs-s5-h2
  rel: compares
- to: run-i560-nocanon-rs-s6-h0
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- chunked
- data-rich
- aux-off
- position-weight
- court-kp-14
- sim-to-real
- rotation-best
---

## 考察 / Findings

### 要約
20 点 baseline [[run-i545-s6-h0-auxoff-posw8]] と同一 recipe のまま `data.num_court_kp=14` に揃えた run。位置は **0.166m → 0.189m** と 2.4cm 悪化した一方、回転は **8.46° → 6.28°** に大幅改善し、既存 knowledge 上の回転ベスト [[run-i560-nocanon-rs-s5-h2]]（7.54°）も更新した。14 点化は position の絶対値を少し犠牲にするが、rotation と sim-to-real 整合性の両面で有望。

### アーキテクチャ詳細
`multiview_axial_split` H=0/S=6 fully separate。loss は `canonical_rot` だが `canonical_pose/joint_angle/torsion_angle/torso_twist/bone_length` は全て 0、`position_weight=8.0`。#590 の唯一の変更は `data.num_court_kp=14` で、ネット 6 点を落として実 `court_detection` の出力契約に揃えた。[[run-i545-s6-h0-auxoff-posw8]] との差分は court token 数のみ。

### メトリクスの解釈
test 位置 mean 0.189 / median 0.155、`@0.5m=0.968` で 20 点 baseline の 0.166 / 0.122 よりやや後退。ただし `position_accuracy` 自体は 0.968 でほぼ同等。回転 mean 6.28 / median 4.42、`@15°=0.910` / `@10°=0.807` で、baseline の 8.46 / 6.58、0.851 / 0.694 から大きく改善した。`curves.png` では loss・`pos_error_m`・`ang_error_deg` がともに滑らかに低下し、10k step 前後の小さな揺れ以外は崩壊や過学習の兆候は薄い。終盤まで val angle accuracy が伸び続けており、rotation 改善は偶然のスパイクではない。

### アーキテクチャ⇄メトリクスの因果考察
14 点化でネット/支柱/ストラップ由来の非平面 token が消え、実検出器と一致した planar court token だけを使うため、rotation 学習にはむしろノイズ源が減った可能性がある（仮説）。一方で position は、ネット 6 点が持っていた追加の 3D 幾何拘束を失い、fully separate + posw8 baseline より 2.4cm 悪化したと解釈できる（仮説）。観測としては **「14 点化 = rotation 改善 / position 微悪化」** であり、少なくとも silent mismatch を避けつつ性能崩壊は起こさない。

### 既存実験との比較
- 単一変数比較の基準 [[run-i545-s6-h0-auxoff-posw8]] に対し、位置は 0.189 > 0.166 で悪化、回転は 6.28 < 8.46 で大幅改善。sim-to-real 整合の代償は小さい position 悪化に留まる。
- fully separate の canonical_rot baseline [[run-i545-s6-h0]]（8.96° / 0.186m）より回転良・位置ほぼ同等。14 点でも fully separate の位置優位は維持。
- 既存回転ベスト [[run-i560-nocanon-rs-s5-h2]]（7.54° / 0.332m）より、回転がさらに良く、位置も大幅に良い。回転最適化のために no-canonical + shared trunk へ寄せなくても、court token 契約の整合だけで上回れた。
- [[run-i560-nocanon-rs-s6-h0]]（8.49° / 0.207m）に対しても回転・位置とも上回るため、fully separate 系では **14 点 canonical_rot posw8** が現時点のより良い deploy 候補。

### 次に有効な実験
- 実 `court_detection` 14 点出力を通した end-to-end 評価を行い、合成 test split 上の改善が real inference でも再現するか確認する。
- 位置 2.4cm の戻りを詰めるなら、14 点前提のまま `position_weight` や court-kp dropout / noise augmentation を振り、rotation 6.3°台を維持したまま 0.17m 台へ戻せるかを見る。
