---
id: run-i576-camtoken-s0-h12
type: run
title: i576_camtoken_s0_h12
issue: 576
provider: claude
session: 83686dbd-1af9-4c61-814d-63204dfb6684
date: '2026-06-28'
status: done
config:
  model: multiview_axial_camtoken
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.563791
  position_error_std_m: 0.375127
  position_error_median_m: 0.470525
  angular_error_deg: 10.178311
  angular_error_std_deg: 9.925077
  angular_error_median_deg: 7.553486
  x_error_m: 0.250545
  y_error_m: 0.442548
  z_error_m: 0.058563
  position_accuracy: 0.541948
  angle_accuracy: 0.796667
  position_accuracy_0.5m: 0.541948
  position_accuracy_1m: 0.881729
  position_accuracy_2m: 0.99196
  angle_accuracy_10deg: 0.620485
  angle_accuracy_15deg: 0.796667
  angle_accuracy_30deg: 0.950827
repro:
  commit: 8ed46945dae97f9307b38562cbbfd79172388929
  branch: exp/issue-576-camera-token-split
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_camtoken data.batch_size=8
    training.trainer.accumulate_grad_batches=1 data=chunked_multiview_sequence_bs8
    data.seq_len_range=[64,256] loss=canonical_rot training.trainer.max_epochs=200
    training.early_stopping.enabled=false training.trainer.check_val_every_n_epoch=10
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i576-camtoken-s0-h12
  predictions: knowledge/runs/run-i576-camtoken-s0-h12/pred_test.npz
  log: .training_queue/logs/1782529475051150394_2429647_i576_camtoken_s0_h12.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/plcs_multiview_axial_camtoken/logs/version_0
  curves: knowledge/runs/run-i576-camtoken-s0-h12/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_camtoken/logs/version_0
parents:
- run-i545-s0-h12
relations:
- to: run-i560-nocanon-s0-h12
  rel: compares
- to: run-i545-s6-h0
  rel: compares
- to: run-i560-nocanon-rs-s6-h0
  rel: compares
tags:
- plcs
- shared-trunk
- readout-split
- camtoken
- chunked
---

## 考察 / Findings

### 要約
共有 trunk のまま head ごとに別 camera トークンを読む readout 分離（pose←cam0 / rot←cam1）は、**同一 loss（canonical_rot）・同一容量の fully-shared baseline（[[run-i545-s0-h12]]）に対して位置 0.664→0.564m（-0.10m, ≈15%）・回転 10.85→10.18° と小幅改善**したが、separate-trunk の位置 ~0.19m には全く届かない（0.564m で頭打ち）。安価な readout 分離だけでは位置⇔回転競合は緩和しきれず、**位置には trunk 分離が依然必須**という #545/#560 の結論を覆さない。

### アーキテクチャ詳細
`multiview_axial_camtoken`（`PLCSMultiViewAxialModel` を継承し `forward` のみ override）。trunk は 1 本の fully-shared（S=0/H=12, hidden=512, layers=12, heads=8）で baseline と同一。axial attention 後に各 camera トークンが全 camera/time を参照済みである性質を使い、`position_head` 入力 = `x[:,:,0,:]`（cam0）、`rotation_head`（および canonical 系）入力 = `x[:,:,1,:]`（cam1）に分離。target は world 座標 per-time 量で camera 非依存なのでどちらのトークンから読んでも整合する。loss は `canonical_rot`、200ep / early-stop OFF / chunked_multiview_sequence_bs8 / seq_len [64,256]。

### メトリクスの解釈
位置 mean 0.564m / median 0.471m、回転 mean 10.18° / median 7.55°。位置@0.5m=0.542・角@15°=0.797。回転は崩壊せず実用域だが、位置は @0.5m が 0.54 と低く（separate-trunk は 0.95+）world 位置の精度が出ていない。z 0.059m に対し x 0.251m / y 0.443m と水平面の誤差が支配的。

### アーキテクチャ⇄メトリクスの因果考察
readout を別トークンに割っても position/rotation の勾配は**共有 trunk 内部で混ざる**ため、競合の本体（trunk 表現の奪い合い）は解けない。改善が +0.10m / +0.7° と小幅に留まったのはそのため、と解釈する（仮説）。cam1 トークンも axial 後はグローバル要約なので回転は cam0 同等に読めて崩壊は防げるが、位置の容量配分は baseline と変わらず ~0.56m に張り付いた。

### 既存実験との比較
- **同 loss fully-shared baseline [[run-i545-s0-h12]]**（cam0 を両 head が読む, canonical_rot）: 0.664m / 10.85°。→ readout 分離で **位置 -0.10m・回転 -0.67°** と両方わずかに改善。分離の効果は正だが小さい。
- **issue が引用した baseline [[run-i560-nocanon-s0-h12]]**（no_canonical, fully-shared）: 0.283m / 50.34°。→ 位置はむしろ悪い（0.564 vs 0.283）が回転は大幅良化。ただしこれは **loss 差（no_canonical は位置良・回転崩壊）** であって readout 分離の効果ではない。apples-to-apples の baseline は同 loss の [[run-i545-s0-h12]]。
- **separate-trunk [[run-i545-s6-h0]]**（canonical_rot, S=6/H=0）8.96° / 0.186m、**[[run-i560-nocanon-rs-s6-h0]]** 8.49° / 0.207m。→ camtoken は回転は同等域だが**位置が約 3 倍劣る**。readout 分離は separate-trunk の代替にならない。

### 次に有効な実験
- readout 分離を **separate-trunk または H>0 の浅い共有（S=5/H=2 等）に重ねて**、位置を保ったまま回転 readout の競合をさらに減らせるか。共有 trunk 単体では位置が頭打ちのため、分離 trunk との併用が筋。
- camtoken を維持するなら position_weight 増量（#545 posw 知見）で 0.56m→0.4m 級に押せるか、安価枠での position 改善余地を確認。
- ただし費用対効果は限定的（separate-trunk が位置で圧勝）なので、本系統は「安価な部分緩和」の記録に留め、主戦は separate-trunk + rot-strong（[[run-i560-nocanon-rs-s5-h2]] 系）を推奨。
