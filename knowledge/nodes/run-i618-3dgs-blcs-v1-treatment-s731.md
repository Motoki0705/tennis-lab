---
id: run-i618-3dgs-blcs-v1-treatment-s731
type: run
title: 3DGS×BLCS 1/6 synthetic treatment seed 731
issue: 618
provider: codex
session: 019f984c-8bc1-7041-8e1d-362a5b11daa2
date: '2026-07-26'
status: done
config:
  model: conv_next_unet
  initialization: ckpt/ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt
  initialization_sha256: cd7927ad27e53ddd6aa77df28eca3c5e674552461ccda083a41e99e629857892
  loss: focal_bce_gamma_2
  data: TrackNet games 1-8 real 5 plus C10 synthetic 1 per batch, game 9 validation
  batch_size: 6
  synthetic_per_batch: 1
  synthetic_batch_period: 1
  steps_per_epoch: 655
  epochs: 8
  seed: 731
  learning_rate: 1.0e-05
metrics:
  best_validation_epoch: 2
  best_val_f1: 0.6631274223327637
  best_val_precision: 0.6368973851203918
  best_val_recall: 0.6916107535362244
  best_val_mean_distance_px: 2.306781768798828
  best_val_loss: 0.0004237828543409705
  final_val_f1: 0.6310160160064697
  final_val_precision: 0.5938425064086914
  final_val_recall: 0.673154354095459
  final_val_mean_distance_px: 2.359797239303589
  paired_control_best_val_f1: 0.6729001402854919
  paired_f1_delta: -0.009772717952728271
  qualification_passed: false
repro:
  commit: ac9e640903a6dfaecb65fc980f5dcf408bbcd589
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: .venv/bin/python -m src.tasks.ball_detection.scripts.train --config-name
    train_3dgs_blcs_v1
artifacts:
  run_dir: knowledge/runs/run-i618-3dgs-blcs-v1-treatment-s731
  log: .training_queue/logs/1784996113586048429_2936455_i618_3dgs_blcs_v1_treatment_s731.log
  output_dir: outputs/ball_detection/3dgs_blcs_paired_v1/treatment/seed_731/logs/version_0
  checkpoint: outputs/ball_detection/3dgs_blcs_paired_v1/treatment/seed_731/logs/version_0/checkpoints/3dgs-blcs-epoch=02.ckpt
  checkpoint_sha256: bcf712e22d7b459294f4a1885d99cbb764c5c253d59f06ca190b28b9ca67f23f
  live_monitor: .codex-loop/C11_LIVE_MONITOR.json
  curves: knowledge/runs/run-i618-3dgs-blcs-v1-treatment-s731/curves.png
  tb_logdir: outputs/ball_detection/3dgs_blcs_paired_v1/treatment/seed_731/logs/version_0
parents:
- run-i618-3dgs-blcs-v1-control-s731
- run-i618-blcs-b00-full-scale-v1
relations:
- to: run-i618-3dgs-blcs-real-baseline-v1
  rel: compares
tags:
- ball_detection
- 3dgs-blcs
- paired-treatment
- simple-sphere
- synthetic-one-sixth
- seed-731
- failed-qualification
---

## 考察 / Findings

### 要約

C10 simple-sphere syntheticを各batchの1/6に固定したseed 731 treatmentは、
game-9 best validation F1 **0.663127**で、paired real-only controlの
**0.672900**を`-0.009773`下回った。事前宣言したqualification gateを満たさず、
同じ1/6 mixを追加seedへ展開しない。

### アーキテクチャ詳細

controlと同じConvNeXt U-Net初期値、AdamW、batch 6、655 step/epoch、
8 epoch、lr `1e-5`、augmentation、seedを使用した。唯一の機能差は、
各batchでreal windowを6から5へ減らし、fingerprinted C10 synthetic
windowを1つ加えた点である。game 9だけでcheckpointを選択し、
`run.test_after_fit=false`によりgame 10は実行していない。

### メトリクスの解釈

epoch 0--7 F1は0.627089、0.623130、0.663127、0.638197、0.631412、
0.612238、0.624030、0.631016。best epoch 2はprecision 0.636897、
recall 0.691611、平均距離2.306782 px、loss 0.000423783だった。
control bestとの差はprecision `-0.000640`、recall `-0.020805`、
距離`+0.023618 px`で、主な退行はrecallである。

### アーキテクチャ⇄メトリクスの因果考察

観測上、localizationはほぼ維持した一方、期待したreal recall改善が出ず、
real windowを毎step 1/6置換した学習量の希釈がsimple-sphereの利益を上回った。
これは因果確定ではなく仮説だが、C10は見かけ径median 1.84 pxでfully
occluded/out-of-frame negativeも多く、flat green appearanceとのdomain gapを
含むため、毎batch投入は強すぎた可能性がある。

### 既存実験との比較

親controlと初期値・実データ・予算・seedは同一で、best F1は
0.672900から0.663127へ低下した。配備モデルを固定manifestで測ったC02値とは
評価集約が異なるため直接改善を主張しない。C10 datasetの幾何・label gateが
passしたことと、real detector F1改善は別であることを負の結果として保持する。

### 次に有効な実験

rendererや球appearanceを同時変更せず、synthetic batchを1つおきに限定して
平均露出を約1/12へ半減する。real window置換を減らした同一seed paired A/Bで、
game-9 F1が同じ更新後コードで再実行したpaired controlを超えるかを一つの
仮説として検証する。
