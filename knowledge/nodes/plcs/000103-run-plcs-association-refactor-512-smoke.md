---
id: run-plcs-association-refactor-512-smoke
type: run
task: plcs
sequence: 103
recorded_at: '2026-09-22'
title: PLCS associationのGlobal MHA＋mHC・D512/12 stage GPUスモーク
provider: codex
session: 01a0c384-7833-7021-868a-22a9433bfc02
date: '2026-09-22'
status: done
config:
  model: plcs_view_association
  hidden_dim: 512
  num_stages: 12
  num_heads: 8
  ffn_dim: 1408
  rope_dim: 32
  num_slots: 4
  max_identities: 10
  attention: global_mha
  mhc: true
  compile: true
  precision: bf16-mixed
  batch_size: 2
  seq_len: 512
  seed: 42
  max_epochs: 1
  train_scenes: 8
  val_scenes: 8
  test_scenes: 8
metrics:
  test/identity_loss: 2.17309832572937
  test/side_loss: 1.784738540649414
  test/identity_accuracy: 0.0764339491724968
  test/side_accuracy: 0.3181818127632141
  test/same_recall: 1.0
  test/opposite_recall: 0.0
  test/fp_precision: 0.0
  test/fp_recall: 0.0
  test/loss: 3.957836866378784
  test/side_balanced_accuracy: 0.5
  elapsed_seconds: 661.0115084179997
  peak_allocated_bytes: 5320590848
  peak_reserved_bytes: 5842665472
repro:
  commit: 8a25c17bf9281f751bc6fd41b6b10af6fb40b23d
  branch: codex/association-task-refactor
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH=. /home/kamimura/projects/tennis-lab/.venv/bin/python
    -u /tmp/association_gpu_smoke.py plcs
artifacts:
  run_dir: knowledge/runs/run-plcs-association-refactor-512-smoke
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790074311564348769_3223468_plcs-association-refactor-512-smoke.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/train/view_association_refactor/gpu_smoke_512_v1
  tb_logdir: outputs/plcs/train/view_association_refactor/gpu_smoke_512_v1/logs/version_0
  curves: knowledge/runs/run-plcs-association-refactor-512-smoke/curves.png
parents: []
relations:
- to: run-blcs-association-refactor-512-smoke
  rel: companion
papers: []
tags:
- association
- smoke
- global-mha
- mhc
- runtime
---

## 確認したこと

再生成済みcamera_view_v2の各split先頭8 sceneを診断process内で選択し、既存PLCS runner経由で1 epoch（4更新）を実行した。Global MHAのみ、時間方向のview queryとmHCを保持するD512/12 stage/8 headモデルで、bf16学習・validation・testが完了した。保存したlast checkpointをCPU predictorで再読込し、カメラローカル2D観測とreferenceから有限なside/ID出力を確認した。3D head/lossやCSWA拡張は使っていない。元データやsplitファイルは書き換えていない。

最大CUDA予約量は5,842,665,472 bytes（約5.44 GiB）、最大allocatedは5,320,590,848 bytesだった。約661秒はcompile、fit、test、保存を含む診断時間で、定常学習速度ではない。後続のCPU checkpoint再読込時間はこのtimerに含まない。D48・2 stage・dropout0の固定CPU入力では、共通stageの配置移動前後の出力・勾配が両taskでbitwise一致した。

## 精度の解釈と次の実験

metricsは8 sceneのsmoke testであり、4更新しか行っていない。side balanced accuracy=0.5、same recall=1、opposite recall=0なので、side分類の有用な収束を示す結果ではない。augmentation無効のvalidation/testではFPの正例を生成しないため、FP precision/recall=0をそれだけでモデル不良の根拠には使えない。

次はこの構成を新規60 epoch、microbatch2・accumulation16（effective32）、再生成V2の800/100/100 splitで学習する。全1000 sceneについて、native120Hzから28/30/60Hzの少なくとも1候補で512出力frameを作れることを確認済み。従来3D position/rotationモデルとは学習目的が異なるので、そのdeployの置換判断には使わない。

診断コマンドの一時scriptの写しを再現bundleのdiagnostic.pyに保存した。再現時は同ファイルを/tmp/association_gpu_smoke.pyへ配置する。TensorBoard曲線は4更新の実行確認用であり、収束比較には使わない。
