---
id: run-blcs-association-refactor-512-smoke
type: run
task: blcs
sequence: 34
recorded_at: '2026-09-22'
title: BLCS associationのGlobal MHA＋mHC・D512/12 stage GPUスモーク
provider: codex
session: 01a0c384-7833-7021-868a-22a9433bfc02
date: '2026-09-22'
status: done
config:
  model: blcs_view_association
  hidden_dim: 512
  num_stages: 12
  num_heads: 8
  ffn_dim: 1408
  rope_dim: 64
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
  test/identity_loss: 2.003100872039795
  test/side_loss: 1.0080628395080566
  test/identity_accuracy: 0.2974843680858612
  test/side_accuracy: 0.695652186870575
  test/same_recall: 0.0
  test/opposite_recall: 1.0
  test/fp_precision: 0.0
  test/fp_recall: 0.0
  test/loss: 3.0111637115478516
  test/side_balanced_accuracy: 0.5
  elapsed_seconds: 635.329637597999
  peak_allocated_bytes: 5312259584
  peak_reserved_bytes: 5853151232
repro:
  commit: 28885e46cb76341001c8d20aaaeb5dcfe987d7a4
  branch: codex/association-task-refactor
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH=. /home/kamimura/projects/tennis-lab/.venv/bin/python
    -u /tmp/association_gpu_smoke.py blcs
artifacts:
  run_dir: knowledge/runs/run-blcs-association-refactor-512-smoke
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790074311418599580_3223446_blcs-association-refactor-512-smoke.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/blcs/train/view_association_refactor/gpu_smoke_512_v1
  tb_logdir: outputs/blcs/train/view_association_refactor/gpu_smoke_512_v1/logs/version_0
  curves: knowledge/runs/run-blcs-association-refactor-512-smoke/curves.png
parents: []
relations: []
papers: []
tags:
- association
- smoke
- global-mha
- mhc
- runtime
---

## 確認したこと

再生成済みcamera_view_v2の各split先頭8 sceneを診断process内で選択し、既存BLCS runner経由で1 epoch（4更新）を実行した。Global MHAのみ、時間方向のview queryとmHCを保持するD512/12 stage/8 headモデルで、bf16学習・validation・testが完了し、保存したlast checkpointをCPU predictorで再読込して6入力から有限なside/ID出力を確認した。3D head/lossやCSWA拡張は使っていない。元データやsplitファイルは書き換えていない。

最大CUDA予約量は5,853,151,232 bytes（約5.45 GiB）、最大allocatedは5,312,259,584 bytesだった。約635秒はcold compile、fit、test、保存を含む診断時間で、定常学習速度ではない。後続のCPU checkpoint再読込時間はこのtimerに含まない。検証時に汎用stageとCourt encoderの配置を追加整理したが、D48・2 stage・dropout0の固定CPU入力で旧PR2モデルとの出力・勾配のbitwise一致を両taskで確認した。

## 精度の解釈と次の実験

metricsは8 sceneのsmoke testであり、4更新しか行っていない。side balanced accuracy=0.5、same recall=0、opposite recall=1なので、有用なside分類の成立や収束を主張しない。旧100 epoch recipeやsingle-ball deployとはモデル・データ・学習予算が異なり、直接の精度比較には使えない。

次はこの512/12構成を新規60 epoch、microbatch2・accumulation16（effective32）、再生成V2の800/100/100 splitで学習し、同側/反対側recall、identity/FP指標、有限loss、学習速度とメモリを確認する。既存の3D推定baselineやdeploy checkpointの採否は変更しない。

診断コマンドが参照した一時scriptの写しを再現bundleのdiagnostic.pyに保存した。再現時は同ファイルを/tmp/association_gpu_smoke.pyへ配置する。TensorBoard曲線は1 epochの実行確認用であり学習曲線の収束比較には使わない。
