---
id: run-blcs-axial-base-t128-v3-6-reprojection-e200-s42-20261006
type: run
task: blcs
sequence: 43
recorded_at: '2026-10-06'
title: 'BLCS axial base: physical KP14・128 frame・3–6 view・200 epochと正式採用'
provider: codex
session: 01a10f9e-7f68-7623-9325-0a5534377076
date: '2026-10-06'
status: done
config:
  model: multiview_axial_base
  loss: reprojection
  data: multiview_sequence
  court_keypoints: physical_v1
  court_kp: 14
  sequence_length: 128
  train_views:
  - 3
  - 6
  seed: 42
  max_epochs: 200
  batch_size: 2
  gradient_accumulation: 8
  effective_batch_size: 16
  validation_every_epochs: 5
  learning_rate: 0.0001
  weight_decay: 0.01
  position_weight: 1.0
  reprojection_weight: 0.1
  predict_velocity: false
  gan: false
  compile: false
  precision: bf16-mixed
metrics:
  best_val_position_error_m: 0.3374446630477905
  test_position_error_m: 0.3824673891067505
  test_accuracy_0p3m: 0.5979779362678528
  test_endpoint_error_m: 1.074790596961975
  full_scene_test_position_error_m: 0.49481457707393484
  full_scene_test_p95_error_m: 1.7049788734541198
repro:
  commit: 1178e66a8873d57083366bfc4b9536eee494878a
  branch: HEAD
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env MPLBACKEND=Agg OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python -u -m src.tasks.blcs.scripts.train model=multiview_axial_base data=multiview_sequence training=default loss=reprojection paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs paths.artifact_root=/home/kamimura/projects/tennis-lab/outputs paths.checkpoint_root=/home/kamimura/projects/tennis-lab/ckpt model.predict_velocity=false model.num_court_tokens=14 data.num_court_kp=14 'data.seq_len_range=[128,128]' 'data.num_views_range=[3,6]' data.camera_mode=random data.batch_size=2 data.num_workers=4 data.augmentation.enabled=true loss.position_weight=1.0 loss.reprojection_weight=0.1 loss.smoothness_weight=0.0 loss.gravity_weight=0.0 training.learning_rate=1e-4 training.weight_decay=0.01 training.warmup_steps=200 training.min_lr=1e-6 training.compile.enabled=false training.gan.enabled=false training.trainer.precision=bf16-mixed training.trainer.max_epochs=200 training.trainer.accumulate_grad_batches=8 training.trainer.check_val_every_n_epoch=5 training.trainer.enable_progress_bar=false training.trainer.log_every_n_steps=10 training.checkpoint.monitor=val/position_error_m training.checkpoint.mode=min training.checkpoint.save_top_k=1 training.checkpoint.save_last=true training.qualitative_logging.every_n_epochs=10 run.seed=42 run.resume=null run.init_weights=null run.test_after_fit=false run.output_dir=blcs/train/axial_base_t128_v3-6_reprojection_e200/s42-20261006-001
artifacts:
  run_dir: knowledge/runs/run-blcs-axial-base-t128-v3-6-reprojection-e200-s42-20261006
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791272345453038964_1013694_blcs-axial-base-t128-v3-6-reprojection-e200-s42-20261006.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/blcs/train/axial_base_t128_v3-6_reprojection_e200/s42-20261006-001/logs/version_0
  checkpoint: blcs/axial-base-kp14-t128-v3-6-e200-s42-epoch189.ckpt
  checkpoint_sha256: c83e56a51c553a10203b509cd307df930ccba59adb72c49cec89d6f42e1d73a8
  predictions: knowledge/runs/run-blcs-axial-base-t128-v3-6-reprojection-e200-s42-20261006/pred_test.npz
  evaluation: knowledge/runs/run-blcs-axial-base-t128-v3-6-reprojection-e200-s42-20261006/evaluation.json
  checkpoint_selection: knowledge/runs/run-blcs-axial-base-t128-v3-6-reprojection-e200-s42-20261006/checkpoint-selection.json
  curves: knowledge/runs/run-blcs-axial-base-t128-v3-6-reprojection-e200-s42-20261006/curves.png
  tb_logdir: outputs/blcs/train/axial_base_t128_v3-6_reprojection_e200/s42-20261006-001/logs/version_0
  full_scene_test: knowledge/runs/run-blcs-axial-base-t128-v3-6-reprojection-e200-s42-20261006/full-scene-test.json
parents: []
relations: []
papers: []
tags:
- axial
- physical-v1
- kp14
- single-object
- validation-selected
- adopted-by-user
---

## 考察 / Findings

### 結果と選定
固定のphysical_v1合成データで200 epoch／8,400 optimizer更新を完了した。validationは5 epochごとに40回実行し、`val/position_error_m`最小のepoch189（0始まり、190 epoch目）を選定した。位置誤差は0.337445 mで、最終epoch199の0.348252 mより良い。選定にはtestを使っていない。

採用重みを明示的に復元したtestは85 scene・128 frameの中心window・seed42の決定的な3–6 camera sampling・augmentation無効で実施した。平均位置誤差0.382467 m、0.3 m以内率0.597978、終端誤差1.074791 m。予測・教師・mask・scene IDを保存し、再集計可能にした。

別protocolの全長診断では、短いsceneも含むtest100 scene・32,501 frameをcamera0–3固定、重複しない128 frame窓、float32 predictorで推論した。frame重み付き平均0.494815 m、中央値0.219434 m、p95 1.704979 m、0.3 m以内率0.636042、scene終端平均2.665169 m。中心window/bf16の85 scene評価とはカメラ・長さ・母集団・precisionが異なり直接順位付けしない。全長診断を使ってcheckpointを選び直していない。

### 学習条件と評価の範囲
baseは幅512、8 stage、MHA/SwiGLU、CourtKP14、位置のみ出力する。位置Smooth L1 weight1.0と再投影weight0.1を使い、GAN・smoothness/gravity priorは無効。AdamW、LR1e-4、weight decay0.01、warmup200更新、cosine最低LR1e-6、bf16-mixed、compile無効、実batch2・蓄積8・実効batch16、seed42で新規初期化した。元splitは800/100/100だが、128 frame未満を除きtrain669・val80・test85 sceneとなる。固定split内のcrop/view抽出であり、全長評価・実動画の独立3D評価ではない。

訓練中の自動testは無効にし、学習終了時のlastを採用重みのtest値と混同しない。後続の選定checkpoint評価は学習時commit1178e66aから実行した。評価初回はLightningのweights-only既定でOmegaConfを復元できず停止し、信頼できる自己学習checkpointに対する`weights_only=False`を明示して再実行した。重み・損失・評価入力は変更していない。

### 正式採用と配置
2026-10-06のユーザー指定により、この選定モデルをtennis_sceneの標準ボール再構成へ採用する。`ckpt/blcs/axial-base-kp14-t128-v3-6-e200-s42-epoch189.ckpt`へ移動し、移動前後の独立2実装SHA-256一致を確認した。digest・元パス・選定根拠は`checkpoint-selection.json`が正本。モデル採用はユーザー方針であり、合成testの数字を実写精度の合格証明とはしない。旧reference/broadcast/residualモデルとは入力・座標・データが異なり、過去値との直接ランキングは行わない。

採用した実checkpointを用いたGPU integration testでも、保存済みphysicalシーンをcamera-local CourtKP14へ変換して新componentに渡し、162 frame全体の有限出力と観測2view以上のvalid/inlier契約を確認した。合成観測での接続検証であり、RGB検出からの実動画E2Eは今回実行していない。

### 残課題
終端誤差が平均より大きく、窓端の挙動・欠測・バウンドを実観測で検品する必要がある。複数seed／別会場の検証は未実施。自動GIFはvalidation5・保存間隔10と0始まりepoch剰余判定が交差せず0件だった。全タスク監査をissue #1031へまとめ、今回の採用PRではGIF修正を行わない。
