---
id: run-i1014-3d-eventhead-regression-eventonly-s42-20261006-v1
type: run
task: ball_refiner_3d
sequence: 11
recorded_at: '2026-10-06'
title: 3D回帰＋イベント確率ヘッドの本学習：L1/soft CE 1:1・GANなし
issue: 1014
provider: codex
session: 01a10697-63da-75b2-ae84-f2980da51c23
date: '2026-10-06'
status: done
config:
  model:
    dimensions: 3
    architecture: regression
    width: 256
    layers: 8
    heads: 4
    dropout: 0.05
    window_length: 128
    flow_steps: 16
    ffn_dim: 704
    ffn_type: swiglu
    rope_dim: 64
    rope_theta: 10000.0
  loss:
    position: normalized L1
    event: soft cross-entropy
    position_weight: 1.0
    event_weight: 1.0
    event_sigma_frames: 2.0
    gan_enabled: false
    loss_schedules_enabled: false
  data:
    dataset: ball_refiner/single_object
    evaluation_event_probability: 0.5
    evaluation_seed: 20991
  corruption:
    event_probability: 0.5
    isolated_probability: 0.0
    gap_min: 3
    gap_max: 10
    noise_p95_px: 0.0
    jitter_sigma_px: 0.0
    outlier_probability: 0.0
    triangulation_steps: 3
  training.steps: 4000
  training.batch_size: 32
  training.learning_rate: 0.0003
  training.weight_decay: 0.01
  run.seed: 42
  dataset_manifest_sha256: 18fe4f79dcc1a030edc41922e61b15085a0de0bd25735c50a4a5e0ec8dd87cca
metrics:
  best_step: 4000.0
  checkpoint_step: 4000.0
  inference_ms_per_frame: 0.019579
  test_event_brier: 0.003244
  test_event_rmse_m: 0.321233
  test_event_soft_ce: 0.086633
  test_frame_missing_rate: 0.113212
  test_missing_rmse_m: 0.427936
  test_rmse_m: 0.157552
  validation_rmse_m: 0.14971549808979034
  test_observed_rmse_m: 0.06791400164365768
  linear_test_rmse_m: 0.30155906081199646
  linear_test_missing_rmse_m: 0.8962436318397522
  training_seconds: 194.27434777002782
  peak_gpu_memory_bytes: 1005275648
repro:
  commit: 09540efcf1cd46e1941eaf31f4062691eeee3c66
  branch: codex/ball-refiner-3d-consolidation
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.tasks.ball_refiner_3d.scripts.train_coordinates
    run.output_dir=ball_refiner_3d/train/rope-regression-eventhead-eventonly/20261006-v1-s42
artifacts:
  run_dir: knowledge/runs/run-i1014-3d-eventhead-regression-eventonly-s42-20261006-v1
  predictions: knowledge/runs/run-i1014-3d-eventhead-regression-eventonly-s42-20261006-v1/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791267710532423200_3920312_i1014-3d-eventhead-regression-eventonly-s42-20261006-v1.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner_3d/train/rope-regression-eventhead-eventonly/20261006-v1-s42/logs/version_0
  checkpoint: /home/kamimura/projects/tennis-lab/outputs/ball_refiner_3d/train/rope-regression-eventhead-eventonly/20261006-v1-s42/logs/version_0/checkpoints/best.ckpt
  checkpoint_sha256: 1a5b1fa7cbe02453e269788fb638f5979a446e363ff18de9f7f627ed3c0ddf8a
  config: knowledge/runs/run-i1014-3d-eventhead-regression-eventonly-s42-20261006-v1/config.yaml
  curves: knowledge/runs/run-i1014-3d-eventhead-regression-eventonly-s42-20261006-v1/curves.png
  scalars: knowledge/runs/run-i1014-3d-eventhead-regression-eventonly-s42-20261006-v1/tensorboard_scalars.json
  tb_logdir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner_3d/train/rope-regression-eventhead-eventonly/20261006-v1-s42/logs/version_0
  browser_checks: knowledge/runs/run-i1014-3d-eventhead-regression-eventonly-s42-20261006-v1/review/browser-results.json
  review_desktop: knowledge/runs/run-i1014-3d-eventhead-regression-eventonly-s42-20261006-v1/review/01-saved-desktop.png
  review_speed: knowledge/runs/run-i1014-3d-eventhead-regression-eventonly-s42-20261006-v1/review/02-event-speed.png
  review_mobile: knowledge/runs/run-i1014-3d-eventhead-regression-eventonly-s42-20261006-v1/review/07-mobile.png
  cpu_gpu_comparison: knowledge/runs/run-i1014-3d-eventhead-regression-eventonly-s42-20261006-v1/review/cpu-gpu-prediction-comparison.json
parents:
- group-i991-i1014-gan-only-eventonly-s42
relations:
- to: run-i1014-rope-3d-gan-eventonly-s42-20261006-v3
  rel: compares
papers: []
tags:
- coordinate-refiner
- 3d
- event-head
- gaussian-target
- regression
- no-gan
- event-only
- seed42
---

## 結果

3D直接回帰＋イベント確率ヘッドの既定構成を、共有training queue経由で4,000更新学習した。train/val/testは1,024/128/128ラリー、testは54,420 frame。既存の共通 `data/ball_refiner/single_object` を再利用し、モデルは3D座標と欠損maskから観測区間も含む全frameを推論する。今回の対象は直接回帰1条件であり、Flowの本学習は実施していない。

幅256・8層・4heads・RoPE dim64・SwiGLU FFN704・窓128frame、batch32、AdamW lr3e-4・weight decay0.01・clip1、seed42。正規化位置L1とイベントsoft CEの係数は1:1で最後まで維持し、GANとloss係数スケジュールは無効。イベント教師はσ=2frameのGaussianをmaxで合成し、イベント2クラスsoftmaxを学習する。完全な解決済み設定は `artifacts.config`、モデル・損失の定義はtask READMEを参照。

入力はイベント選択率50%、左右3〜10frameの異なる幅の連続欠損だけを付加する。jitter・外れ値・P95ノイズ・離散欠損はすべて0。評価seed20991は固定。testで選択したイベントは49.20%、実frame欠損率は11.32%、実測ノイズP95と離散欠損率は0だった。イベント選択率とframe欠損率を区別する。

| 固定testの指標 | 学習済み回帰 | 同じ入力の線形補間 |
|---|---:|---:|
| 全体RMSE (m) | 0.157552 | 0.301559 |
| 欠損RMSE (m) | 0.427936 | 0.896244 |
| 観測RMSE (m) | 0.067914 | 0.000000757 |
| イベント±5frame RMSE (m) | 0.321233 | 0.672984 |

イベント確率のBrier scoreは0.003244、soft CEは0.086633。いずれもGaussian軟教師に対する指標であり、イベント時刻の検出F1や時刻誤差ではない。validation全体RMSEで選択したbestはstep4,000・0.149715m。best/lastは同じ重みでtest結果も同じ。GPU学習の記録時間は194.27秒、PyTorchの最大割当は1,005,275,648bytes。既存の推論やGPTレビューを停止する必要はなかった。

## 解釈・比較の限界

同じ入力の線形補間に対して欠損RMSEは約52%小さい。一方、ノイズのない観測も再推論するため観測位置には約6.8cmのRMSEが生じた。全体の平均距離誤差は回帰0.080664m、線形0.076272mであり、RMSE低減が全frame・全指標の改善を意味するわけではない。

過去のRoPE＋GAN条件（比較先ノード）は全体0.1471m・欠損0.4016mであり、今回が位置で優位という結果ではない。SmoothL1からL1への変更、GAN無効化、イベント補助教師を同時に変更したため、イベントヘッド単独の効果やGANの因果効果は分離できない。最終100更新の平均は位置L1=0.006996、イベントCE=0.087701であり、係数1:1はloss実測値の同等化ではない。

validation位置RMSEは最後の500更新でも0.171174→0.149715mへ改善しており、4,000更新で収束したとは断定しない。単一seed・合成データのみで、実動画の3D品質、長い欠損、追加seedでの再現性は未確認。次は同一入力・L1・GANなしでイベントloss有無を比較し、その後に学習予算やイベント時刻指標を検討する。

## WebUIでの確認

Dataset Reviewの既定候補が本学習のbest.ckpt（step4,000）になり、test全128ラリーの保存済み推論をCPUで再生成した。checkpoint・manifest・評価条件のhashを照合し、GPU評価とCPU表示cacheを全54,420frameで比較した。入力・GT・mask・frame/rally対応は一致、予測座標の最大成分差は1.72e-5m、イベント確率の最大差は2.27e-6だった。

ブラウザでGT/拡張後入力/3D推論、イベント確率とGaussian教師、camera切替、イベント移動、3D回転、比較レイアウト、CPU再推論、JSON/PNG出力と設定復元、遅延応答の排除、390/800/1600px幅の表示を確認した。desktop/mobile・速度曲線・出力PNGを親が撮影して目視確認した。ラリー000030ではイベント確率のピークがGTに近い一方、欠損付近に速度振動が残った（ラリー単体速度RMSE 3.013m/s）。位置・イベントの一致を物理的に滑らかな曲線の達成とはみなさない。

UIの変更・再推論・出力検査には、イベント75%・jitter3px・外れ値10%・P95 300pxの学習外ストレス条件も使用した。この1ラリーでは位置RMSE 1.952mとなり、耐ノイズ性は確認できていない。保存した `exported-comparison.png` はこのストレス条件であり、本学習の固定test結果とは区別する。通常画面の証拠は `01-saved-desktop.png` と `02-event-speed.png`。

学習用ソースはrepro.commitで固定し、今回の本学習後にモデルコードは変更していない。診断JSONの `sampling` 文言は既存の共通Flow表記だが、実際の方式は保存config/checkpointどおり決定論的回帰。原ログは変更せず保持する。`kg_curves.py` は今回の独自scalar名を対象外としてskipしたため、実TensorBoard scalarをJSON保存し、位置/イベントloss・validation位置/Brier・係数を別途描画した。今回の学習ではvalidatorを追加起動していない。
