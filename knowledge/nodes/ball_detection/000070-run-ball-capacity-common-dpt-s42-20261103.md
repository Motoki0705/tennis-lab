---
id: run-ball-capacity-common-dpt-s42-20261103
type: run
task: ball_detection
sequence: 70
recorded_at: '2026-11-12'
title: X3D/CSN共通DPT事前学習：10k更新と固定validationの完了を復元確認
issue: 1675
provider: claude
date: '2026-11-03'
status: done
summary_impact: updated
config:
  encoder: x3d_csn
  input_channels: 5
  frames: 32
  decoder_channels: 128
  seed: 42
  planned_updates: 10000
  input_frames_per_update: 128
  microbatch_windows: 1
  activation_checkpointing: false
  frame_chunk_size: null
metrics:
  committed_updates: 10000
  input_frames_committed: 1280000
  update500_loss: 0.001558732549267367
  update500_gradient_norm: 0.009601211175322533
  observed_after_operator_exit_update: 535
  mean_s_per_update_after_first5: 2.3848610763110885
  peak_allocated_gib: 9.801684379577637
  peak_reserved_gib: 12.68359375
  drive_payloads_verified: 7
  validation_mean_error_px: 34.43154376927215
  validation_median_error_px: 8.872786521911621
  validation_p95_error_px: 170.75442199706973
  validation_within_distance_rate: 0.8614065260374622
artifacts:
  run_dir: knowledge/runs/run-ball-capacity-common-dpt-s42-20261103
  output_dir: outputs/ball_detection/train/capacity_scaling/s42-20261103-main/common-dpt-T32
  startup_evidence: knowledge/runs/run-ball-capacity-common-dpt-s42-20261103/startup.json
  completion_evidence: knowledge/runs/run-ball-capacity-common-dpt-s42-20261103/completion.json
parents:
- run-ball-capacity-scaling-20261103
relations: []
papers: []
tags:
- x3d-csn
- dpt
- pretraining
- completed
- persistence
repro:
  commit: e99753cc7f2d
  command: PATH=/tmp/ball-tools:$PATH .venv/bin/python -B knowledge/runs/run-ball-capacity-scaling-20261103/run.py
    train --workspace-root /content/tennis-lab --run-id s42-20261103-main --rclone-config
    /tmp/ball-drive-mount.conf --benchmark /content/tennis-lab/outputs/ball_detection/train/capacity_scaling/benchmark-20261103-a/benchmark.json
---

共通X3D/CSN＋DPTの新規事前学習は10,000更新と固定validationを完了した。2026-11-04の新VM復旧時に、
完了receiptから全10 payloadを検証付きで復元し、checkpoint SHA、学習済み更新数・予算、元のモデルsource、
train/evaluation manifest、checkpoint内のvalidationと保存済み評価JSONの値の一致を照合した。
model_configのtuple/listという表現差だけをJSON正規化して比較した。
[完了・復元の証拠](../../runs/run-ball-capacity-common-dpt-s42-20261103/completion.json)を保存した。

固定validationは128,668 owner frames、位置誤差の分母は85,473 observed frames。
meanは34.4315px、medianは8.8728px、P95は170.7544px、
20px以内率は86.1407%。元画像pixelの値であり、独立testの結果ではない。
このDPTの結果だけでは、後続queryの容量増加による精度改善は判定しない。

前VMの起動時には、500更新のloss0.00155873・gradient norm0.00960121、全500更新の数値有限性、
最初の5更新を除く平均2.3849s/update、peak allocated約9.80GiBを確認した。
これらは最初の500更新だけの訓練・速度観測であり、10k全体の平均や位置誤差の改善を表さない。
起動担当終了後535更新まで継続したという当時の記録を、[元の起動証拠](../../runs/run-ball-capacity-common-dpt-s42-20261103/startup.json)に保持する。

後続query7条件は、この同じ共通10kから独立に各6kへ分岐する。全条件の比較はまだ未完了である。
checkpointing・frame chunk・allocator overrideは無効、BF16・compile・nvJPEG・128 frames/updateと元の予算を維持する。
seed42のみ・validationのみという限界を維持し、現行sceneモデルの採用判断は変更しない。
TensorBoardは使用せず、train.jsonlと終端validationを正本とする。

prepared manifestsとコードbundleはinputs snapshot、checkpointは500更新ごとのDrive世代に保持する。
[実行・再開手順](../../runs/run-ball-capacity-scaling-20261103/README.md)を参照する。
