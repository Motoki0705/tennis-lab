---
id: run-ball-capacity-scaling-20261103
type: run
task: ball_detection
sequence: 69
recorded_at: '2026-11-03'
title: X3D/CSN・4 query mHC容量比較の実装検証とL4事前実測
issue: 1675
provider: codex
date: '2026-11-03'
status: planned
config:
  input_channels: 5
  resolution:
  - 720
  - 1280
  seed: 42
  encoder: x3d_csn
  residual_blocks:
  - 0
  - 1
  - 2
  - 3
  - 3
  - 2
  streams: 4
  head_width: 64
  activation_checkpointing: false
  frame_chunk_size: null
  pretrain_updates: 10000
  query_updates_each: 6000
  conditions:
  - - 256
    - 4
  - - 384
    - 4
  - - 512
    - 4
  - - 768
    - 4
  - - 1024
    - 4
  - - 512
    - 6
  - - 512
    - 8
metrics:
  download_verified_files: 2925
  download_verified_bytes: 157028755547
  cpu_suite_passed: 219
  post_review_passed: 35
  cnn_gmac_per_frame: 3.29964544
artifacts:
  run_dir: knowledge/runs/run-ball-capacity-scaling-20261103
  output_dir: outputs/ball_detection/train/capacity_scaling
parents:
- run-ball-sequential-length-20261025
relations: []
papers: []
tags:
- x3d-csn
- mhc
- capacity
- preflight
---

データ取得を最初に行い、配布版の全2,925 payload・約157GBをサイズと独立2系統のSHA-256で照合した。
位置教師を含まない組を除外した学習集合は30,478 pack。T32/T64の教師・owner集合は一致した。

X3D/CSN encoderと4 query mHCを実装し、CPU219件、最終確認で補った設定拒否・評価model_kindの35件を通過した。
35件は先のsuiteと重複する再検証で、254件の独立テストとは数えない。
halo11での出力・入力/parameter勾配、元のTを検証したstrict転送、実runnerのmodel・AdamW・RNG・pack位置の厳密再開を確認した。

GPU実測と本学習は未完了。このノードの精度改善や収束の結論はまだない。
[固定構成・実行・保存再開手順](../../runs/run-ball-capacity-scaling-20261103/README.md)、
[CPU検証](../../runs/run-ball-capacity-scaling-20261103/validation.json)、
[理論計算量](../../runs/run-ball-capacity-scaling-20261103/theory.json)を保存した。
理論値の計算では新CNNは3.30、全query構成は3.47〜5.53 GMAC/frameでIssue見積もりと一致した。
実測時間やVRAM、精度を理論MACから推定した結果として扱わない。

次にL4・720p・BF16・compileで8構成のforward/backwardを確認し、peak VRAMと定常s/updateを測る。
メモリ制御は無効で、必要になれば有効化前にユーザーへ相談する。全構成が通れば、新規の共通DPT10kから7条件各6kへ分岐する。
seed42のみ・validation比較のみで、条件差がseedのばらつきを超えるかは判断できない。既存モデルの採用判断は変更しない。
TensorBoardはこの専用runnerでは使用せず、更新ごとのJSONLを正本とする。
