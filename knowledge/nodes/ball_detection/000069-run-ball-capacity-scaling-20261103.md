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
status: done
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
  gpu_configurations_succeeded: 8
  gpu_cases_succeeded: 9
  estimated_training_compute_hours: 26.34450901861025
  query_median_s_per_update_min: 1.673811849499998
  query_median_s_per_update_max: 1.7503488290003588
  query_max_allocated_gib: 14.144630432128906
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
repro:
  commit: 13dff57d41f36f0c5570a76bc6edb2808650f8df
  command: PATH=/tmp/ball-tools:$PATH .venv/bin/python -B knowledge/runs/run-ball-capacity-scaling-20261103/run.py
    benchmark --workspace-root /content/tennis-lab --run-id benchmark-20261103-a --rclone-config
    /tmp/ball-drive-mount.conf
---

全データの取得・CPU検証に続き、L4で720p・5ch・BF16・compile・nvJPEG・128 frames/updateを実測した。
共通DPTとquery7条件の全8構成がforward/backwardに成功し、DPT BS1/2の比較を含む9 case JSONと最終summaryをDriveから読み戻してbyte照合した。
本ノードは事前実測の完了を表し、10k/6kの精度実験の完了ではない。診断の重みを本学習へ転用しない。

| 構成 | 定常s/update | peak allocated GiB | peak reserved GiB |
|---|---:|---:|---:|
| DPT BS1 | 2.341 | 9.80 | 13.56 |
| DPT BS2 | 2.647 | 17.89 | 20.42 |
| query 256×4 | 1.674 | 12.92 | 20.91 |
| query 384×4 | 1.674 | 13.04 | 21.43 |
| query 512×4 | 1.685 | 13.19 | 21.19 |
| query 768×4 | 1.707 | 13.61 | 20.57 |
| query 1024×4 | 1.750 | 14.14 | 21.10 |
| query 512×6 | 1.696 | 13.41 | 21.61 |
| query 512×8 | 1.719 | 13.60 | 20.46 |

CNN理論値は3.30、query全体は3.47〜5.53 GMAC/frame。幅や層数の増加による実測時間の増加は小さいが、
この表だけでCNNの実時間比率や精度改善を判断しない。精度・収束は未評価である。
DPTはBS1が速いため採用する。実効入力は全条件128 frames/updateのまま。
全52k学習更新はjoint定常値換算で約26.34時間。凍結区間を別モデル化せず、compile・データ検証・validation・Drive保存・診断費用を含まない。

メモリ制御は全caseで無効のまま。標準allocatorの回復可能なallocation failure warningは7回あったが、
例外によるcase失敗はなく、数値はfiniteで全更新を通過した。reserved peakとcontextを含む実使用量はallocated peakと区別する。

最初のoperator終了時に、同じagent supervisor配下で起動したqueue workerの子がzombieとして残り、終了確認が待ち状態になった。
operator supervisorの回収後、元のqueue jobはdoneとなり、GPU processも0であることを親が確認した。学習や測定の重複起動はしていない。
本学習では空の共有queue serviceを親側の独立したreaperから起動し、Codex子が登録・最初のDrive保存までを担当する。
この監督処理はモデル・学習・メモリ設定を変えない。

[実測summary](../../runs/run-ball-capacity-scaling-20261103/benchmark.json)、
[理論計算量](../../runs/run-ball-capacity-scaling-20261103/theory.json)、
[検証](../../runs/run-ball-capacity-scaling-20261103/validation.json)、
[実行・再開手順](../../runs/run-ball-capacity-scaling-20261103/README.md)を保存した。
データは全2,925 payloadの独立2系統SHA-256を照合済み、学習は30,478 observed-containing pack。
CPU219件と最終修正後35件（重複する再検証）が通り、halo11の勾配と実runnerのmodel・AdamW・RNG・抽出位置の厳密再開を確認した。

次は新規共通DPT10kからquery7条件各6kを独立に学習する。seed42のみ・validationのみであり、
条件差がseedのばらつきを超えるかは判断しない。既存モデルの採用判断は変更しない。
TensorBoardは使用せず、更新JSONLを正本とする。

2026-11-04の新VM復旧では、元controllerのJSON配列とdataclass tupleの比較が同じ計画を拒否したため、
元の実装・plan・benchmarkを保持したまま、controller内のメタデータ出力だけをJSON正規化する追加ランチャーを検証した。
CPU10件で再開境界の通過と実質変更の拒否を確認し、追加コードを別のDrive snapshotへ保全した。
1,500→2,000更新の全500更新はfiniteで、周期checkpointの全7 payloadを独立復元して照合し、
起動担当の実終了後も継続することを親が確認した。[再開証拠](../../runs/run-ball-capacity-scaling-20261103/resume-attempt-3-startup.json)と
更新した再開手順を参照する。これは起動安定の確認であり、7条件の容量比較の完了・仮説の支持を意味しない。
