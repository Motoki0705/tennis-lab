---
id: run-ball-sequential-length-20261025
type: run
task: ball_detection
sequence: 68
recorded_at: '2026-10-25'
title: 共通DPTからの逐次座標回帰・T64学習と事前学習時間の検証
issue: 1514
provider: codex
date: '2026-10-25'
status: running
config:
  input_channels: 5
  resolution:
  - 720
  - 1280
  seed: 42
  input_frames_per_update: 128
  common_pretraining_updates: 10000
  dpt_posttraining:
    T64: 6000
  query_posttraining:
    T32: 6000
    T64: 6000
  skipped_T:
  - 128
  compile_mode: default
  activation_checkpointing: false
  frame_chunk_size: null
  warmup_updates: 500
  posttraining_cnn_freeze_updates: 600
  encoder_lr_ratio: 0.1
  optimizer_state: inherit matching parameters
metrics:
  download_bytes: 157028755547
  download_files: 2925
  download_seconds: 2068.5761816501617
  accepted_training_packs: 30478
  excluded_position_free_packs: 2031
  unique_observed_training_frames: 491380
  unique_negative_training_frames: 95674
  edge_only_training_packs: 32
  benchmark_updates_each_batch: 30
  benchmark_bs1_median_update_seconds: 2.038441351000074
  benchmark_bs2_median_update_seconds: 2.047568384999977
  benchmark_bs1_pretraining_compute_hours: 5.662337086111317
artifacts:
  run_dir: knowledge/runs/run-ball-sequential-length-20261025
  output_dir: outputs/ball_detection/train/sequential_length
  data_preparation: outputs/ball_detection/precompute/sequential_length/observed-packs-20261025
parents:
- run-ball-colab-checkpoint-resume-20261024
- run-ball-temporal-heatmap-gpu-diagnostic-20261023
relations: []
papers: []
tags: []
repro:
  commit: cda340b2b210
  command: PATH=/tmp/ball-tools:$PATH .venv/bin/python knowledge/runs/run-ball-sequential-length-20261025/run.py
    benchmark --workspace-root /content/tennis-lab --run-id benchmark-20261025-a --rclone-config
    /tmp/ball-drive-mount.conf
---

ユーザーが#1514の実装と本学習を依頼し、最初にDriveから学習データを取得した。
新しい共通事前学習はCNN＋DPTのT32で10,000更新とし、過去の2ch/5ch checkpointを流用しない。
LRとAdamW状態の逐次継承は会話で確認済み。T128はL4の分割なし試走でOOMを観測した後、
ユーザーが今回の実行から明示的に除外した。残る物理実行は共通10kと事後3段×6kの計28k更新。
DPTの来歴は16k、queryの来歴は22kで、共通事前学習を実行費用として二重計上しない。

Driveのtrain/validation配布版を全件取得し、全ファイルのサイズと独立した2系統のSHA-256を照合した。
入力は1,537 clipの既存の不変配布版で、元データ・注釈・splitを変更していない。
同じ128枚組をT32/T64に分割するmanifestを再生成し、32,509→30,478組を確認した。
位置教師のない子窓はT32で10,383出現、T64で2,780出現あり、座標lossのゼロ分子と更新全体の分母で扱う。
全組の先頭/末尾だけに位置教師がある32組では、遮蔽のrequested/eligible/appliedを記録して教師を保持する。
validationの共通集合は128,668 frame、observedは85,473 frameを保持する。
別版test配布の完成・評価は今回の学習完了を根拠に推定しない。

可変Tの直接座標の保存形式、DPTからCNNのみの転送、同構成の全重み転送、
名前に基づくAdamW moment/step継承と段階ごとのCNN凍結/解除を実装した。
CPUの実モデル試験ではDPT T32→T64→T128とquery T32→T64→T128、queryの中断再開を検証した。
T128のCPU対応は実装の検証であり、今回のGPU本学習へT128を追加したものではない。
直接座標評価はheatmapと同じowner/observed集合を採点し、存在head・FPR/FNRを新設しない。

生の合成RGBによる720p・BF16・compile有効・checkpointing無効の予備診断では、
DPT T32/BS1は初回130.61秒、定常約0.48秒/32枚、peak allocated約7.77 GB、
T64/BS1は初回112.56秒、定常約0.98秒/64枚、peak allocated約15.43 GBだった。
T128/BS1とT32/BS4はOOM。T128/chunk4の追加診断はユーザーのT128除外指示で中止し、
成立・不成立や速度を推定していない。これらのraw計算値を、拡張・復号・保存を含む事前学習時間と混同しない。

実データ30更新のBS1/BS2比較が完了し、更新中央値はそれぞれ2.0384/2.0476秒だった。
速度差は小さく、メモリ使用量の少ないBS1を選ぶ。10k更新の計算部分は約5.66時間と見積もる。
現在は全validationの時間計測を実行中で、保存・入力検証・validation込みの時間は未確定。
本学習はまだ開始しておらず、所要時間・収束・精度の改善を確定していない。
終端checkpointで比較し、testでcheckpointを選ばない。
TensorBoardはこの更新数ベースのrunnerでは出しておらず、監査は更新ごとのtrain.jsonlとcheckpointを使う。
Driveの既存mountをrclone aliasで明示接続し、既存RcloneSnapshotStoreで世代別保存と復元を確認した。
小さな保存世代はGoogle Drive connectorでも実体を確認した。Driveの入力や過去世代は削除していない。
