---
id: run-pr937-player-inference-real-frame-20260928
type: run
task: player_detection
sequence: 3
recorded_at: '2026-09-28'
title: 'PR #937 fine-tuned player checkpoint の実frameロード・推論'
issue: 937
provider: codex
session: 01a0e2cc-b573-7011-939d-947448f2d4e0
date: '2026-09-28'
status: done
config:
  checkpoint: ckpt/player_detection/chat-player-v1-e8-best-pr937.pth
  source_checkpoint: ckpt/player_detection/chat-player-v1-e8-best-epoch03.ckpt
  dataset: player_detection/chat-player-v1
  split: test
  confidence: 0.3
  short_side: 800
  max_long_side: 1333
  device: cuda
  matmul_precision: high
metrics:
  gpu_tests_passed: 1
  frames_verified: 1
  detections_above_threshold: 2
repro:
  commit: 665d10d1e25085e991678910dd216e791b29817c
  branch: codex/pr937-player-inference
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PLAYER_DETECTION_GPU_TEST=1 PLAYER_DETECTION_ARTIFACT_ROOT=/home/kamimura/projects/tennis-lab
    .venv/bin/python -m pytest -n 0 tests/integration/tasks/player_detection/test_exported_inference.py
    -q
artifacts:
  run_dir: knowledge/runs/run-pr937-player-inference-real-frame-20260928
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790554677533710453_1498875_pr937-player-inference-real-frame-20260928.log
parents: []
relations: []
papers: []
tags:
- dino
- swin-l
- export-checkpoint
- real-frame-inference
---

## 検証結果

PR #937のbest epoch 3 Lightning checkpointをckpt/player_detectionへ移動し、元の出力パスをsymlinkで保持した。推論形式にexportした`.pth`の`source_checkpoint_sha256`は移動したLightning checkpointのSHA-256と一致した。`DinoPlayerDetector`がplayer専用の`tennis_lab` provenanceを検査してから、共通`DinoPersonDetector`の前処理・strict state-dict load・元画素xyxy decodeを使う。

共有training queueのall予約でCUDA実frameテストを1件実行し、passした。test splitの最初のframe `-6UwVW0DeO4__f056b9d6649bee3a__f000000496-000000821:0`を読み、2個の検出を得た。両方のboxは学習時保存予測と絶対誤差1 pixel以内、confidenceは0.001以内で一致。検出器のload済み状態とexport metadata epoch 3を確認し、unload後の状態も検証した。参照予測のscoreは0.9780と0.9584であり、実推論出力は上記許容差内。推論コードはqueue reproの`uncommitted.patch`に含む（起動前にstage済み）。

通常のCPU unit suiteは23 pass、実artifact GPUテストは通常実行では明示的にskipし、queue実行では1 pass。ruffとmypyも通過した。GPUテストは1フレームのロード・数値再現性確認であり、全動画の精度を再測定したものではない。全動画の性能と観客席への誤検出は前回の別評価資料を参照する。元学習はaugmentation修正前であり、修正後に再学習したとの主張はしない。

## 再現

bundleの`run.json`と`repro.sh`にqueueコマンド、head、provider/sessionを保存した。`PLAYER_DETECTION_ARTIFACT_ROOT`は元repo rootを指す。DINO submoduleとCUDA extensionが必要。pytestは`-n 0`で1プロセス実行する。学習曲線は本runに無く、目的も追加学習ではない。
