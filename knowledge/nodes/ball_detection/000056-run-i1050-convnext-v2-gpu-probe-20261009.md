---
id: run-i1050-convnext-v2-gpu-probe-20261009
type: run
task: ball_detection
sequence: 56
recorded_at: '2026-10-09'
title: ConvNeXt V2系CNNのBF16 GPU smoke完了
issue: 1050
provider: codex
status: done
config:
  model: convnext_v2/factorized
  precision: bf16
  batch_size: 1
  updates: 96
  compile_mode: default
  seed: 42
metrics:
  steady_windows_per_second: 2.662008625006982
  peak_allocated_gib: 7.2639923095703125
  peak_reserved_gib: 9.01953125
artifacts:
  run_dir: knowledge/runs/run-i1050-convnext-v2-gpu-probe-20261009
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/mdd-cnn-smoke/20261009/convnext_v2/training
parents:
- run-i1050-cnn-structure-20261009
relations:
- to: run-i986-dpt-pretrain-smoke-20261009
  rel: compares
papers: []
tags:
- gpu-smoke
- bf16
- convnext-v2
- performance
date: '2026-10-09'
repro:
  commit: 4cb8b70c9cae3e7841944e94295615e584947402
  command: /home/kamimura/projects/tennis-lab/.claude/worktrees/ball-cnn-campaign-20261009/.venv/bin/python
    -m src.tasks.ball_detection.scripts.pretrain_mdd_dpt --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/mdd-cnn-smoke/20261009/convnext_v2/smoke-manifest.json
    --model-config /home/kamimura/projects/tennis-lab/.claude/worktrees/ball-cnn-campaign-20261009/src/tasks/ball_detection/configs/model/mdd_dpt_convnext_v2.yaml
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/mdd-cnn-smoke/20261009/convnext_v2/training
    --epochs 1 --windows-per-epoch 96 --warmup-updates 4 --learning-rate .0002 --device
    cuda --precision bf16 --batch-size 1 --jpeg-decoder nvjpeg --image-prefetch --num-workers
    8 --pin-memory --compile-mode default --selection-scope full --log-every 8 --preview-clips
    1
---

ConvNeXt V2系/factorized時間CNNが実720p・BF16・compile defaultで96更新、validation、checkpoint/GIF保存を完了した。3 train clips・3 val clipsの同じdiagnostic窓を使い、両時間層の勾配はfinite/nonzero。graph breakは0、train/evalで2 graph。これはsmoke substageの完了であり、6万更新の本学習の完了を保証するものではない。

最初の8更新を除く速度2.662窓/秒、PyTorch peak allocated7.264GiB、reserved9.020GiB。先のresidual smokeは3.088窓/秒・allocated5.680GiBであり、この短時間測定ではConvNeXt V2系の速度/メモリ改善は得られなかった。パラメータ・MAC削減を実速度改善と扱わない。測定時刻・マシン負荷・初期化消費順が違うため、精密な因果的速度比較ではない。

本学習はsmoke成功後にscratchで開始され、最初の監視で700更新まで進行を確認した。少数clipのsmoke精度を全validationの性能とは比較しない。FasterNetは待機中、baselineは18,000更新のcheckpointからresume予定。全3構成が完了するまでbestの選択と事後学習へ進めない。

TensorBoardなし。関連するouter queue jobは1791536294779637822_3429084_i1050-convnext_v2-s42-u60000.jobで、repro bundleはshared queueに保持される。本学習はその後異常終了し、別の[失敗記録](000057-run-i1050-convnext-v2-s42-u60000.md)に登録した。
