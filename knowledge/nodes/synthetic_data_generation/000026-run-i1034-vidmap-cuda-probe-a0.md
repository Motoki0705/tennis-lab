---
id: run-i1034-vidmap-cuda-probe-a0
type: run
task: synthetic_data_generation
sequence: 26
recorded_at: '2026-10-07'
title: 'VidMap用CUDA runtime probe: RTX 5060 Tiでcompile/attention確認'
issue: 1034
provider: codex
session: 01a1106e-187c-7163-be4e-9f1673767ad9
date: '2026-10-07'
status: done
config:
  model: runtime probe only
  data: small synthetic tensors; no tennis input
  seed: 42
  torch: 2.14.0+cu130
  torchvision: 0.29.0
  xformers: 0.0.35
  gpu: RTX 5060 Ti
metrics:
  elapsed_seconds: 15.653946
  device_peak_memory_mib: 1430.0
  sfm/torch_peak_allocated_bytes: 10240.0
repro:
  commit: 2eeea0faecac81da5808f34e68dc792ac94c2b8c
  branch: research/sfm-night-20261006
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: /home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/.venv/bin/python
    -m experiments.sfm_comparison.run_command --campaign /home/kamimura/projects/tennis-lab/outputs/aris/i1034-sfm-night-20261006
    --run-id vidmap-cuda-probe-a0 --stage sanity --seconds 180 --metrics-source /home/kamimura/projects/tennis-lab/outputs/aris/i1034-sfm-night-20261006/environment/cuda-probe.json
    --required-path /home/kamimura/projects/tennis-lab/outputs/aris/i1034-sfm-night-20261006/environment/cuda-probe.json
    -- /home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/.cache/runtimes/vidmap-native/bin/python
    -m experiments.sfm_comparison.probe_runtime --output /home/kamimura/projects/tennis-lab/outputs/aris/i1034-sfm-night-20261006/environment/cuda-probe.json
artifacts:
  run_dir: knowledge/runs/run-i1034-vidmap-cuda-probe-a0
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791301965073915284_102935_i1034-vidmap-cuda-probe-a0.log
parents:
- run-i1034-nht-b00-p90-s42-a1
relations: []
papers: []
tags:
- sfm
- aris
- runtime-smoke
---

## 観測

共有queue経由で小さなCUDA tensorを使い、torch.compileの結果がeagerと一致すること、xFormers attentionが有限値を返すことを確認した。約15.65秒で完走し、RTX 5060 Ti、PyTorch 2.14.0+cu130、CUDA 13.0、xFormers 0.0.35を報告した。

## 限界

VidMap本体、事前学習重み、テニス画像は実行していない。16GBでfull frontendが収まることやSfM精度を確認した結果ではない。小tensorのTorch peak allocated bytesはfullモデルのVRAM見積もりに使わない。SfM task下の技術的な事前確認として記録し、手法比較表の精度欄には入れない。学習曲線はない。

## 次

固定版COLMAP/PyCOLMAPとVidMap native extensionの構築、短い実画像sanity、同じ90枚でのSfMへ進む。ホスト停止で破損した可能性があるwheel cacheについて、cuda-bindingsとtorchvisionのmetadataエラーを確認したため、このruntimeのPyTorch群はno-cacheで再取得した。既存のproject runtimeは変更していない。
