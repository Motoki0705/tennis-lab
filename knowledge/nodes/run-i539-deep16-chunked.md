---
id: run-i539-deep16-chunked
type: run
title: i539_deep16_chunked
issue: 539
provider: claude
session: 8722d9dc-5894-4536-8e54-d03e3e34949a
date: '2026-06-21'
status: done
config:
  model: multiview_axial_split_asym_deep16
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.632468
  position_error_std_m: 0.427218
  position_error_median_m: 0.570044
  angular_error_deg: 19.106464
  angular_error_std_deg: 17.564915
  angular_error_median_deg: 15.116028
  x_error_m: 0.386135
  y_error_m: 0.393653
  z_error_m: 0.054669
  position_accuracy: 0.444144
  angle_accuracy: 0.496832
  position_accuracy_0.5m: 0.444144
  position_accuracy_1m: 0.870986
  position_accuracy_2m: 0.975792
  angle_accuracy_10deg: 0.344749
  angle_accuracy_15deg: 0.496832
  angle_accuracy_30deg: 0.804058
repro:
  commit: d407e54cdb903d7082aa4011b2a6f8cb0426c7cc
  branch: exp/i525-asym
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split_asym_deep16 data.batch_size=4
    training.trainer.accumulate_grad_batches=2 data=chunked_multiview_sequence_bs8
    data.seq_len_range=[64,256] loss=canonical_rot training.trainer.max_epochs=200
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i539-deep16-chunked
  predictions: knowledge/runs/run-i539-deep16-chunked/pred_test.npz
  log: .training_queue/logs/1782037138483198726_185287_i539_deep16_chunked.log
parents:
- run-i535-asym-deep16-rerun
- run-i518-exp10
relations:
- to: run-i535-asym-deep16-rerun
  rel: compares
- to: run-i539-wide-chunked
  rel: compares
- to: run-i539-ex10-chunked
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- asymmetric
- depth
- chunked
- data-rich
- capacity-frontier
---

## 考察 / Findings

非対称深さ deep16(rot16/512, 142.3M)を **chunked**(data-rich)で学習。effective batch=8(bs4×accum2), seq[64,256], test は固定 scene_dir。

- **test: 回転 19.11°(median 15.12)/ 位置 0.632m**。固定データ deep16 再実行(10.41°/0.252m)より**悪化**、かつ同一 chunked 条件で **ex10_chunked(15.84°)よりも悪く、本群最下位**。
- **ep86 で early-stop**(3 群中最速)。train ang 13.50° vs val 18.90° ＝ 未収束気味で plateau。
- **結論(幅 vs 深さ, data-rich)**: 容量を「深さ(rot 偏重)」に振る deep16 は、同予算帯の「幅」振り(wide, 228.7M, 10.33°)に**大きく劣る**。固定・chunked いずれでも深さ偏重は非推奨で、[[run-i539-wide-chunked]] の幅広モデルが優位。深さで回転を伸ばす当初仮説(#535)は data-rich でも不成立。
- early-stop×chunk ローテーション交絡の注意は [[run-i539-ex10-chunked]] と同様。
