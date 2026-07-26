---
id: run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep
type: run
title: BLCS point-attention 32 / 2D RoPE lifecycle baseline (#648)
issue: 648
provider: codex
session: 019f8335-ad14-7370-8875-6b082d415aea
date: '2026-07-21'
status: done
config:
  model: track_query_large_point_attention
metrics:
  loss: 0.448417
  loss_position: 0.24477
  loss_position_x: 0.232984
  loss_position_y: 0.237742
  loss_position_z: 0.282399
  loss_presence: 0.203646
  loss_smoothness: 0.0
  loss_gravity: 0.0
  position_error: 1.125558
  presence_precision: 0.66
  presence_recall: 0.897417
  presence_f1: 0.751386
  lifecycle_presence_f1: 0.751386
  birth_frame_error: 29.280127
  death_frame_error: 37.524185
  query_reuse_count: 1.24
  illegal_overlap_count: 0.0
  segment_id_switches: 24.82
  id_switches: 24.82
  duplicate_active_tracks: 15.11
  missed_gt_frames: 27.290001
  inactive_query_false_positives: 85.360001
  position_mae_x_m: 2.973152
  position_mae_y_m: 6.510261
  position_mae_z_m: 0.573248
repro:
  commit: fc45020ec9bb40f4d49b0bb81a5d2822f33f3afc
  branch: feat/issue-643-648-multi-object-tracking
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.blcs.scripts.train
    --config-name train_tracking_chunked model=track_query_large_point_attention model.mask_invisible_observations=false
    data.scene_dir=data/blcs/multi_object_lifecycle_v4 data.batch_size=1 data.seq_len_range='[512,512]'
    data.num_views_range='[3,5]' data.num_workers=4 data.chunk.generation_workers=16
    data.chunk.epochs_per_chunk=3 data.chunk.prefetch_chunks=5 training.trainer.max_epochs=100
    +training.trainer.accumulate_grad_batches=8 training.trainer.check_val_every_n_epoch=5
    training.qualitative_logging.enabled=true training.qualitative_logging.every_n_epochs=10
    training.early_stopping.enabled=false run.output_dir=outputs/blcs/tracking_lifecycle_v4_large_pointattn32_rope2d_t512_b1_100ep
artifacts:
  run_dir: knowledge/runs/run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep
  predictions: knowledge/runs/run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep/pred_test.npz
  log: .training_queue/logs/1784632739204612693_1513834_i648_blcs_lifecycle_v4_large_pointattn32_rope2d_t512_b1_100ep.log
  output_dir: outputs/blcs/tracking_lifecycle_v4_large_pointattn32_rope2d_t512_b1_100ep/logs/version_0
  curves: knowledge/runs/run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep/curves.png
  tb_logdir: outputs/blcs/tracking_lifecycle_v4_large_pointattn32_rope2d_t512_b1_100ep/logs/version_0
parents:
- run-issue-648-multiball-baseline
relations: []
tags:
- blcs
- multi-ball
- tracking
- lifecycle
- point-attention
- mrope
---

## 考察 / Findings

### 要約

512 frameのlifecycle multi-ballデータに対する100 epoch runは、testで
`presence_f1=0.751386`、`position_error=1.125558`となった。recallは
`0.897417`だがprecisionは`0.660000`で、inactive queryのfalse positiveと
duplicate trackが主な残存課題である。

### アーキテクチャ詳細

`track_query_large_point_attention`を使用し、入力は
`data/blcs/multi_object_lifecycle_v4`、sequence長は512、view数は3–5、
batch sizeは1、gradient accumulationは8である。不可視観測をattention
keyから除外しない`model.mask_invisible_observations=false`条件で、chunkを
3 epochごとに更新し、100 epoch学習した。具体的な再現commandと実行時patchは
`artifacts.run_dir`に保存されている。

### メトリクスの解釈

presenceはrecall `0.897417`に対してprecision `0.660000`でrecall寄りである。
`inactive_query_false_positives=85.360001`、`duplicate_active_tracks=15.110000`、
`id_switches=24.820000`は、存在しないtrackの抑制とidentity維持が未解決で
あることを示す。一方、`illegal_overlap_count=0.0`であり、lifecycle slotの
禁止された重複は観測されなかった。位置の軸別MAEはx `2.973152 m`、
y `6.510261 m`、z `0.573248 m`で、特にcourt長手方向のy誤差が大きい。
収束曲線ではvalidation lossが約0.63から約0.47へ低下した後に頭打ちとなった。
position lossはtrain/validationとも低下したが、後半はtrain約0.25に対して
validation約0.27で小さなgeneralization gapが残った。

### アーキテクチャ⇄メトリクスの因果考察

観測事実として、不可視tokenをmemoryとして残す条件でも高いrecallを得た一方、
precisionとidentity診断は低い。仮説として、長いlifecycleとquery再利用に対し、
不可視memoryおよびpresence判定がinactive queryを十分に抑制できず、false
positive・duplicate・ID switchへつながった可能性がある。このrun単独から
point-attentionまたは2D RoPEの個別寄与は断定できない。

### 既存実験との比較

親の`run-issue-648-multiball-baseline`は`presence_f1=0.847570`、
`position_error=0.337709`だった。ただし親は旧ball-tracking taskの短い
8–12 frame / 5 epochデータであり、本runはBLCSの512 frame lifecycle /
100 epoch条件なので、数値を直接の改善・悪化として扱わない。本runは現行BLCS
lifecycle構成に対する比較起点である。

### 次に有効な実験

同じHydra commandとデータ設定を保ち、object-ID順入力契約を適用したqueued run
`i648_blcs_lifecycle_v4_large_pointattn32_id_ordered_t512_b1_100ep`を完走させる。
両runのcommandはoutput directory以外で一致するが、実行時のuncommitted patchは
異なるため、結果の帰属時にはrepro bundle間のpatch差分も併記する。
