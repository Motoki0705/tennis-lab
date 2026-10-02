---
id: run-i935-evidence-mixed-e9-trainval-r17-20260930
type: run
task: ball_refiner
sequence: 15
recorded_at: '2026-09-30'
title: 'epoch 9証拠cacheの全件回収: bf16候補recallとの一致と旧ft-e13比較'
issue: 935
provider: codex
session: 01a0ef7d-1eb8-7973-b7c6-8b0604efd8ef
date: '2026-09-30'
status: done
config:
  selected_epoch: 9
  splits:
  - train
  - val
  sources:
  - tracknet
  - meiji
  - chat_annotation
  precision: float32
  context: not_generated
  K: 8
  stride: 4
  batch_size: 4
metrics:
  clips: 329
  frames: 145767
  npz_bytes: 116274650
  resource_usage:
    status: complete
    seconds: 2962.9155960429925
    peak_cuda_allocated_bytes: 1466511360
    peak_cuda_reserved_bytes: 1786773504
    pytorch_allocator_limit_bytes: 6442450944
    queue_job: 1790725512433633211_3869834_i935-evidence-mixed-e9-trainval-r17-20260930
    output: /home/kamimura/projects/tennis-lab/data/ball_refiner/detector-mixed-e9-trainval-r17-20260930
  validation:
    chat_annotation:
      frames: 11610
      observed: 6622
      recalled_at_1: 5557
      recalled_at_k: 5896
      wrong_ranked_above_true: 339
      wrong_strictly_higher_score: 339
      not_in_candidates: 726
      recall_at_1: 0.8391724554515252
      recall_at_k: 0.8903654485049833
      not_in_candidates_rate: 0.10963455149501661
      wrong_ranked_above_true_rate: 0.05119299305345817
      wrong_ranked_above_true_given_recalled: 0.0574966078697422
      wrong_strictly_higher_score_rate: 0.05119299305345817
      rank_only_due_to_tie: 0
    meiji:
      frames: 26961
      observed: 23007
      recalled_at_1: 18054
      recalled_at_k: 20903
      wrong_ranked_above_true: 2849
      wrong_strictly_higher_score: 2849
      not_in_candidates: 2104
      recall_at_1: 0.7847176946146824
      recall_at_k: 0.9085495718694311
      not_in_candidates_rate: 0.09145042813056896
      wrong_ranked_above_true_rate: 0.12383187725474855
      wrong_ranked_above_true_given_recalled: 0.1362962254221882
      wrong_strictly_higher_score_rate: 0.12383187725474855
      rank_only_due_to_tie: 0
    meiji/cam0:
      frames: 8987
      observed: 7556
      recalled_at_1: 5419
      recalled_at_k: 6597
      wrong_ranked_above_true: 1178
      wrong_strictly_higher_score: 1178
      not_in_candidates: 959
      recall_at_1: 0.7171784012705135
      recall_at_k: 0.8730809952355744
      not_in_candidates_rate: 0.12691900476442564
      wrong_ranked_above_true_rate: 0.15590259396506087
      wrong_ranked_above_true_given_recalled: 0.17856601485523724
      wrong_strictly_higher_score_rate: 0.15590259396506087
      rank_only_due_to_tie: 0
    meiji/cam1:
      frames: 8987
      observed: 7999
      recalled_at_1: 6761
      recalled_at_k: 7613
      wrong_ranked_above_true: 852
      wrong_strictly_higher_score: 852
      not_in_candidates: 386
      recall_at_1: 0.845230653831729
      recall_at_k: 0.9517439679959995
      not_in_candidates_rate: 0.0482560320040005
      wrong_ranked_above_true_rate: 0.10651331416427054
      wrong_ranked_above_true_given_recalled: 0.11191383160383554
      wrong_strictly_higher_score_rate: 0.10651331416427054
      rank_only_due_to_tie: 0
    meiji/cam2:
      frames: 8987
      observed: 7452
      recalled_at_1: 5874
      recalled_at_k: 6693
      wrong_ranked_above_true: 819
      wrong_strictly_higher_score: 819
      not_in_candidates: 759
      recall_at_1: 0.788244766505636
      recall_at_k: 0.8981481481481481
      not_in_candidates_rate: 0.10185185185185185
      wrong_ranked_above_true_rate: 0.10990338164251208
      wrong_ranked_above_true_given_recalled: 0.12236665172568355
      wrong_strictly_higher_score_rate: 0.10990338164251208
      rank_only_due_to_tie: 0
    tracknet:
      frames: 1573
      observed: 1538
      recalled_at_1: 1513
      recalled_at_k: 1523
      wrong_ranked_above_true: 10
      wrong_strictly_higher_score: 10
      not_in_candidates: 15
      recall_at_1: 0.9837451235370611
      recall_at_k: 0.9902470741222367
      not_in_candidates_rate: 0.00975292587776333
      wrong_ranked_above_true_rate: 0.006501950585175552
      wrong_ranked_above_true_given_recalled: 0.006565988181221274
      wrong_strictly_higher_score_rate: 0.006501950585175552
      rank_only_due_to_tie: 0
repro:
  commit: 9f2256370cb52af71c6edf28d58ce7182923adb7
  branch: campaign930/i935-10-detector-selection
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: timeout --signal=TERM --kill-after=15s 5385s env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-10-detector-selection
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-10-detector-selection/.venv/bin/python
    -u /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-10-detector-selection/knowledge/runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/rebuild_cache.py
    --plan /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-10-detector-selection/knowledge/runs/run-i935-mixed-ft-val-recall-s42-r16-20260929/cache_plan.json
artifacts:
  run_dir: knowledge/runs/run-i935-evidence-mixed-e9-trainval-r17-20260930
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790725512433633211_3869834_i935-evidence-mixed-e9-trainval-r17-20260930.log
  output_dir: /home/kamimura/projects/tennis-lab/data/ball_refiner/detector-mixed-e9-trainval-r17-20260930
  collection: knowledge/runs/run-i935-evidence-mixed-e9-trainval-r17-20260930/collection.json
  recall: knowledge/runs/run-i935-evidence-mixed-e9-trainval-r17-20260930/recall.md
parents:
- run-i935-mixed-ft-val-recall-s42-r16-20260929
relations:
- to: run-i935-evidence-ft-e13-trainval-r3-20260928
  rel: compares
papers: []
tags:
- detector-cache
- validation-only
- context-free
- not-deployed
---

## 回収結果

queueのdone配置・`state=done`（wrapperがexit 0でのみ公開）、生成resourceのcomplete、
固定plan、live/snapshot manifest、全329 NPZのSHA-256・byte数、全clipの新しいCPU reader検証が一致した。
[回収結果](../../runs/run-i935-evidence-mixed-e9-trainval-r17-20260930/collection.json)と
[全artifact hash](../../runs/run-i935-evidence-mixed-e9-trainval-r17-20260930/artifact_hashes.json)が証拠。
train 105,623 / val 40,144、合計145,767 frame。NPZは116,274,650 bytes。
manifest SHA-256は`4101cd9f0f480809dfbc8260e9266a453ec65184639d6af4b23bf8bf66a5c2f6`、
checkpointはepoch 9の`37f4c59aead00062829280ad591b3874886978891104a290f704a2c33c3c1b36`。
実時間2,962.916秒、PyTorch peak allocated 1.467 GB / reserved 1.787 GB。
これはGPU全体のVRAM実測ではない。cacheは生成済みの新規directoryへ保持し、旧cacheは変更していない。

## validationの妥当性

[同一frameのsource/camera表](../../runs/run-i935-evidence-mixed-e9-trainval-r17-20260930/recall.md)は
K=8、半径20 source px、単一observedのみ。新cache Meiji 23,007 frameの
recall@8 / @1は0.908550 / 0.784718、旧r3は0.742470 / 0.526275。
新cacheとepoch 9 bf16 Lightning validationの差は+0.0130 / +0.0435 pp、
camera別でも最大0.1208 pp。1 ppを超える差はなく、dtypeの違いと整合する小さい差である。
この比較で検出器を選び直していない。epoch 9の選択はユーザー判断に従い既に固定済み。
旧cacheとr15 float32/batch2比較（約0.743 / 0.526）の小差も1 pp未満で、
r3とr17は同じfloat32/batch4条件である。

全件readerはframe/PTS/実秒・座標・patch格子・checksumを検査した。
store metadata/index、旧manifest、checkpoint、生成codeは計画のhashと再照合したが、
元media/注釈/JPEG shard本体はこの回収では再hashしていない。JPEGは生成時の前後検査を継承する。
test video_001の推論・採点は行っていない。キャッシュ生成には学習曲線やTensorBoardはない。

## 次の実験

同じseed42、12 epoch / 3,000更新、選択/較正split、文脈なしrecipeを維持し、
r4の入力cacheだけをこの成果へ変更する。r5と同じ未較正診断を行い、
旧pilot・検出器との同一val frame比較を追加する。#964完了前にはperson/poseを使用しない。
採点不能なunknown/不存在位置を位置教師として補完しない。次runで結果回収とval overlayを行う。
