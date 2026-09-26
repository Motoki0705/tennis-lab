---
id: run-plcs-fixed-track-reid-eager-e60-s42-20260924
type: run
task: plcs
sequence: 122
recorded_at: '2026-09-24'
title: 固定track Re-IDを60epoch学習し100sceneで対応精度を評価
issue: 915
provider: codex
session: 01a0d0f9-6057-7961-ad6b-adc46744d619
date: '2026-09-24'
status: done
config:
  model: plcs_player_reid
  hidden_dim: 256
  num_stages: 4
  num_heads: 8
  ffn_dim: 768
  num_slots: 4
  seq_len: 512
  batch_size: 4
  accumulate_grad_batches: 4
  epochs: 60
  seed: 42
  compile: false
  precision: bf16-mixed
  learning_rate: 0.0003
  weight_decay: 0.1
  warmup_steps: 100
  data: plcs/tracked_person_reid_v1
  train_scenes: 800
  val_scenes: 100
  test_scenes: 100
  selected_checkpoint_epoch: 41
metrics:
  loss: 0.106688
  pair_precision: 0.979757
  pair_recall: 0.94902
  pair_f1: 0.964143
  pair_balanced_accuracy: 0.96935
  player_accuracy: 0.839552
  cosine_threshold: 0.775
  matching_precision: 0.997382
  matching_recall: 0.747059
  matching_f1: 0.85426
  group_accuracy: 0.33
  track_acceptance_recall: 0.839552
  diagnostic_without_targetness_matching_precision: 0.9937565036420395
  diagnostic_without_targetness_matching_recall: 0.9362745098039216
  diagnostic_without_targetness_matching_f1: 0.9641595153962645
  diagnostic_without_targetness_group_accuracy: 0.77
  same_side_matching_f1: 0.8685524126455907
  opposite_side_matching_f1: 0.8469991546914624
repro:
  commit: 3ca9b8a4d3792dc613d2f30443c8d9e03c12c83c
  branch: codex/plcs-track-reid
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    .venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_reid paths.project_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-track-reid
    paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs
    paths.artifact_root=/home/kamimura/projects/tennis-lab/outputs paths.checkpoint_root=/home/kamimura/projects/tennis-lab/outputs
    data.scene_dir=plcs/tracked_person_reid_v1 run.output_dir=plcs/train/fixed_track_reid_eager_v1_s42
    training.compile.enabled=false training.trainer.enable_progress_bar=false
artifacts:
  run_dir: knowledge/runs/run-plcs-fixed-track-reid-eager-e60-s42-20260924
  predictions: knowledge/runs/run-plcs-fixed-track-reid-eager-e60-s42-20260924/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790220288389042047_787832_plcs-fixed-track-reid-eager-e60-s42-20260924.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/train/fixed_track_reid_eager_v1_s42
  checkpoint: /home/kamimura/projects/tennis-lab/outputs/plcs/train/fixed_track_reid_eager_v1_s42/logs/version_0/checkpoints/plcs-epoch=41.ckpt
  checkpoint_sha256: 0434772423dead3072f7afebe22c126c99909a860e004c2f8affc2179f185207
  tb_logdir: outputs/plcs/train/fixed_track_reid_eager_v1_s42/logs/version_0
  curves: knowledge/runs/run-plcs-fixed-track-reid-eager-e60-s42-20260924/curves.png
parents:
- run-plcs-reid-eager-gpu-smoke-20260924
relations:
- to: run-plcs-fixed-track-reid-e60-s42-20260924
  rel: compares
papers: []
tags:
- reid
- fixed_tracks
- eager
- synthetic
- scene_split
- auxiliary_head_generalization
---

## 結果

Re-IDだけをseed42で初期化し直し、eager/bf16で60epochを完了した。100sceneのvalidationで最低lossだった`epoch=41`（42epoch目）を選び、そこに保存されたcosine閾値0.775を固定して100sceneのtestへ適用した。sideは学習していない。新しいcheckpointは上記pathとSHAで固定する。

| test指標 | precision | recall | F1 |
|---|---:|---:|---:|
| cosineのペア判定 | 0.979757 | 0.949020 | 0.964143 |
| 補助headを含む既定matching | 0.997382 | 0.747059 | 0.854260 |
| 補助headを使わない診断matching | 0.993757 | 0.936275 | 0.964160 |

既定matchingの人物group完全一致は33/100 scene、対象trackの受理率は0.839552だった。補助headを使わない診断では完全一致77/100 sceneとなった。この診断は保存済みのembedding・同じ閾値を使い、checkpointの再選定やtestによる閾値調整はしていない。既定pipelineの成績と診断値を混同しない。

## 誤りの切り分け

既定matchingはTP762/FP2/FN258で、高precisionに対して取りこぼしが多い。補助headを外した診断はTP955/FP6/FN65だった。したがって誤検出trackを除外する補助headによる真の人物の棄却が、主なrecall低下要因と判断する。少人数で顕著で、既定group完全一致は観測1人が2/35、2人が3/19、3人が13/25、4人が15/21だった。

同じcourt sideのcamera pairでは既定F1=0.868552、反対sideでは0.846999。side教師はこの層別診断だけに使い、Re-ID forwardへは渡していない。sideアーキテクチャの有効性を測った結果ではない。

学習後半はtrainの補助head精度がほぼ1となる一方、clean validationの対象人物受理が悪化した。現在のFP augmentationはjoint dropoutの後に全joint可視のFPを追加するため、欠測率を学習上の手掛かりにする可能性がある。これはコードと分布からの仮説で、直接のablationでは未検証。対象人物trackだけを上流から受け取る契約なら、この補助headの責務自体を取り除く選択が考えられる。入力契約についてユーザーへ確認中であり、今回の本学習途中では構造を変更していない。

## 比較の制限と次の判断

合成の固定scene split 800/100/100、各scene1〜4人、生成4cameraから3〜4viewを選んだ評価である。camera-local IDを独立に付けてslot順のGT漏洩を避けているが、未見source motion、実2D tracker、実動画、5view、全camera合計5人以上の精度は確認していない。実装の容量/置換テストとモデルの学習分布を区別する。

旧shared side/IDモデルとは入力track仮定・人数分布・loss・architectureが変わっており、精度差をslot固定だけの因果効果と解釈しない。compiled失敗runからのresumeも行っていない。次は対象track選別の責務を確定し、誤検出除外と純粋な人物対応を分けて検証する。その結果を踏まえて独立sideの構造を決める。現時点で実動画pipelineへのproduction採用を主張しない。

学習/validation曲線、test予測、再計算スクリプト、camera side別・人数別集計、失敗scene一覧をrun bundleへ保存した。元2D入力の監査はエラー0件である。
