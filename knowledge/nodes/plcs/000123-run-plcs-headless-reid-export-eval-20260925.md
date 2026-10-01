---
id: run-plcs-headless-reid-export-eval-20260925
type: run
task: plcs
sequence: 123
recorded_at: '2026-09-25'
title: Re-ID補助headを削除したcheckpointを100sceneで再評価
issue: 915
provider: codex
session: 01a0d0f9-6057-7961-ad6b-adc46744d619
date: '2026-09-25'
status: done
config:
  model: plcs_player_reid
  contract: plcs_fixed_track_reid_v2
  loss: cosine_pair_only
  hidden_dim: 256
  num_stages: 4
  num_heads: 8
  ffn_dim: 768
  num_slots: 4
  seq_len: 512
  data: plcs/tracked_person_reid_v1
  test_scenes: 100
  source_checkpoint_epoch: 41
  retrained: false
  compile: false
  precision: bf16-mixed
metrics:
  loss: 0.058521
  pair_precision: 0.979757
  pair_recall: 0.94902
  pair_f1: 0.964143
  pair_balanced_accuracy: 0.96935
  cosine_threshold: 0.775
  matching_precision: 0.993757
  matching_recall: 0.936275
  matching_f1: 0.964159
  group_accuracy: 0.77
  embeddings_bitwise_equal: true
  max_embedding_difference: 0.0
repro:
  commit: 5d7beca7a404aaf2d3a596ab2e5f23dc88940c1e
  branch: codex/plcs-reid-headless
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
    .venv/bin/python tests/benchmarks/reid_checkpoint.py --checkpoint /home/kamimura/projects/tennis-lab/outputs/plcs/exports/player-reid-headless-v2-s42.ckpt
    --reference-predictions /home/kamimura/projects/tennis-lab/outputs/plcs/train/fixed_track_reid_eager_v1_s42/predictions/pred_test.npz
    --output /home/kamimura/projects/tennis-lab/outputs/plcs/evaluate/headless_reid_v2_20260925
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-plcs-headless-reid-export-eval-20260925
  predictions: knowledge/runs/run-plcs-headless-reid-export-eval-20260925/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790299213752921401_51561_plcs-headless-reid-export-eval-20260925.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/evaluate/headless_reid_v2_20260925
  evaluation: knowledge/runs/run-plcs-headless-reid-export-eval-20260925/evaluation.json
  checkpoint: /home/kamimura/projects/tennis-lab/outputs/plcs/exports/player-reid-headless-v2-s42.ckpt
  checkpoint_sha256: 0f3ea17a6c7c09bb898372cd76b518f9c092b175699e610585b3cfc054fa7451
  source_checkpoint_sha256: 0434772423dead3072f7afebe22c126c99909a860e004c2f8affc2179f185207
parents:
- run-plcs-fixed-track-reid-eager-e60-s42-20260924
relations: []
papers: []
tags:
- reid
- fixed_tracks
- auxiliary_head_removal
- checkpoint_export
- synthetic
- scene_split
---

## 結果

ユーザーの指示により、人物判定の補助headをモデル・損失・推論・設定から削除した。Re-IDの出力を`track_embedding`と`track_valid`に限定し、観測のある全trackをcosine matchingへ渡す。学習用の偽人物track追加も撤去し、観測のあるtrackには人物ID教師を必須とする。sideは独立モデルのまま、今回も学習していない。

既存60epoch学習の`epoch=41`を明示的にv2形式へexportした。取り除いたパラメータは`model.player_head.weight`と`model.player_head.bias`だけで、残る66tensorとcosine閾値0.775は元checkpointと同一である。通常のロード経路で旧契約を暗黙変換せず、export時の出自を保存する。optimizer/scheduler状態は引き継がず、exportは推論または重み初期化用とする。

同じ固定100scene testをCUDAで再評価し、今回の通常matching経路でprecision=0.993757、recall=0.936275、F1=0.964159、group完全一致77/100 sceneだった。cosineペア判定F1は0.964143である。旧保存予測とscene/教師/観測maskを照合し、全sceneの人物embeddingがbitwise一致、最大絶対差0.0であることを確認した。新しい保存予測に`is_player_logit`は含まれない。

## 解釈と制限

親runの「補助headを使わない診断」を、新しい出力契約と通常の推論処理で再現した結果である。従来の補助head込みF1=0.854260・group完全一致33%からの変化は、同じembeddingに対するtrack棄却の撤去によるもので、表現学習の改善ではない。checkpointと閾値の再選定はしていない。

今回は再学習していない。保持したembedding重みは、以前の補助損失・偽track augmentationを含む学習で得られたものである。新コードで最初からpair lossだけを学習した精度は未確認。合成scene splitの同じtestを再利用した整合性確認であり、新しい独立test、未見source motion、実動画、5view以上の精度を示すものではない。sideアーキテクチャと実動画pipelineの採用判断は保留する。

次にRe-IDを再学習する場合は、現在の人物track入力契約とpair lossを固定してvalidationで選定し、headなしで学習する効果を今回の重みexportと区別して比較する。sideの構造はRe-IDの対応失敗を踏まえて別途検討する。

このrunは評価のみで、TensorBoard loggerを無効にしているため学習曲線はない。元の学習曲線は親runに保存されている。今回の予測、metric、embedding一致結果、実行commitとコマンドは本runのbundleに保存した。
