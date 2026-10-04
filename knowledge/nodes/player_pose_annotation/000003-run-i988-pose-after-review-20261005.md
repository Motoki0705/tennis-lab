---
id: run-i988-pose-after-review-20261005
type: run
task: player_pose_annotation
sequence: 3
recorded_at: '2026-10-05'
title: 選手選別後pose生成とball-mix-v2拡張campaignの起動
issue: 988
provider: codex
date: '2026-10-05'
status: running
config:
  generation_mode: review_then_pose.v1
  tracking: strongsort_pp_appearance
  pose_weight: 0
  presence_threshold: 0.4
  review_model: gpt-6.1-sol
  review_effort: max
  review_parallel: 2
  gpu_resource: all
  expanded_store_metadata_sha256: 3c9ebf405ce8346fe692d6bec251614ca97bb0f528b6d7b04082b97f9dbad309
  expanded_store_index_sha256: f552dae1e15094facfa782c9a9483588169ff7984e01fca5c1c0eecb560513c5
metrics:
  planned_clips: 1300
  presence_selected_clips: 1172
  presence_skipped_clips: 128
  added_clips: 485
  added_selected_clips: 483
  unit_tests_passed: 114
  verified_reuse_clips: 442
  new_generation_clips: 730
  selected_frames: 728439
artifacts:
  campaign: /home/kamimura/projects/tennis-lab/outputs/chat_annotation/player_pose/ball-mix-v2-pose-after-review-20261005
  dataset: /home/kamimura/projects/tennis-lab/data/ball_detection/ball-mix-v2-player-pose-v2-20261005
  input_snapshot: /home/kamimura/projects/tennis-lab/outputs/chat_annotation/player_pose/i988-input-20261005/ball-mix-v2
  run_dir: knowledge/runs/run-i988-pose-after-review-20261005
  log: /home/kamimura/projects/tennis-lab/outputs/chat_annotation/player_pose/ball-mix-v2-pose-after-review-20261005/orchestrator.log
parents:
- run-player-pose-v2-pilot-00034-20261002
relations: []
papers: []
tags: []
session: 01a106ac-abf2-73b3-8167-796a547cbd91
repro:
  commit: e266a85b2
  branch: codex/i988-pose-after-review
  command: .venv/bin/python -u -m src.tennis_scene.chat_annotation.player_pose orchestrate
    --campaign /home/kamimura/projects/tennis-lab/outputs/chat_annotation/player_pose/ball-mix-v2-pose-after-review-20261005
---

# 生成順序と拡張の固定

Issue #988の確定方針に従い、全画面人物検出とCLIP外観からposeなしで追跡し、GPTの選手・非選手・重複判定を確定してから選択選手の実観測だけへViTPoseを適用する実装を追加した。通常sceneのStrongSORT++＋pose/CLIPとpose重み0.15は維持する。raw追跡、承認済み選択、pose、公開を別receiptにし、pose生成後にレビューのraw tracks/hashを書き換えない。

2026-10-05時点の1,300 clipsをmetadata/indexのコピーとJPEG shardのhardlinkで固定した。従来のpresence > 0.4を維持すると1,172 clipsが対象、128 clipsがskipとなる。旧815 clipsからの追加485 clipsのうち483 clipsが対象となる。unresolvedを存在として数え、参考frameとout_of_frameは従来どおり除外する。

旧campaignはユーザー停止済みのまま保持する。旧approved 442 clipsのclip/media/annotation identity、frame/PTS、座標、JPEG bytes、レビュー・raw pose・公開artifact hashは全件の実データ照合に通過した。これらを拡張storeに束縛した別datasetへ再利用し、残る730 clipsを新規生成する。生成方式の出自を残すため、旧subsetに対する#986の学習結果と今回の拡張後の実験は同じデータ条件とは扱わない。

# 検証と限界

CPUの114テストと変更箇所のRuff・mypyは通過した。実際のJPEGを使うfixtureと偽modelで、poseモデルなしの全追跡、非選手・重複・補間へのpose呼出し0件、raw row/player ID/frame/PTS/欠測の保持、chunk再開、queue登録直後の中断回収、未承認・pose未生成の公開拒否を確認した。通常sceneの既定profileの回帰も含む。validator評価は指定されておらず実施していない。

本runはデータ生成システムの起動とデータ対応の確認であり、学習・精度比較ではない。TensorBoardと学習曲線は対象外。全対象の生成完了、GPTによる選手IDの意味的正しさ、姿勢の精度や下流の改善は未確認。次に全体・追加clip別の公開/保留/失敗coverageと全検出数対pose crop数を集計し、旧approved subsetから拡張した際の下流効果を別実験で比較する。

2026-10-05 00:17 JSTに新campaignのorchestratorをPID 2980442で起動した。固定config・identity・plan集計と起動receiptをrun bundleへ保存した。GPU処理はメインrepoの共有training queueだけに登録し、旧campaignは再開していない。
