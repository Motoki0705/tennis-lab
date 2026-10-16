---
id: run-i934-mixed-ft-s42-r6
type: run
task: ball_detection
sequence: 21
recorded_at: '2026-09-28'
title: 3 source混合FTは12 epoch完走し、混合validation F1でepoch 0を選択
issue: 934
provider: codex
session: 01a0e5be-53c1-76e2-a5a0-889df629cd9f
date: '2026-09-28'
status: done
config:
  profile: train_meiji_mixed
  model: conv_next_unet
  input_mode: mdd
  data: ball-mix-v1
  source_weights:
    tracknet: 1.0
    meiji: 1.0
    chat_annotation: 1.0
  seed: 42
  max_epochs: 12
  windows_per_epoch: 7680
  batch_size: 4
  learning_rate: 2.0e-05
  checkpoint_monitor: val/f1
  test_after_fit: false
metrics:
  selected_val_f1: 0.5664603114128113
  selected_epoch: 0
  last_val_f1: 0.46595054864883423
  completed_epochs: 12
  optimizer_steps: 23040
repro:
  commit: 8142fc349c268ee567525efc603c11bf705cd5a4
  branch: campaign930/i934-4-mixed-ft
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i934-4-mixed-ft/.venv/bin/python
    -m src.tasks.ball_detection.scripts.train --config-name train_meiji_mixed paths.project_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i934-4-mixed-ft
    paths.data_root=/home/kamimura/projects/tennis-lab/data paths.checkpoint_root=/home/kamimura/projects/tennis-lab/ckpt
    paths.output_root=/home/kamimura/projects/tennis-lab/outputs paths.artifact_root=/home/kamimura/projects/tennis-lab/outputs
    paths.cache_root=/home/kamimura/projects/tennis-lab/.cache paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
    run.output_dir=ball_detection/train/i934_mixed_ft/s42-r6-20260928
artifacts:
  run_dir: knowledge/runs/run-i934-mixed-ft-s42-r6
  log: knowledge/runs/run-i934-mixed-ft-s42-r6/queue.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/i934_mixed_ft/s42-r6-20260928/logs/version_0
  config: knowledge/runs/run-i934-mixed-ft-s42-r6/config.yaml
  training_result: knowledge/runs/run-i934-mixed-ft-s42-r6/training_result.json
  tb_logdir: knowledge/runs/run-i934-mixed-ft-s42-r6
  selected_checkpoint: /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/i934_mixed_ft/s42-r6-20260928/logs/version_0/checkpoints/ball-detection-epoch=00.ckpt
  selected_checkpoint_sha256: 7b9a202b7753edc9edc200271edfad1b09bdee59073bdbaffb2709cbf942aa9b
  curves: knowledge/runs/run-i934-mixed-ft-s42-r6/curves.png
parents:
- run-i618-convnext-v2-ft
relations:
- to: run-i934-mixed-ft-s42-r5
  rel: supersedes
papers: []
tags:
- campaign930
- mixed-ft
- validation-selection
---

## 結果とcheckpoint選択

共有queue job `1790561702018216327_1832087_i934-mixed-ft-s42-r6-20260928` は正常終了した。
ft-e13から全242 tensorをstrictに読み、3 source等比率・seed 42・LR 2e-5で
12 epoch / 23,040 optimizer stepを完走した。前回失敗したvalidation終端の描画も通り、
epoch 0 / 4 / 8の各4 GIFが保存されている。

事前に固定した混合validationのF1最大に従い、**epoch 0（1 epoch学習後）**を選ぶ。
`last.ckpt`の`ModelCheckpoint(monitor=val/f1, mode=max)`のbest path/scoreと、
TensorBoardの12点のF1最大が一致する。最終epoch 11のcheckpointへ置き換えない。
[選択checkpointのpath/hash・全scalar](../../runs/run-i934-mixed-ft-s42-r6/training_result.json)と
[実行log](../../runs/run-i934-mixed-ft-s42-r6/queue.log)、設定・元event・再現bundleを保存した。

| epoch（0始まり） | val F1 | val loss |
|---|---:|---:|
| 0 | 0.566460 | 0.00070278 |
| 1 | 0.489825 | 0.00069620 |
| 2 | 0.476533 | 0.00063845 |
| 3 | 0.520412 | 0.00063950 |
| 4 | 0.491147 | 0.00063642 |
| 5 | 0.442954 | 0.00066595 |
| 6 | 0.371757 | 0.00063908 |
| 7 | 0.456643 | 0.00061420 |
| 8 | 0.484254 | 0.00059597 |
| 9 | 0.477232 | 0.00059061 |
| 10 | 0.461998 | 0.00059502 |
| 11 | 0.465951 | 0.00059581 |

## 解釈と限界

validationはTrackNet・Meiji・chat_annotationを混ぜた非重複窓による学習中の採点で、
Meiji test video_001の全frame評価ではない。F1は初回以降改善せず、
val lossの低下とも一致しない。追加epochだけで汎化が良くなるという仮説は支持しない。
source別の変化・confidenceの変化・忘却のいずれが原因かは、この集約曲線だけでは分からない。

`test_after_fit=false`のためtest prediction bundleがないのは設定どおり。
このrunだけでft-e13に対するrecallやp95の改善は主張しない。
Meiji注釈はChatGPT補助レビューによるもので、独立した人手の正解でもない。
単一seed・Meiji train 1 videoであり、他会場や3D下流への効果は未検証。

## 次の比較

選択済みepoch 0とft-e13を、同じstore・resize・入力正規化・窓・復号で比較する。
Meiji test video_001の63 camera-clip / 36,006 frameを末尾まで1度ずつ集計し、
20 source pxのrecall、予測欠損と大誤検出を除外しない母数、camera・point_kind・visibility・
手首距離の層別結果を残す。手首情報欠落は不明層とする。deployは比較完了まで維持する。
