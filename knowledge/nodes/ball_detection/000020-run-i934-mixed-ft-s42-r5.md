---
id: run-i934-mixed-ft-s42-r5
type: run
task: ball_detection
sequence: 20
recorded_at: '2026-09-28'
title: Meiji混合FT初回はepoch終端のMDD描画で正規化契約が欠落して停止
issue: 934
provider: codex
session: 01a0e592-3e39-74f1-816b-fffe8fead5fc
date: '2026-09-28'
status: failed
config:
  profile: train_meiji_mixed
  model: conv_next_unet
  input_mode: mdd
  loss: focal_bce
  data: ball-mix-v1
  source_weights: {tracknet: 1.0, meiji: 1.0, chat_annotation: 1.0}
  seed: 42
  max_epochs: 12
  windows_per_epoch: 7680
  batch_size: 4
  learning_rate: 2.0e-5
  checkpoint_monitor: val/f1
  test_after_fit: false
metrics: {}
repro:
  commit: ae5f5baaa359ad608b973d0d9b09e8cfaae3a17f
  branch: campaign930/i934-4-mixed-ft
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i934-4-mixed-ft/.venv/bin/python
    -m src.tasks.ball_detection.scripts.train --config-name train_meiji_mixed paths.project_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i934-4-mixed-ft
    paths.data_root=/home/kamimura/projects/tennis-lab/data paths.checkpoint_root=/home/kamimura/projects/tennis-lab/ckpt
    paths.output_root=/home/kamimura/projects/tennis-lab/outputs paths.artifact_root=/home/kamimura/projects/tennis-lab/outputs
    paths.cache_root=/home/kamimura/projects/tennis-lab/.cache paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
    run.output_dir=ball_detection/train/i934_mixed_ft/s42-r5-20260928
artifacts:
  run_dir: knowledge/runs/run-i934-mixed-ft-s42-r5
  log: knowledge/runs/run-i934-mixed-ft-s42-r5/queue.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/i934_mixed_ft/s42-r5-20260928/logs/version_0
  curves: knowledge/runs/run-i934-mixed-ft-s42-r5/curves.png
  tb_logdir: knowledge/runs/run-i934-mixed-ft-s42-r5
  config: knowledge/runs/run-i934-mixed-ft-s42-r5/config.yaml
  event_summary: knowledge/runs/run-i934-mixed-ft-s42-r5/event_summary.json
  fix_cpu_qualification: knowledge/runs/run-i934-mixed-ft-s42-r5/fix_cpu_qualification.json
  fix_cpu_artifact_root: /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i934-4-mixed-ft/outputs/campaign_930/i934/run6/real-data-pytest/test_ft_e13_renders_all_three_0
parents: [run-i618-convnext-v2-ft]
relations: []
papers: []
tags: [campaign930, mixed-ft, failed, normalization, qualitative]
---

## 結果と失敗原因

共有queue job `1790558974032900448_1609437_i934-mixed-ft-s42-r5-20260928` は
exit code 1で終了した。ft-e13の242 tensorのstrict読込、3 sourceのtrain/val窓構築と
学習は開始できたが、最初のepochのvalidation後にqualitative callbackで停止した。
[queue log](../../runs/run-i934-mixed-ft-s42-r5/queue.log)のtraceは
`render_qualitative_samples → build_render_animation_inputs → build_mdd_frames_from_images
→ mdd_features → prepare_images`を示す。

datasetはImageNet正規化済みのRGBを渡し、学習forwardはその宣言を持っていた。
しかしMDD描画helperは正規化情報を渡さずraw RGB用の境界を呼び出したため、
`BallModelIOError: images values must be in [0, 1].` が発生した。
CPUの小規模な3 source storeと実際のrunner/callbackでも同じ例外を再現できた。
入力のクリップや値域検証の緩和ではなく、描画にも学習と同じ正規化情報と
`preprocessed=True`を明示する修正が必要だった。

## 観測できた範囲

[保存したTensorBoard eventの要約](../../runs/run-i934-mixed-ft-s42-r5/event_summary.json)には
epoch 0のtrain/lossが38点あり、最後はstep 1899の`0.0005668875528499484`。
これは最後に記録されたbatchの値で、epoch平均でもvalidation性能でもない。
[曲線](../../runs/run-i934-mixed-ft-s42-r5/curves.png)は途中のtrain lossだけを示す。
validation指標、完了epochのcheckpoint、test予測は保存されていない。
`hp_metric=-1`はlogger初期値であり、評価結果として扱わない。
したがって`metrics`は空とし、混合FTの有効性・ft-e13からの改善は判断しない。

## 修正の検証と次の実験

`test_qualitative_training.py`は修正前に同じ例外で失敗し、修正後は
3 sourceの小規模CPU学習からepoch 0のGIF・val/f1選択checkpoint・last.ckpt保存まで通った。
正規化有効/無効・独自mean/std・sample選択をテストし、MDDがmodel入力と一致すること、
画像を二重に正規化しないこと、宣言のない正規化済み入力や値域違反を拒否することを確認した。
`test_qualitative_local_data.py`では実ft-e13・288×512・8 frameを使い、
TrackNet/Meiji/chat_annotationのvalidation各1窓で同じ描画経路をCPU検証した。
[修正後のCPU検証記録](../../runs/run-i934-mixed-ft-s42-r5/fix_cpu_qualification.json)に
window ID、入力値域、MDD一致、GIF frame数、checkpoint/storeのhashを残す。
JSONのartifact相対pathは`fix_cpu_artifact_root`を基準とし、GIF実体はgit管理外の同directoryに保持する。
これらは描画・学習制御の検証であり、GPUで12 epochを完走した証拠ではない。

再開用checkpointがないため、修正後のcodeで同じft-e13、seed、source比率、学習予算を使い
別runとして最初から再実行する。qualitative loggingは有効のまま保持する。
学習完了後にvalidation F1最大のcheckpointを選び、Meiji video_001の全frameを
ft-e13と共通条件で比較する。deployは変更しない。
