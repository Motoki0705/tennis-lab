---
id: run-i1031-court-ckpt-qualitative-20261008b
type: run
task: court_detection
sequence: 35
recorded_at: '2026-10-08'
title: Court pose学習済みcheckpointから4 dense headのPNGを保存
issue: 1031
provider: codex
session: 01a113c9-1114-71a2-9261-80815f662d33
date: '2026-10-08'
status: done
config:
  checkpoint: ckpt/court_detection/multiscale_depth3/b863df1f01f0.ckpt
  checkpoint_sha256: b863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383
  checkpoint_epoch: 17
  validation_subset_indices:
  - 0
  - 935
  mode: validation_only
  qualitative_interval: 1
metrics:
  validation_samples: 2
  png_files: 8
  tensorboard_image_tags: 8
  state_tensors_strict_loaded: 593
  peak_cuda_allocated_mib: 1137.19091796875
repro:
  commit: 1dcf9407897cefc38842fe02e57e6895c008e6f6
  branch: fix/issue-1031-qualitative-schedule
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env MPLBACKEND=Agg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m scripts.verify_qualitative_checkpoints
    --task court_detection --asset-root /home/kamimura/projects/tennis-lab --checkpoint
    /home/kamimura/projects/tennis-lab/ckpt/court_detection/multiscale_depth3/b863df1f01f0.ckpt
    --output-dir /home/kamimura/projects/tennis-lab/outputs/court_detection/evaluate/issue-1031-qualitative/20261008b
    --sample-indices 0 935
artifacts:
  run_dir: knowledge/runs/run-i1031-court-ckpt-qualitative-20261008b
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791417281600875658_74343_i1031-court-ckpt-qualitative-20261008b.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/court_detection/evaluate/issue-1031-qualitative/20261008b
  verification: knowledge/runs/run-i1031-court-ckpt-qualitative-20261008b/verification.json
  config: knowledge/runs/run-i1031-court-ckpt-qualitative-20261008b/config.yaml
  qualitative: assets/court_detection/issue-1031-checkpoint-validation
  tb_logdir: /home/kamimura/projects/tennis-lab/outputs/court_detection/evaluate/issue-1031-qualitative/20261008b/logs/version_0
parents: []
relations:
- to: run-i1031-court-ckpt-qualitative-20261008a
  rel: resolves
papers: []
tags:
- qualitative-logging
- real-checkpoint
- cuda
- pr-1041
---

epoch 17・global step 29844の既存pose＋4 dense head checkpointを593 tensorすべてstrict loadし、共通QualitativeLoggingCallbackを付けたTrainer.validateをRTX 5060 Tiで実行した。合成B00:court-sample-000448と実画像EF-hx40Q4Mg_700の2 validation sampleについて、KP／SEG／LINE／semantic LINEのPNGを各1枚、計8枚保存した。pose画像は生成していない。PNGの全decode、TensorBoardの画像8 tag、callbackのlast_logged_epoch=1を確認した。検証前後でcheckpointのSHAと全state_dict tensorが一致し、重み更新はない。

保存済みmodel・loss・4 headのtarget bundleと旧hard target schema、validation解像度を維持した。旧derived_target_rootを取り除き、DINOv3資産pathを現配置へ移し、学習用train_scalesだけを現行pose-safe契約へ合わせた。これらは検証runtimeの明示変換で、checkpoint本体は変更しない。設定選択モードを共通fixed_indicesへ統一する不具合修正と74件のCPU回帰テストを行った後の実行である。

実画像では予測点・コート領域・線・線種マップを確認できた。合成の低い斜視sampleでは点が画面端へ集まり、線種の分断も見える。保存経路が成立した証拠であり、2画像の定性確認やreport内のsubset metricから全datasetの精度・pose品質・checkpointの優劣を判断しない。既存モデル採用判断は変更しない。

保存物はartifacts.qualitative、条件・SHA・部分検証値はverification.jsonを正本とする。standalone validationの保存周期は1に設定し、疎なvalidationとresumeの周期自体は既存CPU回帰テストが担う。TensorBoardはこの1回のvalidation値と画像のみで、学習・収束曲線の作成は該当しない。次の機能確認は別checkpointや別sourceでも同じ保存契約を確認すること、精度判断には独立した評価subsetを用いることとする。
