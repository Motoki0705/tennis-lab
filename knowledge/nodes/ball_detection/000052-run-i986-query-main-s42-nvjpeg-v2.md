---
id: run-i986-query-main-s42-nvjpeg-v2
type: run
task: ball_detection
sequence: 52
recorded_at: '2026-10-09'
title: MDD query-only本学習は約4.3万更新で精度停滞し、ユーザー指示で中断
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-09'
status: failed
config:
  model: conv2d-query_only
  dim: 128
  heads: 4
  layers: 2
  precision: bf16
  seed: 42
  batch_size: 1
  learning_rate: 0.0001
  planned_updates: 60000
  jpeg_decoder: nvjpeg
  image_prefetch: true
  selection_scope: common
metrics:
  last_logged_step: 42950
  last_checkpoint_step: 42000
  completed_validations: 7
  best_common_mean_error_px: 230.89350917490336
  selected_full_mean_error_px: 237.8172523836538
  last_full_mean_error_px: 237.40225790668592
  last_common_mean_error_px: 232.4804872352681
repro:
  commit: e75f81a4e3831af6f0ca175f71c0ccb90fd2b021
  branch: HEAD
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
    TORCH_LOGS=graph_breaks,recompiles PYTHONUNBUFFERED=1 .venv/bin/python -m src.tasks.ball_detection.scripts.train_mdd_pose
    --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --model-config /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/models/conv2d-query_only.yaml
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/conv2d-query-only-bf16/s42-u60000-nvjpeg-v2
    --device cuda --precision bf16 --compile-mode default --batch-size 1 --num-workers
    8 --pin-memory --prefetch-factor 4 --cpu-threads 2 --jpeg-decoder nvjpeg --input-verification
    upfront --image-prefetch --epochs 10 --windows-per-epoch 6000 --learning-rate
    0.0001 --seed 42 --mdd-a 0.2 --mdd-b 0.15 --selection-scope common --log-every
    50
artifacts:
  run_dir: knowledge/runs/run-i986-query-main-s42-nvjpeg-v2
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791513534641157109_1753998_i986-query-bf16-s42-u60000-nvjpeg-v2.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/conv2d-query-only-bf16/s42-u60000-nvjpeg-v2
parents:
- run-i986-fenced-startup-cli-20261009
relations: []
papers: []
tags:
- main-training
- user-cancelled
- accuracy-plateau
- query-only
---

ユーザーの「学習は中断して」に従い、2026-10-09 14:14 JSTまでにqueue cancelで停止。queueはcancelled、exit codeは143、対象PGIDの消滅を確認した。knowledgeのstatusにcancelledがないため未完了runをfailedとして記録するが、今回の終了はCUDA異常ではなく明示的な中断である。最終進捗ログ42,950更新は50更新おきの下限で、厳密な最終実行stepではない。epoch-006.ptは42,000更新、選択済みbestのepoch-005.ptは36,000更新。進行中epochの重みは保存されていない。

| 更新数 | train SmoothL1 | full mean px | common mean px |
|---:|---:|---:|---:|
| 6000 | 0.102654 | 239.395 | 231.222 |
| 12000 | 0.100099 | 237.930 | 232.739 |
| 18000 | 0.099967 | 237.495 | 230.951 |
| 24000 | 0.100006 | 239.467 | 231.216 |
| 30000 | 0.099733 | 237.355 | 230.962 |
| 36000 | 0.098723 | 237.817 | 230.894 |
| 42000 | 0.099426 | 237.402 | 232.480 |

validationは位置教師有効frameのsource-pixelユークリッド距離をFPSごとに重複排除し、元FPS/1/2/1/4の平均誤差を等重み集計する。train lossは正規化座標のSmoothL1なので、両者を直接比較しない。選択された36,000更新のfull meanは237.817px、各FPSの中央値は約209px、P95は約528px。初回からの改善はfullで約0.66%、commonで約0.14%にとどまり、最後の42,000更新でも約237pxの停滞が続いた。test評価は未実施。旧モデルとGT/split/出力契約が異なるので直接精度比較をしない。

dim128・2層、細いCNNでの容量不足は仮説であり、本runだけでは原因を断定できない。固定位置への偏り、入力への感度、座標対応の検証を先に行う価値がある。ユーザーはCNNの深層化とTransformer dim拡大の案を要求した。次候補は残差CNN＋dim256・4層とし、同じseed/split/窓sampling/update budgetを明示して比較する。深さ・幅の拡張による精度やVRAMは未確認で、提案段階。学習の再開はしていない。

数値ログ・設定・best選択をbundleに保存した。TensorBoard・test予測・GIFはこのrunにない。中断によるepoch未完了と教師の存在するframeのみの評価という制約がある。CPU供給の改善を精度改善とは扱わず、既存deployの判断は変更しない。

再現用YAMLをbundleへ保存し、launch receiptのSHA-256一致を確認した。repro.shの設定参照先のみbundle内へ移し、run.jsonの原コマンドと設定内容は保持した。
