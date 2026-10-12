---
id: run-i986-native-rgb-numerics-20261009
type: run
task: ball_detection
sequence: 30
recorded_at: '2026-10-09'
title: native RGB MDDとBF16 compileの数値・勾配整合
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-09'
status: done
config:
  model: conv2d-query_only
  precision: bf16
  mdd_precision: float32
  dropout: 0
  frame_steps:
  - 1
  - 2
  - 4
metrics:
  max_mdd_abs_difference: 3.3676624298095703e-06
  gradient_relative_l2: 0.014985211193561554
  max_uv_delta: 0.0016404390335083008
repro:
  commit: 524cef6b95645881ed04174639b6b5a71f4e6a77
  branch: codex/ball-query-only-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 TORCH_LOGS=graph_breaks,recompiles
    PYTHONUNBUFFERED=1 timeout -k 20s 1800s .venv/bin/python tests/benchmarks/ball_native_rgb_correctness.py --manifest
    /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --model-config /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/models/conv2d-query_only.yaml
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/native-rgb-compile/20261009-correctness-v1/report.json
artifacts:
  run_dir: knowledge/runs/run-i986-native-rgb-numerics-20261009
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791474021867879284_825384_i986-native-rgb-correctness-v1.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/native-rgb-compile/20261009-correctness-v1
parents:
- run-i986-native-rgb-compile-20261009
relations: []
papers: []
tags:
- native-rgb
- bf16
- torch-compile
- runtime-diagnostic
---

## 結論

実RGBから作るモデル内MDDは、旧CPU式に対して所定の誤差許容を満たした。
BF16 eagerとcompileの座標・損失・勾配も許容範囲内で、checkpointのstate_dict名は不変だった。
bit一致や精度同等性を意味する検証ではない。

| 検査 | 実測 | 実行前に固定した許容 |
|---|---:|---:|
| 旧NumPy MDD対GPU compiled MDDの最大絶対差 | 3.37e−6 | atol=3e−6＋rtol=1e−5 |
| 勾配全体の相対L2差 | 0.01499 | 0.05 |
| 正規化uvの最大絶対差 | 0.001640 | 0.003 |
| 損失の絶対差 | 0.000573 | 0.002 |
| state_dict名 | 一致 | 同一 |

MDDはchat_annotation / meiji / tracknetの3 source×frame step 1/2/4、計9窓を旧BGR/NumPy式と比較した。
精度はMDD FP32、モデルBF16。学習可能なRGB経路は追加せず、先頭MDDはzeroのまま保持した。
GPU eager MDDの最大差は2.63e−6未満だった。compileによる融合で丸め順序が変わるため、bit一致は要求しない。
勾配・uv比較は1窓、dropout=0の診断。通常学習のdropout=0.1は変更していない。
全parameterの勾配有限性を確認し、同じ初期stateを別モデルへ復元したevalでもuv差は許容内だった。
3 graph（独立MDD moduleを含む）、graph break 0。元データのtest評価は行っていない。

## 限界・次の確認

異なるsourceの全clipや36構成すべてをGPUで比較した実験ではない。
uv差0.00164は正規化単位で、学習済みモデルのpixel精度を保証する値ではない。
定常速度と実CLI保存・再開は別ノードで検証し、本学習の収束は未評価のまま残す。

## 証拠と再現

[report.json](../../runs/run-i986-native-rgb-numerics-20261009/report.json)に窓identity、全差分、許容誤差、compile設定を保存。
TensorBoardは使用していない。前提は保存commit＋patch、GPU実行は共有queueを使う。
repro.shのYAML参照だけを同一内容の`model.yaml`へ修復し、元スクリプトと修復理由をbundleへ保持した。
