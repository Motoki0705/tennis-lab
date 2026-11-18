---
id: run-i2253-ball-model-compare-20261118
type: run
task: tennis_scene
sequence: 66
recorded_at: '2026-11-18'
title: meiji_4cam 3クリップの固定カメラ・人物でボール検出5モデルを比較
issue: 2253
provider: codex
date: '2026-11-18'
status: done
summary_impact: none
config:
  dataset: meiji_4cam
  video: video_000
  clips: [clip_000, clip_027, clip_037]
  camera_ids: [cam0, cam1, cam2, cam3]
  court_side_half_turns: [false, false, true, true]
  pipeline_base: d392dd375776de72f604aca7be0bbb81025b7b2c
  gpu: NVIDIA GeForce RTX 5060 Ti
  gpu_queue_reservation: all
  ground_truth_camera_clips: 0
metrics:
  completed_models: 5
  completed_clips: 3
  npz_files: 15
  source_frames_per_model: 883
  source_camera_frames_per_model: 3532
  reused_nodes_per_clip: 28
  queue_wall_seconds: 816.253454
  valid_3d_frames: {pipeline_x3d_dpt: 410, i935-convnext: 426, convnext-dpt-t64: 350, convnext-query-t64: 595, x3d-query-d512-l8: 609}
  outside_court_frames: {pipeline_x3d_dpt: 29, i935-convnext: 0, convnext-dpt-t64: 0, convnext-query-t64: 13, x3d-query-d512-l8: 111}
  baseline_max_raw_uv_delta_px_cpu_decode: 0.0001220703125
  baseline_exact_raw_uv_frames_vs_gpu_rank1: 3197
repro:
  commit: e40b041cfc616c6ae6735704f881876e35f00275
  command: >-
    env PYTHONPATH=. PYTHONHASHSEED=42 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
    .venv/bin/python -B -u knowledge/runs/run-i2253-ball-model-compare-20261118/compare.py run
    --root outputs/tennis_scene/evaluate/ball-model-compare-20261118 --expect-commit e40b041cfc616c6ae6735704f881876e35f00275
artifacts:
  run_dir: knowledge/runs/run-i2253-ball-model-compare-20261118
  output_dir: outputs/tennis_scene/evaluate/ball-model-compare-20261118
  result_summary: knowledge/runs/run-i2253-ball-model-compare-20261118/summary.csv
  output_audit: knowledge/runs/run-i2253-ball-model-compare-20261118/output-audit.json
parents:
- run-i1752-shared-fp16-20261118
relations:
- to: run-i1752-shared-fp16-reference-20261118
  rel: compares
  note: "#1752のbatch保存結果に対し、今回は同じ3クリップの再推論を照合した"
- to: run-ball-v3-migration-20261113
  rel: related
  note: 4カメラの対象動画に対応する2D正解があるかを移行の対応表で調べた
papers: []
tags: [ball-detection, fixed-geometry, meiji]
---

## モデル選定

固定validationの指標は元画像px。現行と旧既定を両方使い、時系列長・検出head・学習構成が異なる3重みを加えた。保持された実行契約、重みのSHA-256、Driveの世代は [models.json](../../runs/run-i2253-ball-model-compare-20261118/models.json) に記録した。

| モデル | 入力 H×W、T、精度 | 存在閾値 | 既存評価と選定理由 |
|---|---|---:|---|
| `pipeline_x3d_dpt` | 720×1280、32、BF16 | 0.53801244 | 共通validationの平均34.43 px、20 px以内86.14%。#1752の基準。[000070](../ball_detection/000070-run-ball-capacity-common-dpt-s42-20261103.md) |
| `i935-convnext` | 288×512、8、FP32 | 0.5 | chat annotation validationのtop-1 recall 83.92%（観測6,622）。旧既定の短窓MDD。[000015](../ball_refiner/000015-run-i935-evidence-mixed-e9-trainval-r17-20260930.md) |
| `convnext-dpt-t64` | 720×1280、64、BF16 | 0.57128835 | 共通validationの平均36.56 px、20 px以内86.58%。長窓のheatmap。[000068](../ball_detection/000068-run-ball-sequential-length-20261025.md) |
| `convnext-query-t64` | 720×1280、64、BF16 | — | 共通validationの平均28.53 px、20 px以内76.17%。#1899の変更3条件で最小平均誤差の単一query。[000093](../ball_detection/000093-run-i1899-convnext-1query-t64-t64-s42-12k.md) |
| `x3d-query-d512-l8` | 720×1280、64、BF16 | — | 共通validationの平均32.48 px、20 px以内75.59%。#1675の7容量条件で最小平均誤差の4-query mHC。[000087](../ball_detection/000087-run-i1675-capacity-d512-l8-t64-s42-12k.md) |

旧既定のrecallは異なるvalidationの指標。#1899の旧基準（T32→T64）の平均28.20 pxは当時の比較に残し、今回はT64→T64を単一queryの代表にした。#1859のT1は6k中間モデル（平均43.17 px）で、学習時のCUDA拡張MSDAを追加準備する必要があるため見送った。v3移行の記録はラベルの出自確認に使った。[単一queryの対照表](../ball_detection/000094-group-i1899-controlled-query-ablation-s42.md) · [T1の実測](../ball_detection/000091-run-i1859-t1-front-t64-s42-6k.md) · [v3移行](../ball_detection/000095-run-ball-v3-migration-20261113.md)

## 入力と比較の範囲

動画12本のSHA-256を#1752の固定コピーと現在のdatasetで照合した。保存済みのコート、side、校正、カメラ整合、人物関連をread-onlyで復元した。2つの基準は当該worktreeのscene経路とTorch 2.13.0+cu130、追加3モデルは学習時ソースとTorch 2.14.1+cu130で検出する。5モデルとも同一worktreeのball_points、ball_triangulation、scene_assemblyで再生する。

3クリップの登録ディレクトリにはmediaとclip metadataがあり、正解ラベルに対応するファイルは0件だった。v3移行の123 camera-clipは `meiji_3cam` の3収録で、同名クリップも元動画・開始frame・長さが異なる。比較値は今回のマスク、棄却、再投影の記述統計とする。学習validationの指標と今回の定性的な比較ではデータと指標が異なる。[v3対応表](../../runs/run-ball-v3-migration-20261113/correspondence.csv)

## 基準の再現性

保存済みのgate後観測を使うCPU replayで、3クリップのball_pointsからsceneまで完全一致を確認した。共有sideは `[false, false, true, true]`。GPU再推論では全3,532カメラframeのrawのscore、gate後の観測、平滑化の値・マスク、球三角測量の位置・有効性・棄却理由・inlier・再投影、組み立てたsceneのボール配列が一致した。

保存済みheatmapをCPUでsubpixel補正したraw座標との差は最大0.0001220703125 px。同じnative grid cellを指す保存済みGPUのrank-1候補がある3,197 frameではraw座標が完全一致した。CPU上のsubpixel再計算で生じた丸め差と整合する。今回参照と比較した検出証拠は選ばれた窓のtop-1座標とscore。 [差分](../../runs/run-i2253-ball-model-compare-20261118/baseline-parity.json) · [保存GPU候補との照合](../../runs/run-i2253-ball-model-compare-20261118/baseline-gpu-candidate-audit.json)

## 実データの結果

5モデル×3クリップ・各4カメラの検出と3D再構成が完了した。3D有効率は883フレームを母数とする。範囲外は有効点のうち `|x| > 9 m` または `|y| > 18 m` の点。再投影の中央値には、同じモデルが受理した平滑化済み2D観測と有効な3D投影を全クリップから集めた。

| モデル | 3D有効フレーム | 範囲外 | 再投影中央値 |
|---|---:|---:|---:|
| `pipeline_x3d_dpt` | 410 / 883（46.4%） | 29 | 4.59 px |
| `i935-convnext` | 426 / 883（48.2%） | 0 | 3.70 px |
| `convnext-dpt-t64` | 350 / 883（39.6%） | 0 | 5.58 px |
| `convnext-query-t64` | 595 / 883（67.4%） | 13 | 10.64 px |
| `x3d-query-d512-l8` | 609 / 883（69.0%） | 111 | 9.00 px |

この3クリップでi935は範囲外0件かつ再投影中央値が最小だった。直接座標の2モデルでは3D有効フレームが増え、特に4-query mHCはclip_037だけで範囲外71件が残った。直接座標モデルは全frameに提案があるため、`raw_score` はNaN、trajectory gateと後段へ与える提案の重みは1とした。再投影は各モデルの2D観測と固定カメラから導いた残差であり、正解照合0件のため比較結果から採用判断は保留する。 [クリップ・カメラ別集計](../../runs/run-i2253-ball-model-compare-20261118/summary.csv)

## 検証と保存

CPUの関連テスト59件が12.75秒で通過し、GPU queueの1ジョブは816.25秒でexit 0。15個のNPZを読み戻し、配列のshape・dtype・欠測、CSVとの一致、checkpoint・media・ソースindexのSHA-256を照合した。各モデルで各クリップの同じ28ノードをロードし、sceneの人物とコートに関する配列・カメラのメタデータの一致を確認した。出力ディレクトリには `manifest.json`、`<model>/<clip>.npz`、`summary.csv` と証跡を保存した。

再現コマンド、実行計測、ソースの参照記録、出力ハッシュを以下に保持する。 [再現手順](../../runs/run-i2253-ball-model-compare-20261118/reproduce.md) · [実行とラベル確認](../../runs/run-i2253-ball-model-compare-20261118/execution.json) · [loadの証跡](../../runs/run-i2253-ball-model-compare-20261118/loaded-stages.json) · [出力の照合](../../runs/run-i2253-ball-model-compare-20261118/output-audit.json)
