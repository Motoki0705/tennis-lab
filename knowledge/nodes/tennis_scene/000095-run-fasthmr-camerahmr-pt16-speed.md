---
id: run-fasthmr-camerahmr-pt16-speed
type: run
task: tennis_scene
sequence: 95
recorded_at: '2026-11-25'
title: FastHMR CameraHMR PT16の固定24系列における参考速度とVRAM
issue: 2491
provider: codex
session: 01a20d45-575e-7a53-90c5-48b09053fa2b
date: '2026-11-25'
status: done
config:
  model: fasthmr-camerahmr
  backend: PT16
  data: meiji_4cam/person-model-speed-v1
  harness_commit: 13e508cb78c46663fbe58ad8be03bcf9c96816a6
  fixed_inputs_sha256: 64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62
  batch_sizes: [1, 2, 4]
  sequences: 24
  frames_per_sequence: 16
  feature_batch_size: 4
  diffusion_seed: 0
  warmup: 30
  blocks: 3
  repetitions_per_block: 100
  torch_host_threads: 2
  parity: not_evaluated
metrics:
  timed_repetitions: 900
  timed_person_frames: 33600
  batch1_wall_p50_ms: 1090.9225675
  batch1_model_gpu_p50_ms: 665.5379061698914
  batch1_wall_person_frames_per_second: 14.42127577029961
  sampled_device_peak_bytes: 5173776384
summary_impact: updated
repro:
  commit: 8a25fe12b543656cb1e883e7251bd1a3fd299fd0
  branch: codex/parent-2492-issue-2491-primary
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: sha256sum -c outputs/fasthmr-setup/speed-pt16/source.sha256 && bash outputs/fasthmr-setup/speed-pt16/run.sh
    /home/kamimura/projects/tennis-lab/.worktree-outputs/parent-2492-issue-2496-primary/person-model-speed-baseline-v1/inputs
artifacts:
  run_dir: knowledge/runs/run-fasthmr-camerahmr-pt16-speed
  timing_summary: knowledge/runs/run-fasthmr-camerahmr-pt16-speed/speed-summary.json
  raw_reports: .training_queue/repro/0-8204407397945499403_2026-11-25T07-43-22-054500596Z_1639663_i2491-fasthmr-camerahmr-pt16-speed/predictions
  log: .training_queue/repro/0-8204407397945499403_2026-11-25T07-43-22-054500596Z_1639663_i2491-fasthmr-camerahmr-pt16-speed/queue.log
  log_checksum: knowledge/runs/run-fasthmr-camerahmr-pt16-speed/queue-log.sha256.txt
parents: [run-person-model-speed-low-precision, run-fasthmr-camerahmr-pt16-review]
relations:
- to: run-person-model-speed-low-precision
  rel: compares
papers: []
tags: [fasthmr, body_recovery, inference_benchmark]
---

## 考察 / Findings

RTX 5060 Tiの排他queueで、16フレーム×24系列を使ったPT16の系列batch 1/2/4を各300反復した。各プロセスのウォームアップは30反復。固定入力、実際の反復順、計時の区切りは #2497 と一致する。PyTorch 2.13.0+cu130、CUDA 13.0、driver 621.24、180 Wの上限、TF32無効。FastHMRの実行HEADは `8a25fe12b543656cb1e883e7251bd1a3fd299fd0`。共通harnessは `13e508cb78c46663fbe58ad8be03bcf9c96816a6` で、#2497の最終headでは計時実装と入力プロトコルの対象ファイルに差分がない。 [基準の再計測](000086-run-person-model-speed-low-precision.md) [実行設定とレポートSHA-256](../../runs/run-fasthmr-camerahmr-pt16-speed/speed-summary.json)

GPUは各反復のモデルforwardのCUDA event時間の合計。wallはデコード済みBGR画像と固定BBoxから身体パラメータまでで、crop、転送、補助モデル、カメラ変換、HybrIK、native SMPLへの変換とバッチ終端の同期を含む。入力は実長16で、243フレームまでの窓に対するpaddingは0。GPU側のB=1のモデル別P50はCameraHMRの両向き特徴339.66 ms、HumanFoV 182.13 ms、YOLO11x-seg 89.32 ms、diffusion/VAEの両向き40.85 ms。各中央値はモデル別に集計した。 [条件別・段階別の実測](../../runs/run-fasthmr-camerahmr-pt16-speed/speed-summary.json)

| 方式 | 系列batch | CPUスレッド数 | GPU P50 / P95 ms | wall P50 / P95 ms | wall 人・フレーム毎秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HMR2特徴＋GVHMR PT16 | 1 | 4 | 140.60 / 144.34 | 492.28 / 558.10 | 31.82 |
| FastHMR CameraHMR PT16 | 1 | 2 | 665.54 / 836.33 | 1,090.92 / 1,291.98 | 14.42 |
| HMR2特徴＋GVHMR PT16 | 2 | 4 | 266.69 / 273.86 | 981.53 / 1,096.98 | 31.92 |
| FastHMR CameraHMR PT16 | 2 | 2 | 1,590.75 / 2,184.39 | 2,519.06 / 3,381.37 | 12.21 |
| HMR2特徴＋GVHMR PT16 | 4 | 4 | 514.03 / 519.55 | 1,924.92 / 1,983.49 | 33.16 |
| FastHMR CameraHMR PT16 | 4 | 2 | 1,947.84 / 2,459.11 | 3,230.67 / 3,993.48 | 19.20 |

FastHMRのwall P50は系列batch 1・2・4で現行PT16の2.22、2.57、1.68倍だった。B=1（16フレーム）のwall P50はFastHMR CameraHMR PT16が68.18 ms/人・フレーム、現行HMR2特徴＋GVHMRのPT16が30.77 ms/人・フレーム、TRT16が25.54 ms/人・フレーム。現行TRT16は保存CSVの408.717942 ms/系列から換算した。系列batchは1反復に投入した系列数。FastHMRは系列を1本ずつ処理し、特徴抽出を4 cropずつ実行した。現行経路はHMR2特徴を16 cropずつ抽出し、GVHMRで表の系列batchをまとめて推論した。PyTorchのCPUスレッド数と内部batchが異なるため、この実行条件での参考値とする。計時範囲外の動画decode、配置以降と全pipelineの所要時間は未計測。比較図の対象外。 [現行PT16・TRT16の全batch](../../runs/run-person-model-speed-low-precision/cases.csv)

FastHMRの3条件でPyTorch allocated peak 4,299.8 MiB、reserved peak 4,466.0 MiB、NVMLデバイス使用量のサンプル最大値4,934.1 MiBだった。B=1の現行PT16のNVMLは4,926.1 MiB。FastHMRの追跡対象資産は合計4,695,003,483 bytes。ロードと初回推論はそれぞれ別に保存した。B=1の最初の呼び出しにYOLOのNMS時間上限警告が1件あり、ウォームアップ後の反復は全条件で完了した。 [環境・メモリ・資産のハッシュ](../../runs/run-fasthmr-camerahmr-pt16-speed/speed-summary.json)

PT16のparityは2026-11-25のユーザー判断により明示的に未判定。CameraHMRのTRT16は書き出しで停止し、今回の試行では実行不能だった。HMR 2.0ベースは公開版のMask-ToMe非対応。PT32の8フレーム試行は既存記録に保持する。 [保存済み初回試行](000092-run-fasthmr-camerahmr-smoke.md) [TRT16の試行](000093-run-fasthmr-camerahmr-trt-export-probe.md) [代表6 clipの目視資料](000094-run-fasthmr-camerahmr-pt16-review.md)

2026-11-26、利用者は「FastHMRの速度優位性がないなら置き換えは見送りです。」と判断した。実測した系列batchのすべてでFastHMR PT16は現行PT16より時間を要したため、置き換えを見送った。目視評価は未実施（速度条件で見送りが決定）。標準recipeはGVHMRを使い、`recipe=fasthmr` は明示選択の経路としてmainに残す。 [ユーザー判断の記録](https://github.com/Motoki0705/tennis-lab/issues/2494#issuecomment-7178978237)

`repro.sh` は `TENNIS_REPO` に実行HEADのworktree、`PERSON_SPEED_INPUT_DIR` に固定入力を渡し、共有GPUキュー経由で実行する。元ジョブの生JSONは `.training_queue/repro/<job>/predictions/` に保持した。`verify_speed.py` はその3レポートの順序と統計を再計算し、`speed-summary.json` と照合できる。 [元の投入コマンド](../../runs/run-fasthmr-camerahmr-pt16-speed/run.json)
