---
id: run-i936-synthetic-smoke-v2-s936
type: run
task: ball_refiner_3d
sequence: 3
recorded_at: '2026-09-29'
title: 59.94fps合成smoke v2でも残る全成分Laplaceの正depth制限
issue: 936
provider: codex
date: '2026-09-29'
status: failed
config:
  schema_version: 1
  status: cpu_generator_v1
  purpose: '#936 CPU synthetic dataset; no real ball annotations are read'
  seed: 936
  simulation:
    class: src.tasks.blcs.generate_dataset.simulation.rally_simulator.RallySimulator
    device: cpu
    workers: 4
    maximum_physics_attempts_per_rally: 32
    physics_base: src/tasks/blcs/configs/physics/default.yaml
    rally_base: src/tasks/blcs/configs/rally/default.yaml
    targeted_velocity_base: src/tasks/blcs/configs/targeted_velocity/default.yaml
    sim_fps: 240
    native_output_fps: 240
    physics_dt_seconds: 1/240
  sampling:
    fps_numerator: 60000
    fps_denominator: 1001
    interpolation: linear at exact output timestamps
    event_index: nearest output timestamp; also preserve native time in seconds
    max_frames_per_rally: 512
    physics_event_mask_radius_frames: 5
  geometry:
    sources:
    - split: train
      clip_id: video_002/clip_010
      path: /home/kamimura/projects/tennis-lab/data/blcs/meiji_geometry_replay_v1/scenes/meiji_video_002_clip_010_002/scalars.json
      sha256: 9d2488976a168697807e457819ea8801e6dfe1e0035717c023ba740859d51619
      camera_keys:
      - cam_0_params
      - cam_1_params
      - cam_2_params
    - split: val
      clip_id: video_000/clip_003
      path: /home/kamimura/projects/tennis-lab/data/blcs/meiji_geometry_replay_v1/scenes/meiji_video_000_clip_003_000/scalars.json
      sha256: 3ec27b1ef446fd48620cbd6dcab7e87f43082deca24d473525fcc3292dfc1d1c
      camera_keys:
      - cam_0_params
      - cam_1_params
      - cam_2_params
    - split: test
      clip_id: video_001/clip_018
      path: /home/kamimura/projects/tennis-lab/data/blcs/meiji_geometry_replay_v1/scenes/meiji_video_001_clip_018_000/scalars.json
      sha256: 4ca0bc2747ea4dbacbea3b1751d93e2b81051bc7233d52f385578c85d57e0b02
      camera_keys:
      - cam_0_params
      - cam_1_params
      - cam_2_params
    benchmark_used_only: video_002/clip_010
    perturbation_per_scene:
      center_sigma_m: 0.15
      axis_angle_sigma_deg: 0.5
      focal_relative_sigma: 0.01
      principal_point_sigma_px: 2.0
    calibration_error_condition: clean_shared_perturbed_camera
  degradation:
    status: 'hypothesized parameters; recalibrate against #935 train/val before production'
    distribution_class: src.tasks.ball_refiner.refiner_2d.distribution.BallGMM2D
    components_per_camera: 3
    source_pixel_sigma_range:
    - 3
    - 24
    correlation_range:
    - -0.7
    - 0.7
    error_ar1: 0.9
    mean_error_sigma_multiplier: 0.25
    gap_lengths_frames:
    - 1
    - 4
    - 8
    - 16
    - 32
    - 64
    occlusion_changes_presence: false
    out_of_frame_changes_presence: true
    alternatives: persistent shifted trajectory plus broad distractor; no raw detector
      points
    alternative_shift_m:
    - 1.2
    - 2.4
    - 0.4
    observed_weights:
    - 0.7
    - 0.2
    - 0.1
    gap_weights:
    - 0.4
    - 0.4
    - 0.2
    gap_sigma_multiplier: 4.0
    distractor_sigma_multiplier: 3.0
    presence_logit_magnitude: 4.0
    mean_bounds_policy: clip_to_source_grid
    triangulation: src.utils.geometry.probabilistic_triangulation.triangulate_gmm
    prior_mean_m:
    - 0.0
    - 0.0
    - 2.0
    prior_covariance_diagonal_m2:
    - 36.0
    - 144.0
    - 9.0
    max_components: 64
    max_nfev: 100
  counts:
    smoke_rallies_per_split: 4
    pilot_rallies:
      train: 512
      val: 64
      test: 64
  storage:
    format: float32 NPZ per rally plus JSON manifest; all mixture components preserved
    estimated_pilot_limit_bytes: 1500000000
    fields:
    - timestamps_seconds
    - positions_3d_m
    - events_seconds_and_frame
    - occlusion_mask
    - out_of_frame_mask
    - camera_K_R_t
    - source_size_wh
    - 2d_means_covariance_weights_presence
    - 3d_means_covariance_weights_camera_subsets
    - prior_only_probability
    - seed_config_and_source_hashes
  acceptance_checks:
  - exact 60000/1001Hz timestamps without integer-stride drift
  - event seconds retained and masks cover interpolation boundaries
  - no recording/window crosses splits
  - load BallGMM2D then pixel_moments and triangulate; raw observations are not an
    alternate path
  - all finite SPD distributions including long gaps
  - persist full covariance and between-mode uncertainty
  - report frame counts and bytes per split; generation failures are errors
metrics:
  elapsed_seconds: 139.43338676699204
  requested_rallies: 12
  completed_rallies: 1
  recorded_failures: 10
  output_bytes: 1414903
artifacts:
  run_dir: knowledge/runs/run-i936-synthetic-smoke-v2-s936
  output_dir: /home/kamimura/projects/tennis-lab/data/ball_refiner/synthetic-3d-i936-smoke-r2-v2
parents:
- run-i936-synthetic-smoke-v1-s936
relations: []
papers: []
tags: []
repro:
  commit: 8b8a50aa1e92c50f8138e4ec4cd271add6d8f440
  branch: campaign930/i936-2-synthetic-diffusion
  command: bash knowledge/runs/run-i936-synthetic-smoke-v2-s936/repro.sh /absolute/new/output
session: 01a0ed26-b4eb-7c30-baea-c8a3cb41cf9e
---


## 結論

12-rally smokeは再度失敗した。完了1件、失敗JSON10件、未完了1件。
親processの経過時間は139.43秒、保存した出力は合計1,414,903 bytes。
狭い既知priorでの方式Aの比較結果を、広いcourt prior・K=3の長系列へ
そのまま外挿できない。今回のgeneratorはまだ学習datasetを提供できない。

## 変更した条件

設定・展開済みsimulator・入力SHAの正本は
[manifest](../../runs/run-i936-synthetic-smoke-v2-s936/manifest.json)。
前runから、float64での画素変換、既知物理棄却だけ最大32回のrejection sampling、
元のnative上限12000frameでのsimulationを採用した。
mean noiseを3–24px scaleの0.25倍へ限定し、gap/distractorの予測共分散拡大とは分離した。
これは仮定したmemory smoke条件であり、#935の実出力から測った値ではない。

共分散非対称エラーは再発しなかった。一方、背後MAPが9件、max_nfev=100の枯渇が1件。
**平均誤差を小さくするだけでは解決しなかった**。このscale変更を改善策や
本学習の採用条件として確定しない。失敗した成分の削除、behind errorの解除、
成功seedの手動選別は行っていない。

## 成功した1ラリーの検査

val-00001（video_000/clip_003の校正）はnative4273→512frame。
全512frameに64成分を保存し、打球6・bounce5、共有gap16frameを含む。
元native座標からの60000/1001Hz再標本化、イベント秒/最近傍frame/±5 mask、
全subsetの質量とpresence、float32共分散のSPD、true/estimated camera一致を
readerの検証関数で再確認した。

| 実測 | 値 |
|---|---:|
| simulation | 3.001秒 |
| triangulation | 103.147秒 |
| rally全体 | 106.266秒 |
| NPZ | 1,395,322 bytes |
| physics提案 | 1、最初のseedを採用 |

[ラリーmetadata](../../runs/run-i936-synthetic-smoke-v2-s936/val-00001.json)に
全イベントとSHA、process RSSを保持した。NPZ本体はfrontmatterのlocal outputにあり、
Gitには複製していない。全体manifestはfailedなので通常のdataset readerは拒否する。
1件のleaf成功をtrain/val/testの完了へ読み替えない。
64frame共有gapの長系列生成は未通過（unit testの1frame分布検査とは別）。

## 費用予測と次の判断

成功1件だけの参考線形外挿は、640 rallyでserial18.89時間、
理想的な4 workerで4.72時間、NPZ約0.893GB。
3splitの成功例が揃っておらず、失敗対応・並列overheadも含まないため正式な予測ではない。
それでも今回の1時間条件を満たさず、全量生成は行わなかった。

方式Aの条件付きposteriorについて、正depth領域、境界付近の非正則なmode、
異なるcameraのcomponent組合せのevidenceを次runで検証する必要がある。
「全64成分を保持」と「背後modeを拒否」を両立する近似の定義を先に決める。
既存のBの局所積分との対照も候補だが、自動fallbackへは接続しない。
この未解決問題を残したまま実Meiji評価・pipelineへ投入しない。

TensorBoardはない。設定監査98境界、geometry/refinerの22 testsと物理samplingの3 testsは成功。
loss/モデルのCPU検証は[別run](000004-run-i936-diffusion-cpu-memory-s936.md)に分ける。
三角測量の成功や拡散モデルの精度をそこで代用しない。
