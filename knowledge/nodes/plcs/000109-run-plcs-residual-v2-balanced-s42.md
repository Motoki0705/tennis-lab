---
id: run-plcs-residual-v2-balanced-s42
type: run
task: plcs
sequence: 109
recorded_at: '2026-09-21'
title: PLCS Court14校正・持続誤検出v2とbalanced regret損失
provider: codex
session: 01a0bf91-5494-74c0-8793-3424866618ed
date: '2026-09-21'
status: done
config:
  recipe: train_triangulation_residual_v2
  model: plcs_triangulation_residual_v2
  loss: balanced_regret
  features: raw
  seed: 42
  max_epochs: 30
  selected_epoch: 29
  train_scenes: 7998
  val_scenes: 1004
  test_scenes: 998
  train_views: 2..6
  evaluation_views: 3
  true_rig: four corners + near-fence front pair
  camera_error: noisy Court14 -> f/PnP fit
  target: hip-root + court-axis relative COCO17
metrics:
  world_mpjpe_m: 0.13175849616527557
  initial_world_mpjpe_m: 0.1555238515138626
  root_error_m: 0.0803278461098671
  initial_root_error_m: 0.11979939043521881
  relative_mpjpe_m: 0.10234634578227997
  initial_relative_mpjpe_m: 0.15624013543128967
  test_world_median_m: 0.04536234501017109
  initial_world_median_m: 0.04613987371141297
  test_world_p95_m: 0.5437418109654253
  initial_world_p95_m: 0.6624315330662105
  test_sample_improved_fraction: 0.5400801603206413
  test_point_improved_fraction: 0.36897416155841095
  test_central99_mean_gain_m: 0.014028337139691021
  real_before_reprojection_px: 3.2528703509436356
  real_after_reprojection_px: 3.620256367364044
  real_raw_valid_limb_over_1m_before: 28
  real_raw_valid_limb_over_1m_after: 2
repro:
  commit: 4671f9bad0873a23054a3311009b9ecf4a6bbc88
  branch: codex/plcs-triangulation-residual
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tasks.plcs.scripts.train_triangulation_residual --config-name
    train_triangulation_residual_v2 paths.data_root=/home/kamimura/projects/tennis-lab/data
    paths.output_root=/home/kamimura/projects/tennis-lab/outputs run.output_dir=plcs/triangulation_residual_v2_balanced_s42_20260921
    run.seed=42 v2.loss_mode=balanced_regret training.trainer.enable_progress_bar=false
    && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tasks.base.scripts.infer_triangulation_residual --task
    plcs --run-dir /home/kamimura/projects/tennis-lab/outputs/plcs/triangulation_residual_v2_balanced_s42_20260921
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --output '/mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual-v2/plcs/balanced'
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-plcs-residual-v2-balanced-s42
  predictions: /home/kamimura/projects/tennis-lab/outputs/plcs/triangulation_residual_v2_balanced_s42_20260921/predictions/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1789944342452638821_322747_plcs-residual-v2-balanced-s42.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/triangulation_residual_v2_balanced_s42_20260921/logs/version_0
  real_comparison: /mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual-v2/plcs/balanced/index.html
  curves: knowledge/runs/run-plcs-residual-v2-balanced-s42/curves.png
  tb_logdir: outputs/plcs/triangulation_residual_v2_balanced_s42_20260921/logs/version_0
parents:
- run-plcs-triangulation-residual-accad-v1
- run-plcs-residual-v2-gpu-smoke-r2
relations: []
papers: []
tags:
- triangulation-residual-v2
- balanced-regret
- accad
- paired-loss-comparison
---

## 考察 / Findings

### 要約
同じ新しい合成test入力に対し、平均3D誤差0.155524→0.131759 m、中央値0.046140→0.045362 m、p95 0.662432→0.543742 m。改善sample率54.0%、改善point率36.9%で、cleanを含む多数pointの微小悪化は残る。最大補正1%を除いた99%でも平均gainは+0.014028 mだった。

### アーキテクチャと比較条件
モデルはv1と同じcamera/time Transformer。四隅+正面2候補、train2–6/test3 view、Court14を実pipeline共通coreでfit、秒単位の持続誤検出を追加した。balanced lossはsample→severity群平均、3D vector Huber、regretを用いる。ACCAD source-motion-disjointだがsubjectは重複。val最良epoch29を選び、testや実clipをcheckpoint選択には使っていない。v1とは生成誤差・view条件が異なり、数値を直接ランキングしない。同一v2データのlegacy loss runとのpaired比較が必要。

### 実clip
Meiji video_000/clip_000の再投影平均は3.252870→3.620256 px。独立3D正解はない。両端が元から三角測量できた同じ15,775四肢edgeで、1m超は28→2件、該当person-frameは26→2。補完点を含む全集合では49→2件。最大骨長3.038→1.086 mで、frame320のP1左前腕も3.038→0.944 mに縮んだが、なお長すぎるため解決済みとはしない。再投影最大267.7pxの外れもあり、平均値だけで実用性を判断しない。

### 解釈と次の比較
骨格異常の減少は定性的な改善を支持するが、真の3D精度の証明ではない。入力の再投影は初期値自身が最小化する目的であり、true-camera/GTへの学習目的と異なる。入力conditioningのCPU診断は別ノードに記録し、固定scale変換は別の明示profileで検証する。単一seedであり、複数seed・別会場の評価は未実施。
