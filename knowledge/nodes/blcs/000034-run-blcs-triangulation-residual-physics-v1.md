---
id: run-blcs-triangulation-residual-physics-v1
type: run
task: blcs
sequence: 34
recorded_at: '2026-09-21'
title: BLCS三角測量残差モデルの物理シミュレーション学習
provider: codex
session: 01a0bf91-5494-74c0-8793-3424866618ed
date: '2026-09-21'
status: done
config:
  model: blcs_triangulation_residual_v1; hidden256; camera/time axial4; 6.94M parameters
  loss: metre root/world Smooth-L1 + clean true-camera reprojection + supervised velocity
  data: blcs/single_object_camera_view_v2; physics; 8000/1000/1000 scenes; no real-clip
    replay
  training: AdamW lr2e-4; batch32; bf16-mixed; seed42; max30; early-stop patience6
metrics:
  world_mpjpe_m: 0.589038
  initial_world_mpjpe_m: 0.59538
  root_error_m: 0.589038
  initial_root_error_m: 0.59538
  relative_mpjpe_m: 0.0
  initial_relative_mpjpe_m: 0.0
  test_improvement_percent: 1.065201
  real_before_reprojection_px: 2.301307
  real_after_reprojection_px: 2.324268
  real_mean_correction_m: 0.005315
repro:
  commit: 893d0ca4129e9e69f8e8c850d89c6200cf2df2e3
  branch: codex/plcs-triangulation-residual
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: '''env'' ''CUDA_VISIBLE_DEVICES=0'' ''OMP_NUM_THREADS=2'' ''OPENBLAS_NUM_THREADS=1''
    ''MKL_NUM_THREADS=2'' ''/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-triangulation-residual/.venv/bin/python''
    ''-u'' ''-m'' ''src.tasks.blcs.scripts.train_triangulation_residual'' ''paths.data_root=/home/kamimura/projects/tennis-lab/data''
    ''paths.output_root=/home/kamimura/projects/tennis-lab/outputs'' ''paths.artifact_root=/home/kamimura/projects/tennis-lab/outputs''
    ''paths.checkpoint_root=/home/kamimura/projects/tennis-lab/outputs'' ''paths.cache_root=/home/kamimura/projects/tennis-lab/.cache''
    ''paths.external_asset_root=/home/kamimura/projects/tennis-lab/data'' ''run.output_dir=blcs/triangulation_residual_v1_20260921''
    ''training.trainer.enable_progress_bar=false'' && ''env'' ''CUDA_VISIBLE_DEVICES=0''
    ''OMP_NUM_THREADS=2'' ''OPENBLAS_NUM_THREADS=1'' ''MKL_NUM_THREADS=2'' ''/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-triangulation-residual/.venv/bin/python''
    ''-u'' ''-m'' ''src.tasks.base.triangulation_residual.inference'' ''--task'' ''blcs''
    ''--run-dir'' ''/home/kamimura/projects/tennis-lab/outputs/blcs/triangulation_residual_v1_20260921''
    ''--clip'' ''/home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000''
    ''--output'' ''/mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual/blcs''
    ''--device'' ''cuda'''
artifacts:
  run_dir: knowledge/runs/run-blcs-triangulation-residual-physics-v1
  predictions: knowledge/runs/run-blcs-triangulation-residual-physics-v1/pred_test.npz
  output_dir: /home/kamimura/projects/tennis-lab/outputs/blcs/triangulation_residual_v1_20260921/logs/version_0
  curves: knowledge/runs/run-blcs-triangulation-residual-physics-v1/curves.png
  tb_logdir: outputs/blcs/triangulation_residual_v1_20260921/logs/version_0
parents:
- run-meiji-clip000-triangulation
relations: []
papers: []
tags:
- blcs
- triangulation-residual
- physics
- multiview
- real-clip
---

## 考察 / Findings

### 要約

要求された幾何入力からXYZ残差を出すBLCSを訓練した。合成testの平均3D誤差は0.595380→0.589038m（1.07%改善）だが、中央値は0.356384→0.356457mと改善しなかった。
指定実クリップの再投影誤差は2.301→2.324px。改善を支持する結果ではなく、実clipではほぼ一定の5mm補正に留まる。

### アーキテクチャ詳細

共有実装・特徴順・単位・欠測・誤差生成の正本は[残差学習README](../../../src/tasks/base/triangulation_residual/README.md)。BLCSはJ=1、79入力特徴/camera/frameから位置残差3を出す。
true camera/clean 2Dはlossのみで使い、noisy観測とestimated cameraで初期値とmodel入力を作る。学習は2–4視点、val/testは4視点。64 frame、target 30fps。
初期化可能なcamera集合を観測だけから選択し、同じscene/time/GT/splitを保持する。camera集合・corruptionの試行回数はbatchへ記録する。

### メトリクスの解釈

zero-based epoch12まで13epoch学習して早期終了。validation最良はepoch6、world MPJPE=0.633034m（初期値0.633977m）。このcheckpointを明示loadして全1000 test sceneを評価した。
testの95%誤差は1.952414→1.949352m。補正量は平均2.28cm、中央値0.525cm、95%点0.693cmで、ごく一部だけ大きな修正が入る。
通常ノイズ層は0.497197→0.491993m、hard層は1.373656→1.357368m。clean層は0.034381→0.035633mと悪化した。cleanでも画角による不可視とseed補間があるため初期誤差が常にゼロではない。

実clipは全1010 frame、3 camera。外部2D annotationのobserved/interpolated等を区別し、元のcourt-only近似cameraで比較した。confidence≥0.3の同一2872観測に対する再投影誤差は上記の通り。
実補正ベクトルの平均XYZは(-3.176,+3.596,+2.287)mm、時間方向stdは(0.014,0.081,0.008)mmで、軌道に応じた補正はほぼ得られていない。
実映像の3D正解はない。合成testの小幅改善を実3D精度の改善とは解釈しない。

### アーキテクチャ⇄メトリクスの因果考察

観測：大部分は小さなほぼ一定の補正で、median errorが改善せず、極端な一部の初期値を直して平均値を少し下げている。
仮説：camera誤差・時刻ずれの曖昧さと、小さいnormalized UV残差の特徴尺度が、条件付きの有効な補正を学びにくくしている可能性がある。個別原因を切り分けたablationは未実施。

### 既存実験との比較

同一test sampleの初期三角測量＋明示的欠測seedをbaselineとするpaired比較。別モデル・別誤差分布のBLCS runと改善率を比較しない。
checkpoint選択は合成validationだけを使用し、実clipへ合わせた選択はしていない。testはbf16-mixed、実clipはfloat32で推論した。

### 次に有効な実験

まず残差特徴の尺度と、camera誤差／時刻ずれを分けた評価を比較し、初期値をそのまま保つ必要のあるclean caseの悪化を確認する。訓練済みモデルを幾何baselineの全面的な代替として採用する根拠はまだない。

学習前validatorは両task共通で指定1回・試行1回・完了1回。camera集合による初期化停止とweights-only checkpoint読込の2件を採用修正し、通常検証と全20000シーンのepoch0入力走査を通過後に学習した。修正後の独立再評価はしていない。
