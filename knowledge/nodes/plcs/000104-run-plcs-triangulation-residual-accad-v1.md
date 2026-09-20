---
id: run-plcs-triangulation-residual-accad-v1
type: run
task: plcs
sequence: 104
recorded_at: '2026-09-21'
title: PLCS三角測量root・相対姿勢残差のACCAD学習
provider: codex
session: 01a0bf91-5494-74c0-8793-3424866618ed
date: '2026-09-21'
status: done
config:
  model: plcs_triangulation_residual_v1; hidden256; camera/time axial4; 7.05M parameters
  loss: root/relative/world Smooth-L1 + true-camera clean reprojection + GT velocity/bone
  data: plcs/single_object_camera_view_v2; ACCAD; 7998/1004/998 source-motion-disjoint
    scenes
  training: AdamW lr2e-4; batch32; bf16-mixed; seed42; max30; early-stop patience6
metrics:
  world_mpjpe_m: 0.337163
  initial_world_mpjpe_m: 0.421834
  root_error_m: 0.27616
  initial_root_error_m: 0.394697
  relative_mpjpe_m: 0.170996
  initial_relative_mpjpe_m: 0.219008
  test_improvement_percent: 20.072233
  real_before_reprojection_px: 3.25287
  real_after_reprojection_px: 5.938328
  real_raw_valid_limb_outlier_frames_before: 26
  real_raw_valid_limb_outlier_frames_after: 26
repro:
  commit: 893d0ca4129e9e69f8e8c850d89c6200cf2df2e3
  branch: codex/plcs-triangulation-residual
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: '''env'' ''CUDA_VISIBLE_DEVICES=0'' ''OMP_NUM_THREADS=2'' ''OPENBLAS_NUM_THREADS=1''
    ''MKL_NUM_THREADS=2'' ''/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-triangulation-residual/.venv/bin/python''
    ''-u'' ''-m'' ''src.tasks.plcs.scripts.train_triangulation_residual'' ''paths.data_root=/home/kamimura/projects/tennis-lab/data''
    ''paths.output_root=/home/kamimura/projects/tennis-lab/outputs'' ''paths.artifact_root=/home/kamimura/projects/tennis-lab/outputs''
    ''paths.checkpoint_root=/home/kamimura/projects/tennis-lab/outputs'' ''paths.cache_root=/home/kamimura/projects/tennis-lab/.cache''
    ''paths.external_asset_root=/home/kamimura/projects/tennis-lab/data'' ''run.output_dir=plcs/triangulation_residual_v1_20260921''
    ''training.trainer.enable_progress_bar=false'' && ''env'' ''CUDA_VISIBLE_DEVICES=0''
    ''OMP_NUM_THREADS=2'' ''OPENBLAS_NUM_THREADS=1'' ''MKL_NUM_THREADS=2'' ''/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-triangulation-residual/.venv/bin/python''
    ''-u'' ''-m'' ''src.tasks.base.triangulation_residual.inference'' ''--task'' ''plcs''
    ''--run-dir'' ''/home/kamimura/projects/tennis-lab/outputs/plcs/triangulation_residual_v1_20260921''
    ''--clip'' ''/home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000''
    ''--output'' ''/mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual/plcs''
    ''--device'' ''cuda'''
artifacts:
  run_dir: knowledge/runs/run-plcs-triangulation-residual-accad-v1
  predictions: /home/kamimura/projects/tennis-lab/.training_queue/repro/1789927444958097687_81586_plcs-triangulation-residual-accad-v1/predictions/pred_test.npz
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/triangulation_residual_v1_20260921/logs/version_0
  curves: knowledge/runs/run-plcs-triangulation-residual-accad-v1/curves.png
  tb_logdir: outputs/plcs/triangulation_residual_v1_20260921/logs/version_0
parents:
- run-meiji-clip000-triangulation
relations: []
papers: []
tags:
- plcs
- triangulation-residual
- accad
- multiview
- real-clip
---

## 考察 / Findings

### 要約

要求された各cameraの幾何入力からroot residual3とrelative pose residual17×3を出すPLCSを、既存ACCAD由来データで学習した。
合成testのworld MPJPEは0.421834→0.337163m（20.07%改善）、rootは0.394697→0.276160m、relative poseは0.219008→0.170996mへ改善した。
一方、指定実clipの再投影誤差は3.253→5.938pxへ増え、腕の大きな骨長異常も残った。実映像での改善は確認できていない。

### アーキテクチャ詳細

特徴・単位・欠測・モデル・noise生成の正本は[共有README](../../../src/tasks/base/triangulation_residual/README.md)。367特徴/camera/frame、camera/time交互attention、2つの残差head。
このprofileのrootはCOCO11/12の中点で、SMPL translationではない。relativeはcourt axesのままでhipmean0。GT world17から両者を分解して教師とする。
true cameraとclean UVはlossだけで使用し、estimated camera＋noisy UVで初期値を三角測量する。camera誤差分布は合成仮定であり、実calibration誤差を実測fitした分布ではない。
学習は2–4視点、64 frame・約30fps。val/testの4視点とnoise seedは固定。元motion fileのsplit重複は0、subjectは重複する。

### メトリクスの解釈

30epochを完走。選択checkpointはzero-based epoch29、val world MPJPE=0.341528m（初期値0.429299m）。同checkpointを明示loadして998 test scenesを評価した。
testの中央値は0.297539→0.244948m、95%点は1.309357→0.986543m。平均補正量は0.228475m。
通常ノイズ層は0.338822→0.290300m、hard層は1.034010→0.760987m。clean層はほぼ0→0.006692mと悪化し、不要な補正が残る。

実clipの全1010frame・2人・3cameraを推論した。同一98378観測への平均再投影誤差は上記の通り、中央値2.402→4.935px、95%点9.754→13.730px。
平均3D補正は7.21cm、最大40.92cm。欠測を補ったseedを含めると、1m超の四肢骨長を持つperson-frameは40→41、最大骨長3.038→3.047mとなった。
この40は以前のraw triangulationのみの26とは集計集合が異なる。raw-validの同一関節に限定した追加集計は成果物のpaired_anatomy.jsonに保存した。
3D正解はないため、再投影悪化だけから真の3D誤差の増大を断定しない。ただし骨長異常も改善せず、実用上の改善を示す根拠は得られていない。

### アーキテクチャ⇄メトリクスの因果考察

観測：合成側ではroot/relativeとも改善した一方、実clipの誤検出に由来する約3mの前腕は修正できなかった。
仮説：合成の独立camera摂動と実際のCourt14 fitに由来する誤差の相関、観測ノイズ/欠測の分布、ACCADの動作とテニスの動作差が転移を制限している可能性がある。個別原因のablationは未実施。

追加のpost-hoc診断では、実clipの再投影誤差はroot補正のみ5.95px、relative補正のみ4.10px、両方5.94px（初期3.25px）で、root補正の寄与が大きかった。学習時と同じ初期値との一致を確認した合成testの64sceneでは、3D誤差が0.401→0.314m、true camera＋clean UVへの誤差が11.50→8.83pxへ減る一方、estimated camera＋入力UVへの誤差は10.39→12.41pxへ増えた。GT 3D自体をestimated cameraへ投影した誤差も15.29pxだった。この部分集合では教師の3D改善と入力観測への再投影悪化が同時に起きており、実clipの再投影悪化だけから実3D悪化とは断定できない。全testの再評価や原因を分離した再学習ではなく、詳細JSONは会話artifactのgeometric-residual/diagnosis/に保存した。

### 既存実験との比較

同じtest sampleの初期三角測量＋明示的観測seedをbaselineとしたpaired比較。既存PLCSのSMPL-root/yawやfoot priorの指標とは定義・誤差分布が違うため改善率を直接比較しない。
モデル選択は合成validationだけに基づく。testはbf16-mixed、実推論はfloat32。実clipを学習やcheckpoint選択に使っていない。

### 次に有効な実験

同じ合成splitでCourt14からcameraを再推定する誤差生成と独立摂動を比較し、実検証は別clipも含めて行う。継続的な手首誤検出と外れ値を別群として測り、残差特徴の尺度やjoint単位の処理が必要かを評価する。
現段階で幾何初期値からの全面的置換を推奨する結果ではない。

学習前validatorはBLCSと共通で指定1回・試行1回・完了1回。camera集合選択とweights-only読込の2指摘を採用修正し、通常検証と全20000シーンのepoch0走査を通過後にGPU学習した。最終修正後の独立再評価はしていない。

約40 MBのtest予測NPZは上記artifacts.predictionsのローカルqueue保存先に保持し、Gitには含めない。metrics・paired metrics・学習曲線・実行時のcommitとpatchを再現性bundleに保存した。
