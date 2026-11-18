---
id: run-person-stage-precision-meiji-20261118
type: run
task: tennis_scene
sequence: 50
recorded_at: '2026-11-18'
title: Meiji 4camの全尺2クリップで人物推論の精度・flip・追跡依存を比較
issue: 2238
provider: codex
session: 01a1ec52-c2ef-70e3-bd48-588f68ed717e
date: '2026-11-18'
status: done
config:
  clips:
  - clip_037
  - clip_000
  source_frames:
  - 292
  - 356
  cameras_per_clip: 4
  dino_batch_size: 1
  players_per_side: 2
  court_side_margin: 0.15
  clip_037_diagnostic_margin: 0.12
metrics:
  completed_cases: 12
  camera_frames_per_condition: 2592
  fp32_clip_000_exact_dino_and_selected_pose: true
  dino_fp32_seconds_2_clips: 705.89
  vitpose_fp32_seconds_2_clips: 996.94
  vitpose_bf16_seconds_2_clips: 375.65
  vitpose_fp16_seconds_2_clips: 357.26
  dino_bf16_seconds_2_clips: 410.27
  dino_fp16_seconds_2_clips: 410.48
  vitpose_no_flip_seconds_2_clips: 505.38
repro:
  commit: cbefdef4bc913fadc574be83af38e83f8c3f931d
  branch: codex/person-stage-speed
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1 TENNIS_LAB_DINO_EXTENSION=/home/kamimura/projects/tennis-lab/.claude/worktrees/video-batch-cli/outputs/tennis_scene/evaluate/video-batch-runtime-20261117/dino-extension-a2/lib/MultiScaleDeformableAttention.so
    .venv/bin/python -u knowledge/runs/run-person-stage-precision-meiji-20261118/gpu_guard.py
    --health outputs/precision/full/health-resume knowledge/runs/run-person-stage-precision-meiji-20261118/benchmark.py
    --dataset data/tennis_multiview/processed/meiji_4cam/dataset --shared-root /home/kamimura/projects/tennis-lab/.worktree-outputs/court-video-shared
    --pin knowledge/runs/run-person-stage-precision-meiji-20261118/full-inputs.json
    --output outputs/precision/full --smoke outputs/precision/smoke/health/status.json
    --resume
artifacts:
  run_dir: knowledge/runs/run-person-stage-precision-meiji-20261118
  log: .training_queue/logs/0-8205027480292773760_2026-11-18T03-28-39-707226239Z_2481399_i2238-precision-meiji-resume.log
  output_dir: .worktree-outputs/person-stage-speed/precision/full
parents:
- run-person-stage-precision-smoke-20261118
- run-person-stage-precision-interrupted-20261118
relations: []
papers: []
tags:
- inference-precision
- person-detection
- pose-estimation
- pose-tracking
summary_impact: none
---

# 固定入力とFP32の一致

meiji_4camのclip_037（292 frame）、clip_000（356 frame）を各4カメラ・全フレームで測定した。
ダブルス、共有コート校正、X3D＋DPTボール観測を固定し、DINOはbatch 1、ViTPoseはcrop batch上限16とした。
FP32基準からモデル精度またはflip testの一項目を変更した。DINOの混合精度では変形可能アテンションのブロックをFP32で実行した。
[固定入力と重み](../../runs/run-person-stage-precision-meiji-20261118/full-inputs.json)・[コート設定](../../runs/run-person-stage-precision-meiji-20261118/court-settings.json)

旧推論コードとの固定8画像の完全一致に加え、過去の #1752 のFP32単一clip出力とclip_000を照合した。
4台すべてで入力動画のSHA-256・FPS・解像度・フレーム数が一致し、検出offset・box・score・元rowと、
選手の姿勢座標・confidence・local ID・観測maskが完全一致した。
[#1752の既存出力との照合](../../runs/run-person-stage-precision-meiji-20261118/clip_000-prior-fp32-parity.json)

# 推論時間とVRAM

| 変更した条件 | clip_037 推論秒 | clip_000 推論秒 | 2clipのFP32比 | 対象段の割当ピーク最大 GiB | 対象段の予約ピーク最大 GiB |
|---|---:|---:|---:|---:|---:|
| DINO FP32 | 319.7 | 386.2 | 1.00× | 1.74 | 2.12 |
| ViTPose FP32 | 467.7 | 529.3 | 1.00× | 3.10 | 4.08 |
| ViTPose BF16 | 176.5 | 199.2 | 2.65× | 4.24 | 4.68 |
| ViTPose FP16 | 168.4 | 188.9 | 2.79× | 4.20 | 4.59 |
| DINO BF16 | 186.6 | 223.7 | 1.72× | 1.65 | 2.11 |
| DINO FP16 | 183.0 | 227.5 | 1.72× | 1.65 | 1.93 |
| ViTPose FP32・flipなし | 236.8 | 268.6 | 1.97× | 2.95 | 3.39 |

時間はRTX 5060 Tiでの同期wall timeのカメラ合計。DINOはテンソル転送とbox・score復号、ViTPoseは
crop・転送・UDP復号を含む。カメラ別の段全体のwall timeにはモデルロード・動画decode・保存も含む。
比較条件は基準のコートとボールの成果物を再利用し、ViTPoseだけを変更した条件では基準のDINO検出も再利用した。
VRAMは対象段でのPyTorch allocatorの割当・予約を全カメラ・両クリップで集計した最大値。ViTPoseの対象段はCLIPと追跡処理も実行する。
各条件は1回計測した。
[clip_037のFP32段別計測](../../runs/run-person-stage-precision-meiji-20261118/clip_037-fp32.json)・[clip_000のFP32段別計測](../../runs/run-person-stage-precision-meiji-20261118/clip_000-fp32.json)

# FP32に対する出力差

clip_000は通常のcourt side→人物対応→三角測量を使用した。clip_037は通常のmargin 0.15では
`court_side_ambiguous_margin` で停止したため、事前に固定した診断用margin 0.12で人物対応・3D関節を比較した。
全ケースで基準と条件の映像・コート校正を揃えた。

| clip | 条件 | 未対応DINO行 FP32/条件 | box / score P95 | 選手pose P95 px | 追跡ID差 | 選別 減/増 | 人物ID差 | 3D P95 mm / 最大 cm | 3D有効 減/増 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clip_037 | [ViTPose BF16](../../runs/run-person-stage-precision-meiji-20261118/clip_037-vitpose_bf16.json) | 0 / 0 | 0.000 px / 0.0000 | 0.074 | 0 | 0 / 0 | 0 | 0.90 / 44.1 | 0 / 0 |
| clip_037 | [ViTPose FP16](../../runs/run-person-stage-precision-meiji-20261118/clip_037-vitpose_fp16.json) | 0 / 0 | 0.000 px / 0.0000 | 0.005 | 0 | 0 / 0 | 0 | 0.08 / 3.9 | 0 / 0 |
| clip_037 | [DINO BF16](../../runs/run-person-stage-precision-meiji-20261118/clip_037-dino_bf16.json) | 50 / 23 | 0.126 px / 0.0157 | 0.113 | 15 | 1 / 34 | 0 | 2.06 / 14.0 | 0 / 0 |
| clip_037 | [DINO FP16](../../runs/run-person-stage-precision-meiji-20261118/clip_037-dino_fp16.json) | 12 / 16 | 0.036 px / 0.0064 | 0.071 | 46 | 71 / 2 | 0 | 5.54 / 16.5 | 0 / 0 |
| clip_037 | [ViTPose FP32・flipなし](../../runs/run-person-stage-precision-meiji-20261118/clip_037-vitpose_no_flip.json) | 0 / 0 | 0.000 px / 0.0000 | 0.879 | 70 | 0 / 0 | 0 | 12.94 / 48.8 | 0 / 0 |
| clip_000 | [ViTPose BF16](../../runs/run-person-stage-precision-meiji-20261118/clip_000-vitpose_bf16.json) | 0 / 0 | 0.000 px / 0.0000 | 0.089 | 0 | 0 / 0 | 0 | 1.12 / 62.6 | 3 / 1 |
| clip_000 | [ViTPose FP16](../../runs/run-person-stage-precision-meiji-20261118/clip_000-vitpose_fp16.json) | 0 / 0 | 0.000 px / 0.0000 | 0.006 | 0 | 0 / 0 | 0 | 0.08 / 30.9 | 0 / 0 |
| clip_000 | [DINO BF16](../../runs/run-person-stage-precision-meiji-20261118/clip_000-dino_bf16.json) | 70 / 49 | 0.234 px / 0.0223 | 0.149 | 326 | 4 / 4 | 6 | 2.60 / 58.5 | 8 / 3 |
| clip_000 | [DINO FP16](../../runs/run-person-stage-precision-meiji-20261118/clip_000-dino_fp16.json) | 20 / 21 | 0.057 px / 0.0095 | 0.090 | 74 | 1 / 0 | 0 | 1.32 / 64.8 | 1 / 1 |
| clip_000 | [ViTPose FP32・flipなし](../../runs/run-person-stage-precision-meiji-20261118/clip_000-vitpose_no_flip.json) | 0 / 0 | 0.000 px / 0.0000 | 0.837 | 106 | 5 / 0 | 0 | 16.34 / 80.4 | 12 / 15 |

同じframeのboxをIoU 0.5以上で一対一対応させた。box・scoreと選手poseのP95はカメラ別P95の最大。
選手poseは両条件で選別された行のうちconfidence 0.3以上の関節を比較した。
追跡と人物IDは対応した検出行のうち両条件で割当がある行を、重複数が最大になるID対応後に数えた。
3D距離は人物IDを対応付け、両条件で有効な関節を集計した。元のID番号、失われた観測・追加観測、比較対象数は各JSONに記録した。
[比較の定義](../../runs/run-person-stage-precision-meiji-20261118/comparison-method.json)・[集約値](../../runs/run-person-stage-precision-meiji-20261118/overview.json)

ViTPoseの混合精度は両クリップの追跡・選別・人物対応の割当を維持した。FP16の3D距離P95は両クリップで
0.1 mm未満で、clip_000には最大約31 cmの差があった。BF16のclip_037ではframe 152の左足首の3D差が
44.1 cmで、同じ関節のcam3の2D差は55.3 pxだった。各条件JSONに最大差のframe・関節・各viewの観測を保存した。
flipなしではViTPoseの推論が約2倍になり、clip_000の3D距離P95は16.3 mm、最大差は80.4 cmだった。

2026-11-18に決定したmeiji_4cam本番の設定はDINO・ViTPoseともにFP16。標準パイプラインの既定はFP32。
この比較で測ったのは、各モデルの精度を個別に変えた条件。

clip_037の全4カメラのframe 146で、選別済みbox・COCO17を基準と各条件で重ねた。
[ViTPose BF16](../../runs/run-person-stage-precision-meiji-20261118/clip_037-vitpose_bf16.png)、[ViTPose FP16](../../runs/run-person-stage-precision-meiji-20261118/clip_037-vitpose_fp16.png)、[DINO BF16](../../runs/run-person-stage-precision-meiji-20261118/clip_037-dino_bf16.png)、[DINO FP16](../../runs/run-person-stage-precision-meiji-20261118/clip_037-dino_fp16.png)、[ViTPose FP32・flipなし](../../runs/run-person-stage-precision-meiji-20261118/clip_037-vitpose_no_flip.png)

# 選手候補だけに姿勢推定する場合

標準のStrongSORT++は全検出のbox・score・CLIPと、最後に観測した姿勢を使う。pose重み0.15の距離が
外観マッチとIoUマッチの両方に入り、その結果がAFLink・GSIに渡る。clipを通した選別の結果は追跡後に決まる。
[追跡のマッチングと更新](../../../src/tasks/person_tracking/strongsort.py)・[使用profile](../../../src/tasks/person_tracking/sequence.py)

選手選別と人物対応はbox・コート上の位置・外観から計算できる。三角測量と身体viewの選択は選別・対応後の
COCO17座標とconfidenceを使う。GVHMRにはそこから選んだview・区間と人物box・2D姿勢が渡る。
姿勢なしの既存データセット用profileをCPUで再生し、FP32で保存したDINO検出行と追跡用CLIP特徴を固定して
追跡・選手選別を比較した。
[選別段](../../../src/tennis_scene/pipeline/components/player_selection.py)・[姿勢段](../../../src/tennis_scene/pipeline/components/pose_estimation.py)・[身体view選択](../../../src/tennis_scene/pipeline/components/body_view_selection.py)

| clip | 全検出行 | 姿勢なし追跡の選別グループ行 | 選別後に推定する場合の対象減 | 追跡ID差 | 選別 減/増 |
|---|---:|---:|---:|---:|---:|
| clip_037 | 9,356 | 4,506 | 51.8% | 106 | 0 / 0 |
| clip_000 | 10,605 | 4,983 | 53.0% | 182 | 2 / 40 |

固定特徴での比較では、両クリップの追跡IDが変わり、clip_000では選別行も変わった。
選別後の候補にだけ姿勢を付ける順序は実装できる。標準パイプラインに接続する際は別の追跡profileとして評価が必要になる。
表の削減幅は各グループの実観測を一度ずつ姿勢推定する場合の行数。再生では実行時間と3D品質を測っていない。

カメラ間の人物対応では、保存した標準profileのCUDA結果に対し、姿勢なしprofileの選別を使って対応用CLIPもCPUで再推論した。
追跡profileと対応用CLIPの実行deviceをともに変えた条件では、両方で割り当てられた検出行の人物ID差が
clip_037で0件、clip_000で2件だった。この比較では人物ID差への各要因の寄与を分離できない。
姿勢なしprofileの影響は、固定特徴で比較した追跡・選手選別まで確認した。
[姿勢なし追跡の差分](../../runs/run-person-stage-precision-meiji-20261118/pose-scope.json)

# 計測の来歴

共有キューのTERMで最初のジョブが中断した。完了済み5件を保持し、媒体・重み・推論コード・GPU環境を照合して
残りを完了した。原commandと実行HEADは各runの記録に保存した。
[初回の中断記録](./000049-run-person-stage-precision-interrupted-20261118.md)・[再開のrun記録](../../runs/run-person-stage-precision-meiji-20261118/run.json)

全ケースを新規に測る場合は、固定入力に対するsmokeを共有キューで実行した後、全尺のコマンドには空の出力先を指定し、
`--resume` を外す。比較・重ね描きは `compare.py`、姿勢なし追跡は `pose_scope.py` のCPU入口で再計算する。
[計測入口](../../runs/run-person-stage-precision-meiji-20261118/benchmark.py)・[比較入口](../../runs/run-person-stage-precision-meiji-20261118/compare.py)・[姿勢なし追跡の再生入口](../../runs/run-person-stage-precision-meiji-20261118/pose_scope.py)
