---
id: run-i983-court-dinov3-l-s42
type: run
task: court_detection
sequence: 37
recorded_at: '2026-10-10'
title: 凍結DINOv3-Lと共通後段の比較学習 (#983)
issue: 983
provider: codex
session: 01a12538-67e1-7d22-a6ce-2d7add686d59
date: '2026-10-10'
status: done
config:
  backbone: dinov3_vitl16
  backbone_train_mode: frozen
  transformer_dim: 1024
  transformer_depth: 8
  heads: 16
  ffn_dim: 2752
  dpt_channels: 512
  long_side: 512
  batch_size: 8
  mixed_batch_counts:
  - 4
  - 4
  seed: 42
  max_epochs: 20
metrics:
  synthetic_court-test:
    kp_mean_distance_px: 4.48781681060791
    kp_median_distance_px: 2.1366634368896484
    pose_reprojection_mean_distance_px: 9.58015251159668
    pose_translation_l2_m: 0.5280944108963013
    pose_rotation_geodesic_deg: 1.7757552862167358
    pose_focal_relative_error: 0.054690275341272354
    line_dice: 0.650072455406189
    seg_miou: 0.9515018463134766
    semantic_line_miou: 0.4618944227695465
    invalid_depth_rate: 0.0
    visible_point_count: 7928.0
  tennis_court_detector-val:
    kp_mean_distance_px: 1.3675116300582886
    kp_median_distance_px: 1.216549038887024
    seg_miou: 0.9701068997383118
    line_dice: 0.707917332649231
    semantic_line_miou: 0.48592713475227356
repro:
  commit: 7d6f53b94f81f8cf7f08b6b78fddcc94d25cc67f
  command: .venv/bin/python -m src.tasks.court_detection.scripts.train --config-name
    train_i983_l run.artifact_store.mode=rclone run.artifact_store.remote=gdrive run.artifact_store.remote_root=tennis_lab/outputs/court_detection/train/i983-dino-l-shared1024/s42
    run.artifact_store.sync_interval_seconds=60
artifacts:
  run_dir: knowledge/runs/run-i983-court-dinov3-l-s42
  tb_logdir: {status: deleted, record: knowledge/runs/run-i983-court-dinov3-l-s42/artifact-manifest.json, path_prefix: tensorboard/version_0/}
  config: knowledge/runs/run-i983-court-dinov3-l-s42/config.yaml
  environment: knowledge/runs/run-i983-court-dinov3-l-s42/environment.json
  first_checkpoint_receipt: knowledge/runs/run-i983-court-dinov3-l-s42/first-checkpoint.json
  interruption: knowledge/runs/run-i983-court-dinov3-l-s42/attempts/01-interruption.json
  resumed_startup: knowledge/runs/run-i983-court-dinov3-l-s42/attempts/02-startup.log
  resumed_receipt: knowledge/runs/run-i983-court-dinov3-l-s42/attempts/02-running-receipt.json
  output_dir: gdrive:tennis_lab/outputs/court_detection/train/i983-dino-l-shared1024/s42
  curves: knowledge/runs/run-i983-court-dinov3-l-s42/curves.png
  evaluation: knowledge/runs/run-i983-court-dinov3-l-s42/evaluation/evaluation.json
  checkpoint_selection: knowledge/runs/run-i983-court-dinov3-l-s42/checkpoint-selection.json
  evaluation_completion: knowledge/runs/run-i983-court-dinov3-l-s42/evaluation-completion.json
  tensorboard_derivation: knowledge/runs/run-i983-court-dinov3-l-s42/tensorboard-derivation.json
parents:
- run-i983-court-l512-profile-s42
relations: []
papers: []
tags:
- issue-983
- colab-l4
- frozen-backbone
- shared-downstream
---

#983の4条件比較のうちViT-Lの20epoch学習と、validation loss最小checkpointの評価を完了した。凍結backbone、長辺512、混合4:4、batch8、seed42、共通1024次元・8層・16 heads・FFN2752・DPT512を固定した。backboneと射影層の初期化の乱数消費を隔離し、他サイズでも共通後段の初期重みを揃える。科学的コードはcommit `7d6f53b94f81f8cf7f08b6b78fddcc94d25cc67f`で固定した。

全20epochの最終stateはepoch=19・global_step=33160。3epochごとのvalidationで保存した6世代から、epoch=17（18epoch目、step=29844、val/loss=1.143399）を選んだ。最小値の根拠はcheckpoint-selection.jsonのcallbackランキング、評価対象のSHA-256はevaluation/evaluation.jsonに記録した。合成testを選択に使わず、test_after_fitの終端モデル評価とも区別した。学習・正式評価ともreturncode=0とDrive保存を確認し、全jobの保存後にL4 sessionを終了した。

合成V3 test 886件と実画像val 2211件を別々に評価し、入力manifestと順序付きsample IDのSHA-256をpreflightのsnapshotと照合した。各splitの固定4画像について、合成はdense4出力とpose、実画像はdense4出力のPNG計36枚を保存した。距離のpxは長辺512へ等方resizeした入力座標であり、元画像のpxではない。

| 出力 | 合成V3 test | 実画像val |
|---|---:|---:|
| keypoint 平均距離 (px) | 4.4878 | 1.3675 |
| segmentation mIoU | 0.9515 | 0.9701 |
| line Dice | 0.6501 | 0.7079 |
| semantic-line mIoU | 0.4619 | 0.4859 |
| pose 再投影平均距離 (px) | 9.5802 | N/A（教師なし） |

合成poseの並進L2は0.5281 m、回転geodesicは1.7758°、焦点距離の相対誤差は0.05469。keypoint中央値は合成2.1367 px・実画像1.2165 pxで、合成の平均が中央値より大きく、一部の大きい誤差の影響がある。固定例B00:court-sample-000897のpose投影にもGTとのずれが見える。実画像xzjBgO51hsA_1350では主要な線配置が得られているが、この数例を全体精度に一般化しない。実画像valはcheckpoint選択にも参加するため、独立testの成績とは扱わない。無効化したconsistency項目の0を整合精度と解釈しない。

初回sessionは2026-10-10 16:08 UTCの監視でassignment消失を確認した。最後のDriveログはEpoch 2の111/1658で、終了コードと終了原因は不明。直前のlast.ckptをDrive上の`resumes/attempt-01-last.ckpt`へcopy・検証した。Colab skillの実ジョブをSSHからNotebook kernel経由へ変更し、科学的コードと学習条件を維持して新L4 VMで再開した。全9入力のSHA-256と退避checkpointを検証し、epoch=1・global_step=3316・optimizer/scheduler各1件を復元した。解決済み設定は`run.resume`以外が初回と一致する。

Lightning 2.6.6の通常checkpointはRNG stateを保存せず、混合samplerも内部epochをcheckpoint化しない。したがって中断なしrunとbitwise同一の継続は保証しない。未完了だった3epoch目の途中更新は再開元に含まれない。TensorBoardの曲線は旧attemptのstep<3316と再開attemptの全train/val scalarを採用した。未保存部分の72 scalarを除き、24,448 scalarのtag/step重複が0であることを確認した。元eventのhashとDrive参照、抽出スクリプト、導出記録をbundleに残し、値・step・wall_timeを保持したscalar-only eventからkg_curves.pyで曲線を生成した。終端testの50 scalarは別JSONで保持する。

初回のcommand/config、attemptごとのrequest・中断・再開・完了証拠はbundleに保持する。共通入力snapshotは親preflightのbundleが正本で、checkpoint・大容量の全件予測はDriveに保存した。repro.shは同条件の新規学習用であり、今回の中断軌跡のbitwise再現を保証しない。短いpreflightで更新した重みは本学習へ流用していない。

S/S+/Bの同条件runが揃うまでサイズ間の優劣や採用変更は判断しない。次は同じ入力・split・後段容量で残り3条件を学習し、最良validation checkpointを同じ固定画像と別split表で比較する。過去のLoRAありBの成績を今回のbaselineに混ぜない。
