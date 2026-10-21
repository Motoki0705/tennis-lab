---
id: run-i983-court-dinov3-s-s42
type: run
task: court_detection
sequence: 38
recorded_at: '2026-10-11'
title: 凍結DINOv3-Sと共通後段の比較学習 (#983)
issue: 983
provider: codex
session: 01a12538-67e1-7d22-a6ce-2d7add686d59
date: '2026-10-11'
status: done
config:
  backbone: dinov3_vits16
  backbone_train_mode: frozen
  native_feature_dim: 384
  projection: four independent learned 1x1 384 -> 1024
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
    kp_mean_distance_px: 4.034967422485352
    kp_median_distance_px: 2.306164264678955
    pose_reprojection_mean_distance_px: 184.27508544921875
    pose_translation_l2_m: 0.6720935702323914
    pose_rotation_geodesic_deg: 2.4906294345855713
    pose_focal_relative_error: 0.062411729246377945
    line_dice: 0.6370841860771179
    seg_miou: 0.9494348168373108
    semantic_line_miou: 0.4481726884841919
    invalid_depth_rate: 0.0002522704307921231
    visible_point_count: 7928.0
  tennis_court_detector-val:
    kp_mean_distance_px: 1.3464598655700684
    kp_median_distance_px: 1.1999969482421875
    seg_miou: 0.96958988904953
    line_dice: 0.705335795879364
    semantic_line_miou: 0.484466016292572
repro:
  commit: 75e30a565b1f7a5f7fd4f08cea2daf458d377653
  command: .venv/bin/python -m src.tasks.court_detection.scripts.train --config-name
    train_i983_s run.output_dir=court_detection/train/i983-dino-s-shared1024/s42/attempt-02
    run.artifact_store.mode=rclone run.artifact_store.remote=gdrive run.artifact_store.remote_root=tennis_lab/outputs/court_detection/train/i983-dino-s-shared1024/s42/attempt-02
    run.artifact_store.sync_interval_seconds=60
artifacts:
  run_dir: knowledge/runs/run-i983-court-dinov3-s-s42
  tb_logdir: {status: deleted, record: knowledge/runs/run-i983-court-dinov3-s-s42/artifact-manifest.json, path_prefix: tensorboard/version_0/}
  config: knowledge/runs/run-i983-court-dinov3-s-s42/attempts/02-config.yaml
  environment: knowledge/runs/run-i983-court-dinov3-s-s42/attempts/02-environment.json
  input_verification: knowledge/runs/run-i983-court-dinov3-s-s42/input-verification.json
  condition_comparison: knowledge/runs/run-i983-court-dinov3-s-s42/condition-comparison.json
  running_receipt: knowledge/runs/run-i983-court-dinov3-s-s42/attempts/02-running-receipt.json
  output_dir: gdrive:tennis_lab/outputs/court_detection/train/i983-dino-s-shared1024/s42/attempt-02
  log: knowledge/runs/run-i983-court-dinov3-s-s42/attempts/02-startup.log
  startup_status: knowledge/runs/run-i983-court-dinov3-s-s42/attempts/02-startup-status.json
  failed_startup: knowledge/runs/run-i983-court-dinov3-s-s42/failed-startup.log
  autocast_fix: knowledge/runs/run-i983-court-dinov3-s-s42/autocast-fix.patch
  curves: knowledge/runs/run-i983-court-dinov3-s-s42/curves.png
  evaluation: knowledge/runs/run-i983-court-dinov3-s-s42/evaluation/evaluation.json
  evaluation_verification: knowledge/runs/run-i983-court-dinov3-s-s42/evaluation-verification.json
  checkpoint_selection: knowledge/runs/run-i983-court-dinov3-s-s42/checkpoint-selection.json
  pose_diagnostic: knowledge/runs/run-i983-court-dinov3-s-s42/pose-diagnostic/diagnostic.json
  diagnostic_points: {status: deleted, record: knowledge/runs/run-i983-court-dinov3-s-s42/artifact-manifest.json#/files/0}
  completion: knowledge/runs/run-i983-court-dinov3-s-s42/attempts/02-completion.json
parents:
- run-i983-court-l512-profile-s42
relations:
- to: run-i983-court-dinov3-l-s42
  rel: compares
papers: []
tags:
- issue-983
- colab-l4
- frozen-backbone
- shared-downstream
---

#983の4条件比較のうち、凍結DINOv3-Sの20epoch学習と最小validation loss checkpointの正式評価を完了した。384次元の4段特徴を、それぞれ独立した学習可能な1×1射影で1024次元へ揃える。凍結backboneは21,601,152 parameter、学習する射影は1,576,960、共通Transformer/DPT/headはLと同じ128,635,436である。seed42の共通後段初期化を維持する。

最終stateはepoch19・global_step33160。保存済み6世代からepoch17（18epoch目、step29844、val/loss=1.1579619646）を選択した。monitor=Noneのlast保存callbackとmonitor=val/lossの世代管理callbackを区別し、後者のランキングで最小値を確認した。合成testを選択には使わず、test_after_fitの終端モデル評価とは別に評価した。学習・正式評価・補足診断はいずれもreturncode=0とDrive保存を確認し、全job終了後にsession03を停止した。

入力は親preflightのB00〜B03とTennisCourtDetector-v1、同じsnapshotとsplitである。manifest・index・画像shard・S pretrained weightのSHA-256を照合し、Lとのcomparison_contractも完全一致した。データ・順序・前処理・教師・学習条件・共通後段parameter・評価precision・固定画像の一致が評価証跡に残る。主要環境はPython3.11.16/PyTorch2.14.1/Lightning2.6.6/L4でLと同じ。学習コードはc4b12f923で、Lの実行版からの追加コード変更は後述のautocast入力検査だけである。コード版の違いによる演算・学習条件の変更はない。

| 出力 | 合成V3 test 886画像 | 実画像val 2211画像 |
|---|---:|---:|
| KP 平均距離 (px) | 4.0350 | 1.3465 |
| SEG mIoU | 0.9494 | 0.9696 |
| LINE Dice | 0.6371 | 0.7053 |
| semantic LINE mIoU | 0.4482 | 0.4845 |
| pose 再投影平均距離 (px) | 184.2751 | N/A（教師なし） |

pxは長辺512へ等方resizeした入力座標で、今回の固定入力は512×288。KP中央値は合成2.3062px・実画像1.2000px。合成poseの並進L2は0.6721m、回転geodesicは2.4906°、焦点距離相対誤差は0.06241。実画像valはcheckpoint選択にも参加し、独立testとは扱わない。無効なconsistency項目の0を整合精度と解釈しない。

pose再投影平均が大きかったため、同じcheckpoint・合成886画像・batch8・bf16で点別診断を追加した。元のCourtPoseGeometryMetricsに処理を委譲した上で、実際に集計された誤差・予測depth・GT可視mask・sample IDを捕捉した。再実行の全体指標は元の184.275085pxと一致し、可視7,928点・無効depth2点も一致した。元の指標は一切変更しない。

2点はいずれも`B01:court-sample-002077`にあり、semantic index1の予測depthは-1.3537m・誤差2405.67px、index6は0.002696m・誤差1,336,525.63pxだった。しきい値0.1m以下のこの2点が誤差総和の91.65%を占める。全可視点の中央値は7.9122px・p95は46.4149pxで、平均が極端な予測に引き上げられている。深度有効な7,926点だけの補足平均は15.3926pxだが、これは正式値184.2751pxを置き換える指標ではない。詳細は[補足診断](../../runs/run-i983-court-dinov3-s-s42/pose-diagnostic/diagnostic.json)と点別JSONを参照する。診断コード・コマンドもbundleに保持し、ローカルでscriptを直接呼ぶ場合はrepo rootをPYTHONPATHに指定する。

今回の固定データとseedでは、Sのdense KP平均距離はL（合成4.4878px/実画像1.3675px）より小さい一方、合成poseの大きな外れ値と他のdense指標は改善していない。KPはheatmap、poseは独立したpose headの投影で、両者は別の予測経路である。固定4画像だけでは稀なpose失敗を網羅できない。Lは途中checkpointから再開したという制約もあり、S+/Bを含む4条件と再現上の制約を揃えるまで、サイズ間の一般的な優劣や採用変更は判断しない。

初回はLと同じcommit98b737829で開始したが、sanity validationの最初のbatchで、Conv射影のbf16出力とfp32 parameterをdtype完全一致検査が拒否した。optimizer更新前の失敗で、returncode=1とDrive保存を確認した。c4b12f923ではfp32 parameter・有効なdevice autocast・指定autocast dtypeという組み合わせだけを許可し、演算やcastは追加しなかった。CPUで同じ失敗を再現後、射影・Transformerの有限かつ非zeroなgradient、出力、不正dtypeの拒否を含む31テストが成功した。

同じVMへGitで修正を反映し、初回のconfig/logを保持した別出力`.../s42/attempt-02`へ同じseedから新規実行した。checkpoint再開は使っていない。解決済み設定は初回Sから出力先だけが変わり、初回SとLの差もbackbone名・weight・取得層・保存先だけだった。attemptごとのcommand/config/失敗/完了証拠はrun.jsonから辿れる。

TensorBoardは最終flush・Drive保存後のsourceを再取得し、SHA-256を固定した。成功attemptのtrain/val 24,448 scalarを元の値・step・wall_timeで抽出し、tag/step重複0・除外0を確認した。scalar-only eventからkg_curves.pyで曲線を生成し、終端testの50 scalarは別JSONへ分離した。checkpointと大容量の終端全件予測はDriveに保持し、全量ローカル回収はしていない。次はS+/Bを同じ条件で実行し、4条件の別split表・固定画像比較に統合する。
