---
id: run-i983-court-dinov3-s-s42
type: run
task: court_detection
sequence: 38
recorded_at: '2026-10-11'
title: 凍結DINOv3-Sと共通後段の比較学習 (#983)
issue: 983
provider: codex
date: '2026-10-11'
status: running
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
metrics: {}
artifacts:
  run_dir: knowledge/runs/run-i983-court-dinov3-s-s42
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
session: 01a12538-67e1-7d22-a6ce-2d7add686d59
repro:
  commit: c4b12f923b243da6b502149aa6f3912c706045e7
  command: .venv/bin/python -m src.tasks.court_detection.scripts.train --config-name
    train_i983_s run.output_dir=court_detection/train/i983-dino-s-shared1024/s42/attempt-02
    run.artifact_store.mode=rclone run.artifact_store.remote=gdrive run.artifact_store.remote_root=tennis_lab/outputs/court_detection/train/i983-dino-s-shared1024/s42/attempt-02
    run.artifact_store.sync_interval_seconds=60
---

#983の4条件比較の2条件目として、凍結DINOv3-Sの20epoch学習をColab L4で実行する。Sの384次元の4段特徴にそれぞれ独立した学習可能な1×1射影を追加し、1024次元の共通Transformer・DPTへ渡す。backboneと射影の初期化に使う乱数を隔離し、Lと同じseed42で共通後段を初期化する。backboneと射影のparameter数は最終比較で共通後段と分けて報告する。

親preflightの入力snapshotからB00〜B03・TennisCourtDetector-v1の5 datasetとS pretrained weightを新VMへ取得した。Driveとの転送検証に加え、manifest・index・画像shard・weightのSHA-256をすべて照合した。input-verification.jsonはこの6資産の証跡で、準備jobのreturncode=0とDrive保存を確認してから学習を開始した。入力の採用版やsplitを変更していない。Python3.11.16・PyTorch2.14.1・Lightning2.6.6・L4で、主要package・DINOv3 submodule・driverはLの記録と一致する。

初回はLと同じ科学的commit98b737829で開始したが、sanity validationの最初のbatchで停止した。混合精度のConv射影がbf16特徴を出す一方、Transformerのparameterはfp32であり、既存のdtype完全一致検査がこの正当なautocast入力を拒否した。学習更新前の失敗で、returncode=1とDrive保存を確認した。failed-startup.log / failed-startup-status.jsonに証拠を残す。

c4b12f923では、fp32 parameter・device autocast有効・指定autocast dtypeの入力という組み合わせだけを許可した。演算やcastを追加せず、Lが通ったdtype一致経路も変更していない。Lからの学習コード差分はこのguardだけで、autocast-fix.patchに保存した。CPU bf16で射影ありの2ケースが修正前に同じTypeErrorで失敗し、修正後はモデル/Transformerの31テストが成功した。射影・Transformerの有限かつ非zeroなgradient、出力、dtype不正入力の拒否を含む。CPUテストだけでCUDA実機の成功とはみなさない。

同じVMへGit経由で修正を反映し、train-s-s42-attempt02を新しい出力`.../s42/attempt-02`へ提出した。初回のconfig/logを保持し、resumeを使わず同じseed42から20epochを学習する。初回と再試行の解決済み設定の差分は出力先だけだった。初回SとLの設定差分もbackbone名・weight・取得層・保存先だけで、データ・loss・後段容量・学習条件は一致する。両比較の実際の差分をcondition-comparison.jsonとattempts/02-condition-comparison.jsonに記録した。

実処理はNotebook kernel経由で実行する。共通runnerのArtifactStoreでcheckpoint保存直後と60秒ごとのDrive保存を行い、起動時にDrive上のconfigがVMとbyte一致することを確認した。修正後はL4実機でsanity validationの2batchを完了し、Epoch 0の80/1658まで学習更新が進んだ。表示lossは有限で、射影を含む混合精度forward/backwardの実行を確認した。これは起動障害の解消確認であり、全学習や精度の評価ではない。このノードは再試行中の記録で、完了や精度はまだ報告しない。TensorBoard曲線と最終metricsは全20epochと正式評価の完了後に追加する。

checkpoint世代を自動削除せず保持し、全20epoch後にvalidation loss最小の保存checkpointを選ぶ。合成test886件と実画像val2211件を別に評価し、test_after_fitの終端モデル評価とも分ける。合成testをcheckpoint選択には使わない。実画像valは選択にも参加するため独立testとしては扱わない。中断が生じた場合はattemptと復旧元を記録し、bitwise継続を仮定しない。

Lだけではサイズ間の優劣を決められない。S完了後にS+/Bを同じguard修正版と条件で揃え、固定した同じ画像・評価指標・後段容量で4条件を比較する。過去のLoRAありBの異なる条件は混ぜない。
