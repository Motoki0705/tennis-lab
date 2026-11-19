---
id: run-court-scene-pair-execution-s42
type: run
task: court_detection
sequence: 56
recorded_at: '2026-11-12'
title: Court全5ペア学習と9条件共通評価の完了・再開来歴
issue:
- 1585
- 1586
provider: claude
date: '2026-11-09'
status: done
config:
  run.seed: 42
  updates: 33160
  new_training_conditions: 5
  evaluated_conditions: 9
  runner_outputs: 8
  persist: 0
metrics:
  completed_stages: 16
  evaluated_conditions: 9
  new_training_conditions: 5
repro:
  commit: 17ad301bf8d80a5d094f28bb48f3431c7d9b5d89
  command: env TRAINING_QUEUE_DIR=/content/.tennis-colab/court-scene-pair-20261110-recovery5/queue
    TRAINING_QUEUE_LOCK_FILE=/content/.tennis-colab/court-scene-pair-20261110-recovery5/control/gpu.lock
    PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=1 COURT_RECOVERY_DRIVE_CLI=/content/.tennis-colab/court-scene-pair-20261110-recovery5/control/drive-cli/drive.py
    COURT_RECOVERY_ADAPTER_SHA256=3dd67698ff56d2d217cc8b1deff81105942336d49939b7f8d559ba4aeb8b6c28
    COURT_RECOVERY_OPERATIONAL_REF=113cd45b894ae19ba964188ef04838de6db95e5b COURT_RECOVERY_EVIDENCE_DIR=/content/.tennis-colab/court-scene-pair-20261110-recovery5/control
    python3 -u -B /content/.tennis-colab/court-scene-pair-20261110-recovery5/control/queue_service.py
    -- bash .agents/skills/training-queue/scripts/training_queue.sh serve --idle-timeout
    3600
  cwd: .
artifacts:
  run_dir: knowledge/runs/run-court-scene-pair-execution-s42
  drive_record: operations/colab/court-scene-pair-20261110-recovery5/court-campaign-service-20261110-r6-nb3
  drive_outputs:
  - outputs/court_detection/train/scene-pair-5500-s384-colab-probe/P02-s42
  - outputs/court_detection/train/scene-pair-5500-s384-colab/P02-s42
  - outputs/court_detection/train/scene-pair-5500-s384-colab/P03-s42
  - outputs/court_detection/train/scene-pair-5500-s384-colab/P12-s42
  - outputs/court_detection/train/scene-pair-5500-s384-colab/P13-s42
  - outputs/court_detection/train/scene-pair-5500-s384-colab/P23-s42
  - outputs/court_detection/evaluate/scene-pair-5500-s384-colab
  - outputs/court_detection/campaign/scene-pair-5500-s384-colab
parents:
- run-court-scene-pair-manifests-v1
relations:
- to: group-court-scene-diversity-colab-s42
  rel: related
  note: 旧4条件のcheckpointを保持し、新5ペアと共通評価を加える
papers: []
tags:
- execution
- colab
- resume
- seed42
summary_impact: none
---

## 完了したcampaignと最終管理job

新5ペアの学習と9条件の最終checkpoint評価が全16段階で完了した。campaignのcomplete/completed公開は2026-11-10 04:37:28 UTC、最終管理jobはcompleted/returncode0、実Drive保存は04:55:25.769262 UTC。科学writer終了とqueue done（固定scriptではexit0の場合だけdone）を親が確認し、空queueの協調終了後に所有Court sessionを停止した。[完了抽出](../../runs/run-court-scene-pair-execution-s42/final-acceptance-summary.json)と[原validatorの親監査](../../runs/run-court-scene-pair-execution-s42/report-audit.json)を参照。

このColab routeで取り込んだrequest/status/source snapshotは最後の復旧管理jobのもの。既存8評価を保持し、P23を29,844→33,160へ復元して完了させた。全条件をこのjobで最初から再学習したわけではない。P23のscheduled validationはTensorBoard33159＝完了33,160更新、val/loss 4.684573650360107。最終checkpointは536,530,037 bytesで元のsaved recordと実Drive内容が一致した。[P23復元・最終保存抽出](../../runs/run-court-scene-pair-execution-s42/p23-resume-final-summary.json)。旧Notebook提出2件はworker起動前のpendingでcancel/保存済みという運用履歴であり、科学stage失敗として数えない。

## 科学・運用sourceと再現

科学source、運用adapter、host Notebook transportは別々のrefとして完了抽出に記録した。モデル・入力・数値recipeの変更を運用復旧に混ぜていない。元source snapshot/request/statusはkg_from_colab.pyで取り込み、captured patchとexact argvを保持した。Driveに残るraw評価は約574KBずつでGitの上限を超えるため複製せず、小さい抽出だけを比較bundleへ置く。

kg_from_colab.pyが生成した歴史的reproは`colab-repro.sh`へbyteを変えず保持し、元のchecksumをcontrol-sources.jsonに記録した。新しい`repro.sh`は、元VMのcontrol以下にしかなかったqueue_serviceとDrive CLIを、実deployment manifestに一致する保存sourceから配置して元reproを呼ぶ準備wrapperである。元requestは空serviceの起動で、科学workloadの登録は別操作だった。`workload-command.json`に元のcommandを残す。再実験は今回の範囲外であり、これらの再現入口を今回実行していない。

元campaignは全5入力/pose・全9実checkpoint payload・設定・評価37成果物をDriveに照合してから各完了印を公開した。親は85ファイル/8,316,576bytesのcampaign状態と9 raw評価を取得し、元Python3.11 validator、checkpoint SHA/33160、ID/順序/件数・有限値・4scene平均、全5設定guardを確認した。このCPU側確認を全checkpointの再downloadと混同しない。full-state復元はoptimizer/scheduler/loops/RNGを含むが、mid-epochのsampler列を厳密には保たない。未保存末尾のactual_last_updateはnullのまま。restart全9記録（開始0・probeを含む）は比較bundleのsources.jsonから元へ辿れる。

このnodeは実行・証跡のrunでfit曲線を持たない。新5条件の受入履歴は各training node、旧4条件の曲線は元training nodeに保持する。summary_impact: none。実行完了を支える来歴で、採用checkpointは変えない。比較による現在判断はgroupで更新する。
