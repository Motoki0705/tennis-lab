# #1675 追加12k実験の投入と起動安定確認

科学条件と受入条件の正本は[2026-11-10追加実験の合意](https://github.com/Motoki0705/tennis-lab/issues/1675#issuecomment-6758820955)。
この有限workerは既存ColabのL4一台を使い、9モデルを投入して第一モデルの601更新以上と最初のDrive checkpointを検証する。
全9モデルの完走・比較・採用判断は後続作業とする。子issue、追加VM、GitHubへのworkerコメントを作らない。

## 固定条件

順序はd256-l4、d384-l4、d512-l4、d768-l4、d1024-l4、d512-l6、d512-l8、d512-l12、d512-l24。
既存7モデルは各6k full checkpointから追加6k。最後の2モデルは共通DPT10kから各ジョブ内部でT64前半6k→後半6kを連続実行する。
計9ジョブ、11段、追加66,000更新。全段seed42、T64、RGB+MDD 5ch、128frames/update、固定snapshot/split/拡張を維持する。

各段の更新数6000、checkpoint間隔500、evaluation間隔6000、CNN凍結600更新、解除後encoder LR比0.1。
AdamWを名前対応で継承する。LRはlinear_warmup_cosine、peak/initial 7.360330616529733e-05、minimum比0.0135863462132287、
horizon12000、warmup0。既存継続・新規後半はoffset6000、新規前半だけoffset0。
compile default、BF16、nvJPEG、microbatch1、activation checkpointing=false、frame_chunk_size=null、allocator overrideなし。
OOM、不互換、破損、Drive保存失敗をweights-onlyや別構成へ自動変更しない。

元のmodel/training実装162ファイルとlockは、元inputs snapshotとGit bundleから復元して照合した。
現在mainの新しいデータ実装を混ぜないため、専用worktreeの対象source集合を元版へ戻した。
Torch 2.14.1+cu130、torchvision 0.29.1+cu130、Python3.11.17を使用する。main worktreeのtrackedファイルは変更しない。
`source-provenance.json`、`plan.json`、`parents-verified.json`が具体的なidentityと対応表を持つ。

固定runnerは段境界でモデル・AdamW・sample位置を継承するがRNGを再seedするため、`train_model.py`は同じモデルの継続時に限り
最初のbatch直前でfull checkpointのPython/NumPy/Torch/CUDA RNGを復元し、全named AdamW状態と転送model tensorを照合する。
新規12/24層のDPT→queryでは、旧7モデルと同じseed42による新head初期化を維持する。
このadapterは固定sourceを書き換えず、`continuity.json`をcheckpointとともに保存する。

## 配置と起動

workspaceは`/content/tennis-lab`、cwdは`/content/tennis-lab/.claude/worktrees/ball-capacity-12k`。
出力はworkspaceおよびDrive rootから同じ相対pathの
`outputs/ball_detection/train/capacity_scaling/s42-20261110-12k/`。
共有queueは`TRAINING_QUEUE_DIR=/content/tennis-lab/.training_queue`。
全9件はresource allで登録し、共通queueのcapacity lockとcampaign単一writer lockで直列化する。

親が開始した元downloadを重複起動しない。`download-startup.log`・`startup.json`・`progress.json`・`files.json`を保持する。
学習前に2,925 payload、157,028,755,547 bytesの全件dual SHA・size・manifest・membershipを検証する。
8入力full checkpointはreceipt→marker→全payload→checkpoint構造・予算・model/source/optimizer/RNG/sample位置を照合する。
prepared manifestは元snapshotの6ファイルを復元し、再生成しない。

```bash
# cwd: 専用worktree。復元・検証はGPU学習を開始しない。
PATH=/tmp/ball-tools:$PATH .venv/bin/python -B knowledge/runs/run-ball-capacity-12k-20261110/restore_parents.py
.venv/bin/python -B knowledge/runs/run-ball-capacity-12k-20261110/campaign.py prepare
# 実験ファイルをcommit後、独立復元できるbundle/configを先にDriveへ保存。
PATH=/tmp/ball-tools:$PATH .venv/bin/python -B knowledge/runs/run-ball-capacity-12k-20261110/persist_preparation.py inputs
# データ完全性確認後、この単一コマンドで9件すべてを冪等登録する。
.venv/bin/python -B knowledge/runs/run-ball-capacity-12k-20261110/campaign.py launch --session "$CODEX_THREAD_ID" --attempt 1
```

`queue-registration.json`は全9 ID、順序、各段予算、親checkpoint/hash、LR、出力を保存する。
登録完了後だけ`queue-service-request.json`のready=trueを公開する。
親はそこに保存したcwd/argv/env/logで、有限workerの祖先process treeの外からserviceを起動する。
worker自身はdaemon化しない。serviceは自身の祖先を検証し、要求をready=falseへclaimする。

```bash
# 親が外部service起動後、workerが有限の起動受入を実行。
PATH=/tmp/ball-tools:$PATH .venv/bin/python -u -B knowledge/runs/run-ball-capacity-12k-20261110/verify_startup.py
.venv/bin/python -B knowledge/runs/run-ball-capacity-12k-20261110/campaign.py status
```

受入は601更新以上、全loss/grad/LR finite、600/601のfreeze境界、連続global/sample位置、段境界の全状態継承を確認する。
500更新checkpoint・ログ・continuityをDriveから新directoryへ独立restoreし、receipt/marker/全payload/hash/内容を検証する。
`startup-acceptance.json`に実測、8 queued、service/worker/trainingのPIDと祖先、GPU processを残す。
起動安定後は終了し、全9モデルの完走まで監視しない。

## 永続化・停止・通知

各500更新および完了時に、runnerを同期停止したままimmutable generationを`RcloneSnapshotStore`でDriveへ公開する。
receiptは`persistence/<stage>/`、各段の状態は`stage-status/`、各モデルの状態は`job-status/`。
Driveの全checkpoint世代を保持する。元campaignと8入力を変更・削除しない。
236GiB VMの容量内で全11段を実行するため、この新runだけは最新のcheckpoint/weightsをローカルに残す。
より新しい世代のDrive保存後、前世代のmarker・元の検証receipt・local hashを照合して前世代のローカル複製を解放し、
`local-cache-receipts/`へ記録する。Driveのpayload、receipt、markerは削除しない。各段の最終checkpointはローカルにも残す。

例外時はGPU all lockを解放する前にqueue stopと`FAILED.json`・`notifications/failed-*.json`を公開して非zero終了する。
後続は前件完了を検証し、勝手に学習開始・skipしない。serviceもinotifyでdone/failed/cancelledを検知し、通知をDriveへ独立保存する。
定期的な進捗pollingは行わない。serviceの終了記録と`queue-service.log`も引継ぐ。

## 同じVMまたはVM消失後の再開

1. 旧service・queue worker・GPU processの生存と所有を確認する。生存中は再起動しない。新VMの自動確保は行わない。
2. 最新の`documentation/<generation>-receipt.json`と、新runの`inputs-receipt.json`を取得する。
   `RcloneSnapshotStore.restore`で新runのinputsを新しいscratch directoryへ検証付き復元する。
   その`source.bundle`を`git bundle verify`→安全なrefへfetchし、同じ専用worktreeを復元する。
   documentationにincremental bundleがあれば、`restoration.json`のprerequisite commitを満たしてから検証・fetchする。
3. Python3.11、保存lockの`uv sync --locked`、rclone v1.75.2、mountを指すprivate alias設定を復元する。
   元campaignのinputsを元receiptから`/tmp/ball-capacity-inputs-v1`へrestoreする。
   同じ絶対pathへdatasetを取得・全件検証し、上の`restore_parents.py`で8入力とpreparedを戻す。
   新inputsのplan/controlを元の新run出力へ一致確認付きで配置し、最新documentationのqueue登録を戻す。
4. `recover_campaign.py`を実行する。各段の最高更新の検証済みreceiptを選び、同更新ではcompletedを優先する。
   破損・曖昧さ・initializedしか無い段・ローカルとDriveの食い違いは停止する。無断で古い世代やscratchへ戻さない。
5. 原因を解決した上で、旧workerのteardown完了後に`campaign.py recover --session <新workerのsession> --attempt 2`。
   旧登録とFAILEDを履歴へ残し、このcampaignの未開始8件だけをcancelして新attemptの9件を順序どおり登録する。
   完了段は設定・checkpoint・validationを検証して計算を繰り返さず、途中段はfull checkpointからstrict resumeする。
   親が新しいservice要求を外側から起動する。以後も復旧起動と最初のdurable checkpoint確認は有限Codex子へ委任する。

再開で変えないものはmodel、AdamW、LR schedule、RNG、sample位置、元snapshot/split/拡張、Torch版。
未保存の末尾更新は最後の検証済みcheckpointから再計算する。全9完走と最終評価は別途確認し、issueをcloseしない。

## 検証記録

モデル・逐次学習の既存CPU回帰テスト22件と、実queueの冪等9件登録／失敗停止・named AdamW照合とlocal cache保護の5件を通過。
具体的な実測、有限workerの進捗、最終受入、bundle、documentationのreceiptは新run出力を正本とする。

起動受入は2026-11-10 05:31 UTCに完了した。第一モデルの601更新（query累積6601、DPTを含むglobal16601）を確認した。
500更新の8 payloadをDriveから独立復元し、checkpoint・ログ・全状態継承・600/601の凍結解除を検証した。
具体的な実測値、9 queue ID、receipt、プロセス、残る範囲は[startup.json](startup.json)に記録した。
GPU学習は外部service配下で継続し、後続8件の完走・評価や12/24層の実行可能性はこの起動受入に含めない。
