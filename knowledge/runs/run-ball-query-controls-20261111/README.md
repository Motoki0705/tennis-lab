# 旧Transformerに対するquery・CNN・前半系列長の3条件比較

科学条件とこのColab Codexの担当範囲は [#1899本文・2026-11-11の実行指示](https://github.com/Motoki0705/tennis-lab/issues/1899) を正本とする。
専用worktreeで固定した旧実験コードを使い、#1675の8層の残りを完了してから3条件を直列実行する。12/24層は登録しない。#1859のVM・キューは操作しない。

| 順番 | 条件 | query更新 | 初期CNN |
|---|---|---:|---|
| 前提 | #1675 dim512・8層 T64の後半 | 保存済み4kから残り2k | そのモデルのfull checkpoint |
| ① queries | ConvNeXt V2＋4本query＋mHC、T32→T64 | 6k＋6k | #1514 common DPT T32・10k |
| ② cnn | X3D/CSN＋1本query、T32→T64 | 6k＋6k | #1675 common DPT T32・10k |
| ③ length | ConvNeXt V2＋1本query、T64→T64 | 6k＋6k | ①と同じfull DPT checkpoint |

①〜③はdim256、4層、8 heads、FFN704、seed42、RGB＋MDD 5ch、720p、128frames/update。
T32は4窓、T64は2窓に分ける。各段600更新までCNNを凍結し、その後のCNN LRはheadの0.1倍。
queryの12k cosine曲線は旧基準のDPT終端LRから最終1e-6へ減衰し、offsetはTに依存せず段階順に0→6000とする。
後半は全モデル重み・named AdamW・sample位置を継承する。
新しい段境界のRNGは旧#1514のseed42による構築順を保ち、同じ段の中断再開ではPython/NumPy/Torch/CUDA RNGをfull checkpointから戻す。
#1675の8層は保存時の継承方針も含めた元コードのまま再開する。

## 固定した実装と検証

8層はDriveの検証済みGit bundleのacceptance commitから復元した `ball-capacity-12k` worktreeを使用する。
新条件はそれに基づく `ball-query-controls` worktreeで実行する。現在のmainとはデータ実装も違うため、mainのコードを混ぜない。
`plan.json` が実行commit・全training source・controller・lock・モデル・予算・checkpoint来歴を固定する。
変更はmHCへのConvNeXtと任意の有効head幅の許可、および同一CNNのDPT T32からfresh query T32/T64への初期化である。
#1892の移植PRを前提にしたレビュー用差分と、実験の固定source bundleは区別する。

`verify_initialization.py` は旧#1514の実行コードを別processで実行し、新しい1本queryのT32/T64初期値と全tensor・Torch RNGを比較する。
`control.py prepare` は2つのDPTについて学習予算・runtime・拡張・データ・decoder契約と10k full stateを照合する。
`train.py` は初期AdamW転送を名前で照合し、500更新ごとの保存前に全行の入力pack順・拡張hash・教師数を旧基準と照合する。
全条件でfiniteなloss/勾配/LRと600/601更新の凍結境界を検査する。
③の両段でも開始行・600行・601行・最終行を `acceptance.json` に残す。

新組合せのCPU検証は `tests/integration/tasks/ball_detection/test_query_control_training.py`。
実モデル・実AdamW・128枚組・3更新×2段で継承、freeze/LR、T64初期化、中断再開の完全一致を確認する。
既存の逐次学習とmHCの数値テストを再利用する。新3ケースのcallは5〜9秒程度で、実runnerの2段と独立した中断・再開を比較するためintegrationに維持する。
`test_control.py` は学習条件、checkpoint選択、不一致拒否、実rcloneによる比較表の公開を確認する。
GPUの実行可否・安定性はCPUテストから推定せず、本番の最初の周期保存とCNN凍結解除後の実ログで受け入れる。

## 保存先と実行

workspaceは `/content/tennis-lab`。①〜③のcwdは `.claude/worktrees/ball-query-controls`。
Driveの `tennis_lab` とworkspaceから同じ相対pathの
`outputs/ball_detection/train/query_controls/s42-20261111-main/` を使う。
8層の元run・既存成果物は `capacity_scaling/s42-20261110-12k/` に保持する。

`inputs/inputs-v1/` はfull Git bundle、復元用コード、lock、固定plan、初期化照合、事前学習来歴、親checkpoint receiptを含む。
`inputs-receipt.json` を根拠にmarkerと全payloadを照合する。
学習中は500更新ごとの不変snapshotとreceipt、各6k/12kのcheckpoint・固定validation・フレーム別予測をDriveへ保存し、世代を削除しない。
元data、入力checkpoint、既存runを削除・移動しない。ローカルもこの3条件の全checkpointを保持する。

以下は初回投入手順。準備・保存が既に完了しているrunでは再生成しない。

```bash
# cwd: /content/tennis-lab/.claude/worktrees/ball-query-controls
.venv/bin/python knowledge/runs/run-ball-query-controls-20261111/verify_initialization.py
.venv/bin/python knowledge/runs/run-ball-query-controls-20261111/control.py prepare
PATH=/tmp/ball-tools:$PATH .venv/bin/python knowledge/runs/run-ball-query-controls-20261111/persist_preparation.py
# 固定dataの全件照合・data-verified.jsonの作成後にだけ投入する。
.venv/bin/python knowledge/runs/run-ball-query-controls-20261111/control.py launch --session "$CODEX_THREAD_ID" --attempt 1
.venv/bin/python knowledge/runs/run-ball-query-controls-20261111/start_service.py
PATH=/tmp/ball-tools:$PATH .venv/bin/python knowledge/runs/run-ball-query-controls-20261111/verify_startup.py
```

共有queue stateはworkspace直下の `.training_queue/`。source bundle内のqueue runtimeで全4件をresource all・冪等key付き登録する。
serviceは起動スクリプトの終了後にPID1配下となり、queueとtrainerを保持する。完了・失敗の通知はinotifyで検知する。
失敗時はGPUのall lock解放前にqueueのstopと失敗条件・最新durable stateを記録して後続を保留する。自動再試行・skip・weights-onlyへの変更を行わない。

`verify_startup.py` は①の601更新以降の有限受入である。最初の500更新checkpointをDriveから別directoryへ復元し、
全payload、model、named AdamW、RNG、sample位置、log prefixとコードを照合し、CNN凍結解除後のfiniteなloss/勾配/LRと独立serviceの生存を確認する。
これが終わった時点で対話担当は終了できる。全3条件の研究完了を意味しない。

## 新VMでの復旧

最初に既存GPU process・queue・writerの生存を調べる。稼働中なら二重起動しない。
Driveを `/content/drive` にmountし、uvとGitが使える**新VM**で次の明示コマンドを実行する。
`--session`にはその作業を担当する実際のCodex session ID、`--attempt`には未使用の新しい試行番号を渡す。
新しいVMをこのコードが自動確保することはない。

```bash
python /content/drive/MyDrive/tennis_lab/outputs/ball_detection/train/query_controls/s42-20261111-main/inputs/inputs-v1/reproduction/bootstrap.py \
  --session "$CODEX_THREAD_ID" --attempt 2 --launch
```

bootstrapは自身と入力markerを照合し、Git bundleから2つの専用worktree、Python3.11.17・lock環境・rclone、
事前学習checkpoint・旧基準・固定dataを復元する。`restore.py` は8層と開始済み条件だけの最高更新を選び、同更新ではcompletedを優先する。
不正な最新世代、重複、初期化だけの保存、既存ローカルworktreeとの衝突は停止し、古いcheckpointや新規初期化へ黙って戻さない。
既存FAILEDがあれば原因の診断と明示的な履歴保全後まで再投入しない。
同VMでの再開は、稼働process・未保存末尾・最新receiptを確認してから `restore.py` と新attemptのlaunchを個別に使う。

## 評価と解釈

各段のvalidationは全128,668フレーム、位置主指標はobserved85,473フレーム、元画像pixelの距離で採点する。
全条件T64の最終reportを旧基準と比較し、target/teacher/scored ID、snapshot、observedの実PTS付きIDを照合する。
`comparison.json` と `comparison.md` は全4条件のmean・median・P95・20px以内率を並べる。
誤差差分はpx、率の差分はpercentage points。未完了条件を実測値で埋めない。
全条件のstage・Drive保存・ID照合が完了して初めて研究の比較を確定する。

①はquery数とmHCを合わせた効果で、mHCだけの因果効果を示さない。②はこの事前学習を含むCNN置換、③は等枚数での前半T変更の効果である。
seed42の各1回・固定validationで、交互作用・seed分散・独立testはこの実験では特定しない。
改善と悪化は4指標ごとに記録する。採用モデルは変更せず、研究issueも自動closeしない。
