# 系列長条件のデスクトップ継続

データ・固定環境・実GPU更新の継続可否は確認できた。本番は実装worker runtimeの
終了待ちで、まだ開始していない。最初の周期checkpointと後半開始による起動受入は未完了。

対象は [#1899](https://github.com/Motoki0705/tennis-lab/issues/1899) の③だけ。
①・②の12k学習と旧基準は再実行しない。固定実験sourceをdetached worktree
`ball-query-local-runtime` に保持し、移管用コードだけをこのdirectoryに置く。

## 入力と移管の契約

最新の検証済みColab世代は `length-s1-T64/checkpoint-00005500-0281b1ba`。
8 payload、150,525,901 bytes。前半の残り500更新、後半6,000更新を継続する。
sample位置15,500、AdamW 201件（CNN116件step14,900、query85件step5,500）、
Python・NumPy・Torch・単一CUDA RNGを保存している。

配布版は元の720p storeからtestと128frame未満のclipを除き、再indexしている。
元storeはtrain 1,370 clips / 762,004 frames、val 200 / 144,449、test 230 / 159,496。
配布版はtrain 1,342 / 759,799、val 195 / 144,008。残る33本のtrain/val除外は128frame未満。
配布版で変わるclip metadataはindexとannotation_pathだけで、注釈の内容・split・FPS・画像サイズは一致する。

`rebuild_dataset.py` は固定配布manifestを認証し、元storeの対応する1,537 shardを
全件SHA照合してからhardlinkする。RGB 156,770,864,484 bytesを再利用する。
注釈1,379本・inputs7本・index・metadataの追加payloadは257,891,063 bytes。
`verify_dataset.py` は注釈などのSHAと全ファイル集合を確認する。
2,925 payloadの合計は157,028,755,547 bytes、distribution自体を含むファイル集合は2,926本。
元store・既存Colab出力は変更しない。

凍結manifestの共通native spanではtrain 664,349 frames、val 128,668 frames。
位置教師はobserved-onlyで、val 85,473 frames。masked・reviewed_empty・out_of_frameを位置誤差の分母に加えない。
同じ128frame pack・順序・拡張を旧基準の全学習行と照合し、T64の2窓に分ける。

Python 3.11.17、lock固定のPyTorch 2.14.1+cu130 / torchvision 0.29.1+cu130を専用envへ復元する。
共有venvは変更しない。GPUは実名でRTX 5060 Tiを検査する。
L4からの移管なのでGPU演算のbitwise同一性は保証しない。
BF16、default compile、nvJPEG、image prefetch、4 workers、128frames/update、
microbatch_windows=1、seed42、モデル・teacher・augmentation・AdamW・LR曲線を保つ。
各段600更新までCNN凍結、解除後のCNN LRはheadの0.1倍、offsetは0→6,000。

`desktop.py prepare` は元manifestのdata rootだけをローカルへ移し、派生した2つのmanifest digestと
checkpoint recipeの対応3fieldをコピー上で変更する。元checkpointは保持する。
全model・AdamW・schedule・RNG・sample・残るmetadataの再帰一致を確認し、`relocation.json` に記録する。
固定training source・controller・lockは全件照合し、historical runner自体を変更しない。

## 検証と資源

CPUの検証設計は [test-design.md](test-design.md)。固定版の10ケース、adapterの7ケースを実行した。
固定版で初回skipだったrclone publicationは実rcloneを用意して単独再実行し成功した。
数値モデル・段境界・中断再開は既存テスト、path変更の限定と未知root拒否は追加テストで確認する。

実GPU probeは別directoryに復元して5,501・5,502更新を実行する。
PyTorch allocatorの上限80%と、host available RAM 4 GiB未満での停止を設ける。
`TORCHINDUCTOR_COMPILE_THREADS=1` はコンパイルの並列数だけを制御する。
本番のrecipeにmicrobatch縮小・activation checkpointing・precision変更を加えない。
probe失敗時は自動再試行しない。

実測値と最新状態の正本は、割当worktree内の
`outputs/ball_detection/train/query_controls/desktop-resume-20261112/`。
`progress.json`、`dataset-verified.json`、`relocation.json`、`probe-acceptance.json`、
`production-status.json`、`production-host-memory.json`、`production/readback-receipts/` を参照する。

実probeでは5,501・5,502更新が成功し、両行のpack・教師・拡張・LRが旧基準に一致した。
model・AdamW・RNGを完全復元し、loss・勾配も有限。allocated peak 10,581,860,352 bytes
（9.86 GiB）、reserved peak 11,777,605,632 bytes（10.97 GiB）。
host availableの最小値は17,317,019,648 bytes（16.13 GiB）、trainer自身のpeak RSSは約3.82 GiB。
最初の更新はcompile込み159.57秒、次は0.798秒（160.45 frames/s）。
この1つの定常更新からの単純見積りは残り6,500更新で約86分だが、
固定validation・再検証・compile・保存時間は別に必要。最終速度や完了時刻を保証する測定ではない。

## 実行と状態確認

次の変数をそれぞれの実パスへ設定する。`ROOT` は上記出力、`SOURCE` は固定版のdetached worktree、
`ADAPTER` はこのdirectory、`QUEUE` はメインrepoの共有 `.training_queue`。

```bash
# cwdはSOURCE。準備は一度だけ。既存productionへの上書きを拒否する。
CUDA_VISIBLE_DEVICES='' "$ROOT/env/bin/python" "$ADAPTER/desktop.py" prepare \
  --source "$SOURCE" --root "$ROOT"

# probeと本番は共有queueのresource allからだけ実行する。
# probeが成功するまでproductionは登録しない。正確な投入コマンドは各registration.jsonに保存。
TRAINING_QUEUE_DIR="$QUEUE" bash "$QUEUE_CLI" status
cat "$ROOT/progress.json"
cat "$ROOT/production-status.json"
cat "$ROOT/production/length-s1-T64/latest.json"
cat "$ROOT/production/length-s2-T64/latest.json"
systemctl --user status tennis-i1899-length-desktop-a1.service
journalctl --user -u tennis-i1899-length-desktop-a1.service -n 30 --no-pager
```

本番queueのserviceはsystemd user managerが所有する。単なる`queue start`だけでは
共通agent runtimeのdetached descendant cleanupに巻き込まれるので、本番には使わない。
`service.py` はqueue終了をblockして待ち、完了・失敗・プロセス終了のterminal eventだけを
`notifications/` へ保存しDriveへ検証転送する。systemdのExecStopPostでも異常終了を記録する。
安定後の進捗pollingや定期follow-upは登録しない。

今回のprobeでは、終了済みPyTorch resource trackerが実装workerのsubreaperに引き取られ、
runtimeが会話ターン末尾でreapするまでkernel PGIDが残った。queueは
`terminating / process-group-teardown` のまま安全にallocationを保持している。
この状態でlock・履歴を消して本番を開始しない。
systemdの`service.py wait-and-serve`が当該runtimeの終了をpidfdイベントで待ち、
旧probe PGIDの消滅を証明してから本番を共有queueへ登録する。
`deferred-launch.json` が待機／登録の正本、`production-registration.json` が実際のqueue ID。
登録失敗や残存PGIDでは停止し、自動再試行しない。
このworkerターン内では本番開始と安定条件は未確認であり、親が同じworker sessionへ
継続指示を送り、本番の最初の更新・6,000保存とreadback・後半開始を照合する必要がある。

保存先は既存Colab履歴と別の、Drive root相対
`outputs/ball_detection/train/query_controls/s42-20261111-main/desktop-attempts/20261112-a1/`。
500更新ごとの不変snapshot、各6k/12kの固定validation、全世代を保持する。
各checkpointをDriveから別directoryへ復元し、full stateを元と再帰比較する。
最初の前半6,000更新の保存・独立readback、後半の開始までを起動の安定条件とする。
最終評価では85,473 observed / 128,668 framesのidentityを旧基準と照合する。

失敗・VM/host消失後は自動再開しない。最新のverified receiptを選び、既存writerの不在、
source・入力・状態・原因を確認したうえで、新attemptへfull-stateを復元する。
`desktop.py` の初回入口は意図的に再実行を拒否するので、既存attemptを消して使い回さない。
merge・Issue close・全4条件比較の確定はこの起動作業には含めない。
