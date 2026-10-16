# 注釈ワークフローの接続

`python -m src.tasks.annotation` は、共通基盤とball/playerのadapterを接続する入口。
責務分担と最終構成は[アーキテクチャ資料](../base/annotation/README.md)を正本とする。

現在実装した範囲は**preview campaignの取得→プレー窓選別→clip準備→ball/player予測**。
未レビュー予測から教師採用やdataset公開は行わない。定性確認後のreview policy、
公開・追加学習・モデル切替の接続は後続工程として残る。

`Runtime` は固定plan・資源scope・queue script版・Codex保存先・工程別のPSS/保存量/
timeout/GPU時間を参照する。実験固有のURL、件数、モデルpathは設定artifactに置き、
ソースへ埋め込まない。全設定とreceiptはmain repositoryの `outputs/chat_annotation/` に置く。

```bash
.venv/bin/python -m src.tasks.annotation register --runtime /absolute/runtime.json --sha256 HASH --candidates /absolute/candidates.json
.venv/bin/python -m src.tasks.annotation run --runtime /absolute/runtime.json --sha256 HASH --max-seconds 600
.venv/bin/python -m src.tasks.annotation status --runtime /absolute/runtime.json --sha256 HASH
```

上記pathは固定annotation root内の実ファイルを指定する。相対path・外部path・hash不一致を
拒否する。`CandidateManifest` はcanonical source IDと根拠参照の一覧。
過去のsource/channelは、収集前に `HistoryManifest` を持つ `HistoryOperation` を
`submit --operation PATH` で登録する。完了receiptを確認してから新規候補を登録する。

`run` は台帳と不変receiptから再開する。終了時間はcontrollerの実行時間であり、
既に受付済みのsupervisorを強制終了する指定ではない。再開も同じruntimeを使う。
`cycle` / `status` はsourceの状態にかかわらず登録済みworkerを照合する。
選別がsourceを更新した直後に中断しても、workerの完了receiptと台帳更新を回収する。
sourceの先行数とCPU/Codex/GPUの枠を守り、GPUは共通queueへ投入する。
queue workerの起動は[training-queue skill](../../../.agents/skills/training-queue/SKILL.md)に従う。

screening workerの追加画像要求は、workerの書込領域から親controllerへ渡す。
親がrequest/attempt/区間を検証し、CPU工程で描画してから不変の画像参照を返す。
受付済み要求は親所有のSQLite台帳にattempt単位で累積保存し、固定上限を判定する。
inboxの並び順・削除・親の再開で枠を戻さず、同一要求の回収で二重計上しない。
workerにソース動画の全量decodeや共有cacheへの書込みを任せない。採用時は実際の
rollout画像提示と選別結果を照合する。単なる画像生成や自己申告では選別を完了しない。

予測は新規clipだけをtask別に束ね、1工程で同じモデルinstanceを共有する。
失敗した試行を自動で繰り返さず、状態・理由を残す。`submit --attempt N` は
明示的な新試行であり、通常の再開では最新の明示試行のreceiptを回収する。
資源上限のstop解除は原因解消後の `resume-budget --reason TEXT` を使う。
scopeや累計を初期化して上限を回避しない。

`stage` subcommandはsupervisor専用で、固定した入力digestとの対応を要求する。
通常操作の入口にはしない。旧ChatGPT Project/QA入口の削除は、本番レビュー・公開を
この入口へ接続する段階で行う。既存キャンペーンや保存成果物はここでは変更しない。
