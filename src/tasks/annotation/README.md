# 注釈ワークフローの接続

`python -m src.tasks.annotation` は、共通基盤とtask adapterを接続する入口。
責務分担と最終構成は[アーキテクチャ資料](../base/annotation/README.md)を正本とする。

preview campaignでは取得→プレー窓選別→clip準備→ball/player予測を実行する。
保存済み予測の定性確認には、以下の `prediction_preview` を使う。
production campaignでは、現在 **ball / framesのみ**、レビュー→全clipの採用確認→
累積snapshot公開→追加学習・固定評価・モデル採用まで接続している。
player bbox/poseを含むproduction planは明示的に拒否する。実装状態と実動画検証の進捗は
[親Issue](https://github.com/Motoki0705/tennis-lab/issues/1088)で管理する。

```mermaid
flowchart LR
    Predict["Predictions.completed\n保存済み予測を検証"] --> Review["BallReviewWorker\n画像・編集は親サービス経由"]
    Review --> Close["CampaignReview\n要求動画数と全clipの採用を照合"]
    Close --> Publish["publish_ball工程\nBallPublisher"]
    Publish --> Completion["CampaignCompletion\n公開receiptと範囲を照合"]
    Completion --> Learning["Learning.register / tick\nBallLearningAdapter"]
    Learning --> Model["固定評価・採用判定\nModelRegistryの版を照合して更新"]
```

`reviews.py` がcampaign全体の採用範囲を、`production.py` が公開と学習の接続を所有する。
base内部には具体taskの分岐を置かない。SQLiteのworkflow進捗は表示用cacheであり、
再開時は不変の要求・実行・公開receiptを照合する。

## 固定入力と起動

`Runtime` はplan・資源scope・queue script版・Codex保存先・工程別のPSS/保存量/
timeout/GPU時間を参照する。設定とreceiptはmain repositoryの `outputs/chat_annotation/`
以下へ置く。`schema --type configuration` が設定JSONの契約を返し、`configure` が
plan・scope・runtimeとその参照を持つreceiptを不変保存する。実験固有のURL、件数、
モデルpathをソースへ埋め込まない。

```bash
.venv/bin/python -m src.tasks.annotation schema --type configuration
.venv/bin/python -m src.tasks.annotation configure --configuration /absolute/configuration.json --sha256 INPUT_HASH
.venv/bin/python -m src.tasks.annotation register --runtime /absolute/runtime.json --sha256 HASH --candidates /absolute/candidates.json
.venv/bin/python -m src.tasks.annotation run --runtime /absolute/runtime.json --sha256 HASH --max-seconds 600
.venv/bin/python -m src.tasks.annotation status --runtime /absolute/runtime.json --sha256 HASH --details
```

`Configuration` は `plan`（CampaignPlan全文）、`budget`、`runtime`（plan/scope参照以外の
RuntimeSettings）を持つ。`budget.action = reuse` と既存 `scope` 参照で一連のcampaignの
使用量を継承する。新しい予算を明示作成する場合のみ `action = create` とscope ID、全計測root、
開始前の保存/GPU使用量、待機上限を指定する。作成の再送や中断後の再開でも同じbaselineを使う。
既存の停止状態をconfigureで解除しない。
計測rootはディレクトリまたは通常ファイルを指定できる。既存scopeの一時ファイルが
消えていても、元のbaselineとrootを維持して再利用する。Codex状態の保存先は、全体を
計測するディレクトリrootの配下に置く。状態ファイルの一部だけを列挙したroot集合で
Codex状態ディレクトリ全体が計測されているとは扱わない。

configureは参照hash、selection/preparation形式、対応task、工程costと枠、Codex保存先、
queue script、productionのcode版を検証してから保存する。planが別の資源scopeへ変更される
ことを拒否する。scopeを作る場合は保存量baselineの走査を行うが、動画取得・モデルロード・
Codex/学習の起動は行わない。作成後は `configuration/result.json` の `runtime` 参照のpath/hashを
以降のCLIへ渡す。`schema --type` はruntime/candidates/operation/recoveryにも対応する。
既存scopeの作成入力をscope lock下で照合し、campaignの紐付けをDB transactionで確定してから
requestとplanを保存する。scope入力や紐付けの競合で拒否された設定は保存されず、修正して再送できる。
planの詳細は[plan](../base/annotation/plan.py)、予算の計測定義は
[resources](../base/annotation/resources/README.md)を参照する。

JPEGを残すball campaignでは、[素材の保存契約](../base/annotation/collection/README.md#jpegを残して動画を削除する保存方式)
のpreparation・推論decoder・公開policyを一緒に指定する。保存済みJPEGがある通常処理は
元動画を要求しない。元動画が必要になった場合だけ、`restore_source` operationへ
`source_id` と明示した `attempt` を渡す。取得不能・元のbytesと不一致の結果も記録し、
同じattemptを自動再実行しない。controllerは再取得receiptを専用に検証し、`restored` を
`complete`、`unavailable` / `different_content` を理由の一致した `not_applicable` として受理する。
後者も正常な終端結果であり、保存済みJPEGによるcampaignの継続を妨げない。

pathは固定annotation root内の実ファイルを指定する。相対path・外部path・hash不一致は拒否する。
`CandidateManifest` はcanonical source IDと根拠参照の一覧。過去のsource/channelは収集前に
`HistoryManifest` を持つ `HistoryOperation` を `submit --operation PATH` で登録し、
完了receiptを確認する。レビュー範囲を閉じたcampaignに候補は追加しない。

player推論の `infer_player` はpreviewでもGPU工程と正のGPU時間予約を必須とする。
productionには `review`（Codex）と `publish_ball`（CPU/I/O）のplan枠・runtime costが必要。
学習を設定したplanには `train_ball` / `evaluate_ball` のGPU枠・timeoutを含む予約が必要。
両worker poolは共通台帳と同じ資源scopeを使い、screen/reviewそれぞれの予約量で受付する。
GPU工程はmain repositoryの共有queueに投入する。queue workerの起動は
[training-queue skill](../../../.agents/skills/training-queue/SKILL.md)に従う。

学習の初期モデルは、根拠artifactを持つ `BootstrapModelOperation(task, evidence)` を
`submit` して明示的に登録する。モデルを自動選択・初期登録しない。planに固定した
レビュー方針、公開split、学習recipe、validation、採用条件の仕様は各taskのREADMEを参照する。

保存方式の検証も同じ資源scopeで実行する。`inspect_storage` で既存store全体の参照を固定し、
`compare_storage` で形式・shard共有・RGB cacheを比較する。操作入力・測定定義は
[保存比較](../base/annotation/storage/comparison/README.md)を参照する。

10/25/50の並列条件は `benchmark-observe` / `benchmark-report` で既存workerの生存数と
資源・usageを記録する。固定入力、測定値の意味、未到達・欠測の扱いは
[並列条件の観測](../base/annotation/benchmarks/README.md)を参照する。
同じ固定予測を条件別にレビューするball専用の起動・再開経路は
[benchmark runner](benchmark/README.md)を参照する。

## 画像確認・公開・学習

preview campaignの初期モデル確認は `PredictionPreviewOperation` を通常の
`submit --operation PATH` で登録する。operation JSONの `plan` はruntimeと同じ参照、
`operation` は次の形にする。`PREDICTION_RECEIPT_REF` は予測工程が返した参照を使う。
これはCPUの `render` 枠を使用する。

```json
{
  "action": "prediction_preview",
  "task": "ball",
  "prediction": "PREDICTION_RECEIPT_REFをArtifactRef objectで指定",
  "drawing": "prediction",
  "view": {
    "tiles": [{"frame_index": 60, "crop": {"left": 0, "top": 0, "right": 1920, "bottom": 1080}}],
    "columns": 1,
    "tile_width": 960,
    "tile_height": 540,
    "reason": "初期候補モデルのnative座標と実画像を確認"
  }
}
```

taskは `ball` / `player`、drawingは `plain` / `prediction` を明示する。
frame/cropは実clipの範囲内で選ぶ。登録済みで未レビューの予測だけを受け付け、planの
モデル・設定、manifest、全予測行のframe/PTS・task固有の内容を検証する。表示座標は
task adapterが予測から取得するため、callerは描画座標を指定できない。推論・モデル読込や
注釈状態の変更は行わず、画像生成をレビュー・教師採用の証拠として扱わない。
結果receiptの `results` は [RenderReceipt](../base/annotation/images/contracts.py) の参照で、
その `image` が表示対象、`request` が固定した予測版と表示条件を示す。
ページの上限は [images](../base/annotation/images/README.md) に従い、playerの候補が
1 tileの描画上限を超えた場合も黙って切り捨てず停止する。

screen/review workerの画像・編集要求は親controllerへ渡す。親がrequest/attempt/版/区間を
検証し、CPU工程で処理してから不変の参照を返す。採用には実際のrollout画像提示を照合し、
単なる画像生成や自己申告では完了にしない。
受付済み要求は親所有のSQLite台帳にattempt単位で累積保存し、固定上限を判定する。
inboxの並び順・削除・親の再開で枠を戻さず、同一要求の回収で二重計上しない。
workerにソース動画の全量decodeや共有cacheへの書込みを任せない。

予測は新規clipだけをtask別に束ね、1工程で同じモデルinstanceを共有する。
保存済み予測をreview requestへ変換する際は推論しない。
要求した動画数が準備済みで、`state=prepared` のsourceの準備receiptにある全clipとレビュー対象が一致し、全workerが採用まで
完了した場合だけ公開する。保留・失敗・対象外workerを成功や公開対象外へ読み替えない。
公開と学習へ進む直前には、取得・準備・推論の各操作の最新attemptの成功終了とoperation receiptを検証し、
その成果物が対象source/clipの台帳参照と一致することを要求する。予測の保存と台帳更新だけでは
進めず、失敗・中断・未回収は明示的な新attemptによる復旧まで停止する。
内容重複で除外されたsourceの取得も、台帳の取得参照・source・除外理由と照合済みの
`not_applicable` receiptで完了照合する。除外sourceの準備receiptは理由とともにレビュー集合へ保持し、
hashとsourceの対応を検証するが、公開動画数やレビュー対象clipには数えない。
取得時・準備時いずれの重複除外でも、補充sourceの準備と全レビューの完了後に公開へ進める。

公開結果のtask/product/base snapshot・採用版・追加clip集合を照合してから
`workflow/products.json` を作り、`workflow/completion.json` でレビューと公開操作に結ぶ。
学習recipe未設定なら公開結果を保持した `blocked`。採用条件未設定、学習失敗、採用版競合も
理由を表示して停止する。`complete` は追加教師なし、または学習後の採用/不採用の確定を表す。
不採用は `workflow.learning.state = rejected` として採用と区別する。
公開clipがheatmap負例のみ、またはMDDの有効窓を持たない場合はcoverageと公開receiptを保持し、`learning.status = no_additional_teachers`で完了する。

## 再開と報告

通常の再開は同じruntimeで `tick` または `run` を実行する。完了工程を再実行せず、
`run --max-seconds` の終了時も受付済みsupervisorは自身のtimeout・資源管理に従う。
待機列上限で受付を拒否され、launchだけが保存された工程は、枠が空いた後の再送で
同じattemptの資源要求を登録する。CPU起動・GPU queue登録前の未受付を実行失敗と扱わない。
`status --details` はworkerの現在attempt、全工程の履歴、公開完了参照、学習結果を返す。
最新attemptに置き換わった古い失敗は `superseded: true` と表示し、現在の失敗と分ける。
workflowの完了cacheがあっても、最新工程の未解決の失敗をcampaign完了で隠さない。
`cycle` / `status` はsourceの状態にかかわらず登録済みworkerを照合する。
選別がsourceを更新した直後に中断しても、workerの完了receiptと台帳更新を回収する。
`preview_ready` は台帳のprepared/predictedに加え、工程ごとの最新試行の成功終了、
operation receiptと登録成果物の一致、予測receiptの入力・モデル・成果物hashを要求する。
内容重複などで除外されたsourceは、取得・選別・準備の対象外receiptの取得参照・除外理由と台帳を照合する。
重複clipを検出した準備receiptは証拠として保持するが、除外済みsourceの準備完了としては要求しない。
predictedの保存後でも終了receiptがなければ完了にしない。未回収・失敗はblockedとし、
`submit --attempt N` で新しい試行を明示して回収する。保存済み予測は検証して再利用し、
回収のために推論し直さない。

失敗を自動で再試行しない。取得・推論・公開などは、失敗または中断を確認してから
同じ `OperationInput` を `submit --attempt N`（直前の番号+1）で実行する。
実行中・成功済み工程に新しいattemptを作ることは拒否する。
取得operationは `retry` を除いた同じ論理IDで試行を管理する。
取得失敗後は `AcquireOperation(retry=True)` のsubmitでも次のattemptを作れる。
通常の再開は最新attemptに固定された入力を回収し、同じretry指定を繰り返しても再起動しない。
そのattemptが再度失敗した場合は、さらに大きい `--attempt N` を明示する。
取得工程は入力とattempt番号・保存ディレクトリを照合し、検証済み番号または明示retry指定で再取得を許可する。

選別workerの失敗・timeout・検証失敗は、同じruntimeを使う
`retry-screen --source-id SOURCE_ID` でも新しいworker attemptを明示できる。
`screen_prepare` の `submit --attempt N` はworker attemptを増やさない。
quotaによるcampaign全体のworker停止は、原因解消後に同コマンドへ
`--resume-quota` を付けて明示解除する。通常の `run` / `tick` は失敗workerを再試行しない。
Controllerは予算停止時とcampaignのquota停止確認時に、選別・review両poolの未起動資源要求を
[workerの取消契約](../base/annotation/workers/README.md)に従って取り消し、共有待機枠を返す。

worker/学習の再試行は、`RecoveryInput(plan, reason, operation)` を固定root内に保存し、
`recover --runtime PATH --sha256 HASH --operation PATH` で実行する。

| operation | 指定内容 |
|---|---|
| `retry_worker` | 既存WorkRequest参照、再試行元の `after_attempt`。保留・失敗だけが対象 |
| `retry_learning` | 登録済みintent参照、`train` / `evaluate`、必要なら同intentのresume checkpoint |
| `resume_workers` | quota等で止まった当該campaignのworker受付を再開。資源stopは解除しない |
| `cancel_pending_stage` | 未起動stageの `launch` 参照。理由を残して予約を取消す。稼働中stageは対象外 |
| `reopen_incomplete_screening` | `request`・`after_attempt`・`selection` を固定。未応答/失敗の画像要求が残る、準備前の誤確定selectionだけを保留へ戻す |
| `observe_host_space` | 認可記録と結んだ `ObservedHostSpacePolicy` の参照。間接host mountの予約控除を実空き監視へ変更し、下限到達時は実行中の所有処理も停止する |

再試行理由と結果を不変保存し、同じ要求の再送は結果を返す。worker開始後、結果保存前に親が
落ちた場合は、変わったattemptを再起動せず現在状態の確認を要求する。学習再試行は次のtickで
queueへ投入する。資源stopは原因解消後の `resume-budget --reason TEXT` で解除する。
scopeや累計を初期化して上限を回避しない。

### 不具合修正後に同じcampaignを続ける

controllerと所有jobを停止・終了確認し、修正版をテストしてcommitする。
`code-update --runtime PATH --sha256 HASH --verification EVIDENCE_PATH --reason TEXT`
は、旧runtime・同じplan/scope・修正commit/diff・検証記録を結ぶ不変の更新記録と
新runtimeを保存する。旧plan、候補、成果物、保存baseline、GPU積算、工程costは変更しない。
以降は返されたreceiptのruntimeを指定する。全stage・screen/review worker・学習launchが
更新記録を参照し、子processでもコード版を照合する。コードがdirty、jobが稼働中、
またはqueue待ちなら更新を拒否する。旧controllerを同時に再起動しない。
資源待ちでlaunchだけを保存したscreen/review workerは、資源要求の登録前も含めて
更新時に旧attemptを `cancelled` として履歴へ回収し、pending予約を取消す。
新runtimeで `retry_worker`（または `retry-screen`）を明示すると、次attemptを新コードで起動する。
通常tickは取消済みworkerを自動再試行しない。取消途中で中断した更新も同じ入力で再送できる。

成功済みstageは旧launchの結果を読む。旧版で未起動のstageは勝手に新コードへ付け替えず、
`cancel_pending_stage` で取消し、必要な工程だけ通常の `submit --attempt N` で再試行する。
画像サービスの未完了を素材不足として誤確定した選別は、全関連jobの停止・予約取消後に
`reopen_incomplete_screening` で戻し、`retry_worker` を使う。旧selectionとworkerの結果は
不変保存し、撤回理由と対応先を台帳に残す。準備済みclipや公開済み教師はこの操作の対象外。

`stage` はsupervisor専用で、固定した入力digestとの対応を要求する。
置換済みballの旧run/qa/intakeと旧annotateのball注釈・finalize入口は撤去済み。置換先が未実装のplayer用入口は残す。旧形式の読取と保存台帳への登録は
[migration](migration/README.md)を参照する。
