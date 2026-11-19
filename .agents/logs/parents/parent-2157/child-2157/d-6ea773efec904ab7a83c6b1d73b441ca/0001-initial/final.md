推奨は、**レビュー記録を簡素化し、private stateを正本とする自動進行処理へ置き換える設計**です。現行gateへ待機ループを追加するだけでは、厳密照合と大量取得が残ります。

調査対象は `codex/merge-pipeline`、commit `a14a7bacf83a7cf7b10a61e0d621df5d7b920a30`。既存の担当・worker・worktreeの対応を確認しました。コード変更・commit・PR作成は行っていません。以下、現状は確認済みの事実、変更後の構成・数値は提案です。

1. **削除・維持・作り直しの範囲**

| 対象・関数 | 現状と変更案 |
|---|---|
| [parent/snapshot.py](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/parent/snapshot.py#L76) — `snapshot`、`prepare`（140行）、`transition`（269行） | diff SHA-256、収集後の再取得、公開済みtargets固定記録との照合を削除。対象repo・PR・review時head・Issue・親の指示・モデル選択を保存する要求作成へ置換する。diffとcheckoutはレビュー資料として生成するが、以後のmerge許可を固定するdigestには使わない。 |
| [parent/review_handoff.py](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/parent/review_handoff.py#L14) — `runner_ended`、`request`（68行） | 現workerの終了・回収確認と成果PRの対応付けは維持。`prepare → transition`を、新しいpipelineの登録・切離し起動に変更する。 |
| [review/parent_review.py](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/review/parent_review.py#L163) — `_collect`、`_run_review`（541行） | `_collect`の固定snapshot一致・収集後の再取得、`_run_review`のdiff再hash・postcheckを削除。対象一式を収集し、共通ランナーでCodexを1回実行し、結果を保存する責務に縮小する。 |
| 同ファイル — `publish`（366行）、`publish_result`（476行） | pending→各PR projection→親resultという公開順序、canonical digest、公開内容照合を削除。結果公開は表示・記録として扱い、公開コピーからmerge権限を再構築しない。 |
| 同ファイル — `recover`（947行）、`resume_publication`（1199行）、`complete_evidence`（1274行）、`resolve`（1360行） | 現行の採用・公開復旧・4項目証拠・変更パス完全一致の入口は廃止。中断復旧はpipelineの再開へ、major対応は親の明示的な続行指示へ集約する。 |
| [parent_review_policy.py](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/review/parent_review_policy.py#L326) — `evidence`、`validate_evidence_completion`、`validate_resolution`、`BaseAdvance`、`evaluate_coverage` | 廃止対象。要求・出力parserは新schemaへ作り直す。業務上の結果は`approve`またはmajor一覧。実行失敗・出力不正は別のruntime状態にし、approveへ変換しない。 |
| [review_workspace.py](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/review/review_workspace.py#L131) | `workspace`・`build_prompt`・diff取得は維持。`verify_base_advances`（52行）・`verify_rebased_snapshot`（198行）の継承判定は削除する。 |
| [review_evidence.py](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/review/review_evidence.py#L82) | `followups`の公開／ローカル照合は削除。`python_check_runs`はCI観測モジュールへ移し、条件付きGETを追加する。 |
| `review_recovery.py`・`review_inputs.py`・`review_interruption.py`・`review_locks.py` | 保存結果のdigest連鎖、固定ファイル検査、旧stage依存の復旧を削除。**PR排他、同じruntimeの識別、終了・回収確認**は新しい実行管理に残す。 |
| [merge_gate.py](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/review/merge_gate.py#L123) — `_collect`、`_load_evidence`、`_evaluate`、`observe`、`_merge_request` | 全体収集・coverage・公開照合を廃止。PRごとの順次取得と短絡判定へ全面置換する。現在のmerge直前の全体再取得は551行、merge後の再収集ループは629行。 |
| [merge_gate_policy.py](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/review/merge_gate_policy.py#L43) | `publication_errors`・旧`MergeInput`・coverage依存判定を削除。`python_ci_errors`、`auto_merge`、依存順計算の必要部分を再利用する。 |
| [parent/cli.py](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/parent/cli.py#L287) | `review-request`を一括起動入口へ変更。分離した`review-prepare`・`review-state`は新規操作から撤去し、状態表示・続行・待機の入口へ置換する。`record-result`は通常の成果受理用として維持する。 |

レビューモデルの選定方針は維持します。ただし一致検査はレビュー側だけでは完結していません。[`review_settings.runtime_summary/verify_runtime`](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/lib/review_settings.py#L346)に加え、共通ランナーの[`Events.observed`](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/lib/agent_events.py#L319)も選択値と実効値の不一致で失敗します。レビュー用途では観測記録へ変更し、他用途の契約は維持する必要があります。

2. **削除で影響する利用側と既存記録**

特に次の接続は同時変更が必要です。

| 利用側 | 必要な対応 |
|---|---|
| [parent/development.py — `verify_completion`](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/parent/development.py#L571) | 668行で旧gateの`collect`、669行で`load_evidence`を再実行している。新しいPR別のmerge後確認記録を読む方式へ変更する。これを残すと完了時に大量取得と旧照合が復活する。 |
| [dispatcher/supervisor.py — `_finish`](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/dispatcher/supervisor.py#L1267) | 現在は実装workerが終了するたびに親を起こす。pipelineが管理する修正ターンはpipelineへ返し、親を起こすのはmajor・判断要求・上限到達・最終結果に限定する。 |
| [parent/state.py — `Store.load`](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/parent/state.py#L171) | review request v1～v3を前提にしている。新形式と旧形式の読取を分離する。既存registryを消したり、旧記録を書き換えたりしない。 |
| [parent/history.py — `_payload` / `read_history`](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/parent/history.py#L97) | 旧レビューparserを削除すると、レビュー以外の親履歴読取まで壊れる。旧`result/adoption/pending/pr/resolution/evidence`は履歴表示用に読み、新しいgateへ旧検査を持ち込まない。 |
| [review_inspection.py — `compare_snapshot` / `inspect_run`](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/review/review_inspection.py#L105) | digest・継承・公開証拠判定を削除。review時head、現在head、変更ファイル数・追加削除行数、major、現在段階、修正回数、不明点を表示する。 |
| [decision_logs/review_capture.py](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/decision_logs/review_capture.py#L236)・`parent_commands.py` | 廃止CLIと旧stageへの対応付けを更新。CLIによる自動修正の確認を「親AIが確認した」と記録しない。判断記録の専用公開・merge処理は維持する。 |
| [worker_logs/schema.py](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/worker_logs/schema.py#L178) | effort=`max`だけを許可しているため、追加指示の`high`では公開検証が失敗する。修正ターンの設定表現を対応させる必要がある。専用mergeの安全条件は変更しない。 |
| ops-report | `review_policy.normalize_rollup`と`CheckState`をmemory PRが再利用している。共有部品は維持する。全体調査や専用merge例外へ通常pipelineの変更を波及させない。 |
| [review/evaluation/execution.py](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/review/evaluation/execution.py#L351)・`replay.py` | 旧parser・`runtime_evidence`・`source_evidence`に直接依存している。歴史データの読取と新形式の評価を分け、削除した本番gateを比較用に残さない。過去の比較結果・固定入力は書き換えない。 |

API範囲には追加の修正点があります。PR取得は既に[`GitHub.pulls`](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/parent/github.py#L368)で家族のtimelineに限定されています。一方、[`EvidenceIssues.find`](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/parent/evidence.py#L104)は、保存済みの証拠Issue番号があっても全親分の`parent-evidence`一覧を取得します。

新pipelineでは、保存された証拠Issue・合意・採用元の参照を直接取得する経路が必要です。番号不明時にrepo全体走査へ戻さず、開始時に対象の参照を確定します。また、review MCPの`github_items`もrepo全体一覧を取得できるため、通常pipelineの対象指定を取得側へ渡す必要があります。

旧記録については、**履歴の読取互換は残し、旧操作・旧判定経路は残さない**方針を提案します。旧レビューを新pipelineへ引き継ぐ場合は親が明示して対象・結果・未解消指摘を確認し、履歴が存在するだけで自動承認しない設計です。

3. **自動進行CLIと状態機械**

公開入口は、拡張した`review-request`、`merge-status`、`merge-continue`、`wait-merge`に集約する案です。`review-request --wait`なら、1コマンドで処理を切り離して起動し、会話の背景実行から通知を待てます。

| 段階 | 遷移 |
|---|---|
| `requested` | 担当・実行根拠・対象・作成workerを確認して保存する。 |
| `review_running` | 対象全体をCodex 1回でレビュー。approve→`ci_wait`、major→`parent_wait`。runtime失敗も親へ返す。 |
| `parent_wait` | 親の続行指示を待つ。major対応済みの明示でCIへ進む。`--rereview`を付けた場合だけレビューを追加する。 |
| `ci_wait` | 現headのPython CIを待つ。head変更は観測対象を更新。成功→gate、失敗→修正worker。 |
| `gate` | 手元確認→CI→競合・draft・head→needs-human→依存→mergeの順。最初の不合格で終了する。 |
| `dependency_wait` | 指定した依存Issue・PRだけを待つ。他の進行可能な対象は処理できる。 |
| `repair_reserved / repair_running` | 元workerの同じsessionを再開。終了・回収・報告を確認後、修正済みならCIへ、判断要求なら親へ。 |
| `merge_pending` | merge試行を保存して`gh pr merge --match-head-commit`を実行。応答不明なら対象PRの事実を照合する。 |
| `post_merge_wait` | PRのmerge、対象Issueのclosed/completed、merge commitのmain CIを確認する。 |
| `completed / parent_wait` | 正常完了または異常を、永続化した通知イベントとして返す。 |
| `rate_limit_wait` | 元段階・再開時刻を保存し、待機後に同じ取得から続ける。 |

private stateには、少なくとも以下を保存します。

- pipeline ID、schema、repo、親・対象Issue・PR、担当generation、実行根拠への参照。
- review時head、結果、major、親の続行判断、再レビュー履歴。
- PR別の現在head、CI run/attempt/check、段階、次回照会時刻。
- 作成workerのdelegation・session系列・最新run・worktree・branch。
- 自動修正の通算回数と各試行ID、起動予約、実行・終了状態。
- merge試行、merge commit、Issue確認、main CI確認。
- 通知イベントIDと配送状態、HTTPのETag・取得時刻・rate limit待機時刻。

長い待機中はregistry lockを保持しません。pipeline単位とPR単位の排他、短いatomic更新を使い、無関係な親の更新でレビュー結果を捨てない構成にします。

二重実行防止では、現在の仕組みを次のように再利用します。

- 手動workerは[`prepare_run/run_worker`](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/parent/manual_workers.py#L445)の終了確認・予約・同session再開を再利用する。ただし、修正試行IDから同じ予約を再取得できるようにする。
- 共通ランナーの[`session.lock`と`latest_run`確認](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/lib/agent_runtime.py#L424)を維持する。controller再起動時に既存runを観測し、毎回新しいrunを作らない。
- dispatcherでは、現在の[`start_child`](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/dispatcher/supervisor.py#L1058)が「実行中の親」を要求するため、そのまま呼べない。親が登録したpipelineに限定した修正起動経路を追加する。親runtimeの環境を偽装して通さない。
- merge応答を失ってもPRがmergedなら後確認へ進む。既存のmerge試行記録を消して再送しない。
- 手動親の通知は[`wait-worker`](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/parent/worker_wait.py#L107)と同様の読取専用待機にする。dispatcher親にはイベントID付きのwake要求を保存し、親がまだ終了していない場合も通知を失わない。

レビュー後の変更量はreview時headと現在headのtree差分から計算し、再レビューの判断資料として返します。変更量を閾値で自動判定したり、mergeの一致条件にしたりしません。

4. **CI待機・条件付きGET・rate limit**

既存の`python_ci_errors`は、対象headの最新`Python` check runについて、`success`以外を成功扱いしません。この契約を維持します。rollupや別headの成功は代用しません。

取得は、現在のPRと次のcheck-runs endpointを中心にします。

```text
GET [local path]
GET [local path]
```

現在の[`HTTPClient.request`](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/issues/transport.py#L191)は任意の条件付きheaderを受け取れず、258行で応答headerをstatus行だけに縮めています。[`GitHub.get`](https://github.com/Motoki0705/tennis-lab/blob/a14a7bacf83a7cf7b10a61e0d621df5d7b920a30/.agents/ops/issues/cli.py#L320)も200以外をエラーにします。この境界に、status・必要なheaders・bodyを返す取得APIを追加するのが先です。

提案する待機条件は次のとおりです。

- 通常照会は**60秒間隔**。サーバーの`x-poll-interval`が長ければ従う。
- ETagを保存し、`If-None-Match`を送る。304は、そのURL・認証主体・headに対応する保存応答だけを再利用する。
- ETagがないendpointはその事実を記録して通常GETを使う。304なのに保存応答がない場合は明示エラーにする。
- CI待機期限は**6時間を初期案**とし、設定可能にする。超過時は状態を保持して親へ通知する。数値は未合意。
- head変更後は新headのCI待機へ戻る。待機中にCodexを起動しない。
- main側は可動なmain先端ではなく、確認済み`merge_commit_sha`のCIを待つ。

認証付き条件付きGETの304はprimary rate limitを消費しません。ただしHTTP問い合わせ自体は残るため、計測ではGET総数と200/304を分けます。[GitHub公式の取得指針](https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api)

403/429では`Retry-After`、`x-ratelimit-remaining`、`x-ratelimit-reset`から再開時刻を決めます。通常の権限不足403をrate limit扱いして待ち続けません。reset情報のないsecondary limitは公式指針に沿って待機を延ばします。[GitHub公式のrate limit仕様](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)

GET数の目標は、対象家族・PR・依存／出典のページ数と、待機照会回数に依存する形です。repo全体のIssue件数を項に含めません。`ci_watch.collect`はmain履歴と全attemptを収集するため、この待機ループには使いません。

5. **修正workerへの入力と最終報告**

失敗情報は、PR・head・workflow run ID・attempt・失敗job・失敗step・URL・ログ本文の保存先・取得できなかった範囲をまとめて渡します。集約job`Python`だけでなく、そのrunの失敗jobを取得します。

runはhead／check suiteで絞り、attemptを固定してjobsを取得できます。[workflow run API](https://docs.github.com/en/rest/actions/workflow-runs)、[attempt別jobs API](https://docs.github.com/en/rest/actions/workflow-jobs#list-jobs-for-a-workflow-run-attempt)

jobログは302で一時URLが返るため、現在のredirect拒否のHTTP clientだけでは取得できません。既存のログ取得部品を利用するか、ログ取得専用にredirectを扱います。ログ欠落を空ログとして渡さず、欠落理由を明記します。[ログ取得API](https://docs.github.com/en/rest/actions/workflow-jobs#download-job-logs-for-a-workflow-run)

workerの最終文は、例えば次の厳格なJSONにする案です。

```json
{
  "schema_version": 1,
  "repair_id": "保存済みの修正試行ID",
  "pr": 123,
  "outcome": "repaired",
  "head_sha": "修正後のhead",
  "summary": "原因と修正内容",
  "verification": ["実行した検証と結果"],
  "parent_request": null
}
```

`outcome`は`repaired`または`needs_parent`。後者では判断してほしい内容・理由・必要な次の操作を返します。自然文から成功を推測しません。CLIがremote headとrunner終了を確認し、workerの自己申告だけでCI成功やmerge成功にはしません。

既存sessionへ途中から`--json-schema`を追加すると、共通ランナーの保存contract一致検査に抵触します。まずは**最終文をpipeline側でschema検証する方式**が変更範囲を抑えられます。

さらに、[追加コメント](https://github.com/Motoki0705/tennis-lab/issues/2157#issuecomment-6897046131)の`effort=high`は確定指示として扱います。現状は以下の3段で拒否されるため、引数追加だけでは動きません。

- `agent_settings.resolve`の実装worker=max固定。
- `agent_runtime.run`のresume時settings一致。
- `manual_workers.model_settings`とworkerログschemaのmax固定。

同じモデル・sessionを維持し、**CI/gate修正という明示されたターンだけhighにする契約**が必要です。通常実装の設定変更や別sessionへの付替えに広げません。修正workerの指示にはpush後にターンを終了することを明記し、通常worker指示のCI結果報告要求との矛盾も解消します。

6. **子Issueへの分け方と検証案**

3つのコード変更単位を提案します。今回は起票していません。

| 単位 | 依存 | 独立した完了条件 |
|---|---|---|
| A. 条件付きHTTP取得とrate limit情報 | なし | 応答header保持、条件付きGET、304再利用、rate limit分類を実装。既存取得の挙動を維持し、権限403・応答不正・ログ取得境界を検証する。 |
| B. 同じworkerの修正ターン再開 | なし | 同session・同modelでhighの修正ターンを扱い、予約の重複防止、終了確認、最終報告parser、workerログを対応させる。通常workerは従来契約を維持する。 |
| C. レビュー[local path] | A・B | 新状態機械、親CLI、dispatcher通知、完了確認、履歴reader、文書・旧経路削除をまとめて切り替える。fake e2eで受入条件1～11を満たす。 |

レビュー簡素化とgate置換は同じPRが適切です。別々にmergeすると、新レビューに旧coverageがない期間や、完了確認だけ旧公開証拠を要求する期間が生じます。

実GitHubの受入条件12は、Cのmerge後に親が実PRを1件通して確認する作業として残します。確認だけの子Issueを必須にはしません。

検証は既存fixtureを再利用できます。

- `test_merge_gate.py`のPython必須・別head拒否・人間停止・依存・merge応答喪失のケースを維持／更新。
- `test_wait_worker.py`、`test_manual_worker_runs.py`の終了・回収・同session再開を拡張。
- `test_merge_gate_reads.py`のfake API記録を使い、**失敗段階より後の取得がないこと**と、無関係Issue数を増やしても要求数が増えないことを検証。
- 新e2eで、一括完了、major→続行、明示再レビュー、CI/gate混合3回、head更新、依存待ち、main CI失敗、各副作用の前後での中断を検証。
- 旧`test_fix_push_requires_current_published_resolution_and_actual_diff`、`test_inheritance_uses_file_deltas_and_the_base_period`などは廃止契約として削除し、「head変更後も再レビュー・resolveなしで通る」テストへ対応付ける。

待機テストにはfake clockを使います。ケース数・所要時間は実装時に収集・実測し、現時点で未計測の予算を固定しません。

7. **判断が必要な点と、原則との整合**

実装前に親へ返したい判断は次のとおりです。

- **短絡gateと人間停止の優先。** CI失敗や競合で止まった後に一切の後続取得を禁止すると、その時点の親・子のneeds-humanを確認できません。提案は「gateは短絡するが、worker起動前には独立した人間停止・実行権限確認を行う」です。これを省くと合意した人間の制御が失われます。
- **複数PRでの修正3回の単位。** 本文はCI修正とgate修正の合算を明示していますが、PR別かpipeline全体かは明記していません。上限を拡大しない案はpipeline全体で3回、PR別内訳も保存する形です。再開・再レビューで回数をリセットしません。
- **WSL再起動後の自動起動範囲。** 保存状態から明示resumeできることと、OS起動後にcontroller自体が自動起動することは別です。後者まで必要なら、登録済みpipelineだけを復帰させるuser serviceと導入確認が必要です。
- **副作用の成否が確定できない中断。** GitHubへのmerge送信直後やworker起動直後の応答喪失では、完全自動復帰と「同じ操作を絶対に再送しない」を常に両立できません。既存記録・remote事実で照合できれば続行し、確定不能だけ`recovery_required`として親へ返す案です。
- **待機間隔・期限。** 60秒[local path]。期限到達は成功・失敗の捏造ではなく、状態を保持した親への差戻しとします。
- **依存先が調査のみの場合。** 現行`dependency_status`は`completed_without_pr`を許可します。新要件の「依存先の子がmerge済み」を文字どおり全子へ適用するか、実装成果を持つ依存だけに適用するかを明確にする必要があります。

main CIのflaky失敗は、合意どおり親へ通知します。CLIによるrerunや原因の自動断定は追加しません。

静かなフォールバック禁止については、壊れたstate、未知schema、元worker不明、ログ欠落、未確認の起動を明示します。旧レビューへ戻る、新workerを勝手に作る、失敗をapproveに変える処理は設けません。

文書は、review READMEを状態機械・gateの正本、parent READMEをCLIの正本、parent-session skillを親の判断手順にします。development workflowは接続方法を説明して参照し、同じ状態表を複製しません。`.agents/README.md`の旧merge手順と、今回変更で無効になるmemoryも更新対象です。

今回の検証では、既存benchmarkをfake GitHub・fake agentで1回実行しました。PR 3件・他親20件で、**REST GET 247回、`gh pr view` 12回、merge 3回、gate計測時間12.06秒**でした。証拠Issue一覧取得は9回です。[測定結果]([local path])

これは現行実装の測定であり、改善後の性能や実GitHubの速度を示すものではありません。新設計のテスト、実PRでの通過確認、pre-commitは未実施です。作業treeはcleanのままです。