# 親issue本文の補足追加

[`issue-supplement.yml`](../../../.github/workflows/issue-supplement.yml) は、作成直後の親issueの本文へ
Geminiで背景を補足する。原文は空白・改行を含めそのまま保存する。
補足する箇所と要否はGeminiが判断し、既存コード・READMEに根拠のある客観的背景だけを追加する。
新しい決定・推測・仕様・タスクや起票者への指示は追加しない。

## 起動と状態

repository ownerによる `issues.opened`（`issue:parent` がある場合）、
`issues.labeled`（`issue:parent` を付けた場合）、mainへの `workflow_dispatch` だけを受け付ける。
`ISSUE_SUPPLEMENT_ENABLED_SINCE` が未設定ならjobを起動しない。
対象・確定状態・有効化時刻・ラベル定数は
[`issues.supplement_state`](../issues/README.md#親issueの補足追加状態)を使う。
workflowは `vars.ISSUE_SUPPLEMENT_ENABLED_SINCE` の値をCLIの `--enabled-since` へ明示的に渡し、
共通の `supplement_state` で日時と状態を検証する。空値は未設定として渡す。
Variables APIの追加権限を持たないjobでは `read_enabled_since` を呼ばず、権限エラー後のfallbackも行わない。
workflow以外の親CLI・dispatcherの取得契約は変更しない。
有効化前のissue、親ラベルのないissue、処理済みの自動イベントには何も書き込まない。
子も兼ねる親は親・子両方の本文書式を検査する。

同じissueのrunをconcurrencyで直列化し、実行中のrunは取り消さない。
workflowファイル名 `issue-supplement.yml` と表示名 `issue-supplement #<番号>` は
[`parent wait-supplement`](../parent/README.md#補足追加の確定待ち)との共有契約。
本文競合によるattempt 2も同じrun内で実行し、runの表示名とowner起点の認可を保持する。
wait-supplementはこの1つのrunが全jobを終えるまで待つ。別runの起動や追跡は行わない。

## 処理と責務

| 部品 | 処理 |
|---|---|
| `pipeline.prepare` / job `prepare` | GitHub-hosted runnerで手動のfailed再実行をpendingに戻してから合意を確認する |
| `pipeline.generate` / job `supplement`・`supplement_retry` | self-hosted runnerで入力を保存し、安定待ち・writer・不変条件・verifierを実行する。GitHubへの変更権限は持たない |
| `invariants.py` | 原文と挿入指示から本文を組み立て、原文完全保存と補足領域を検査する |
| `workspace.py` | 呼び出しごとの使い捨てworktreeと、検査側checkout・使用したprompt/schemaの完全性を管理する |
| `github.py` | 既存の合意・実行根拠の解析、GitHubへの要求、編集履歴の全ページ取得と照合 |
| `publication.py` | 公開する項目の選別、token等の検査、summaryの出力。生のagent診断は公開しない |
| `pipeline.apply` / job `apply`・`apply_retry` | GitHub-hosted runnerでartifactを再検査し、最新入力との照合・PATCH・編集履歴の検査・確定ラベルの付与を行う |
| `pipeline.fail` / job `failed` | 上流jobの失敗時に、GitHub-hosted runnerで対象・合意・本文・履歴を再取得し、書き込み結果を確定する。確認できなければpendingを保つ |

1. ownerによる手動再実行でfailedの場合、prepareが現在の状態を読み直してfailedラベルだけを個別に削除し、
   pendingを再取得してから合意・実行根拠を確認する。記録済みならskippedを付けて終了する。
   done/skippedや非対象は再実行しない。通常のpendingでも生成前に記録を確認する。
   title・body・labels・created_at・作者・全コメントをprivateなrun dirへ保存する。
   本文が60秒変わらなくなるまで10秒以下の間隔で確認する。10分以内に安定しなければ失敗。
2. repository ownerをpublisherとして [共通reader](../parent/README.md#機械証拠の保存共通reader) 経由の合意・実行根拠と、
   `parent.judgement_comments.owned_comment` が完全検証した `startable` の判定記録を調べる。
   既に記録があれば本文を変更せず `supplement:skipped` にする。引用・別publisher・保留判定は実行根拠にしない。
   壊れた所有コメントは解析失敗とし、補足せず `supplement:failed` にする。
3. writerとverifierは、[`agy_headless.py` の公開API](../lib/README.md#agyの共通呼び出し部品)を使う。
   版・モデルは同部品と `agents.conf` のpinだけに従う。呼び出しごとに別の会話とlogを作り、
   repo・関連issue/PRはGemini自身が読み取る。1呼び出しの上限は共通部品の既定値（600秒）。
   許可外のシェル形式は共通部品のPreToolUse hookが理由付きで拒否し、読み取りと構造化出力の生成を続けさせる。
   workspace外への `read_file` もsymlink解決後のパスで拒否する。
   streamのシェル・読み取り試行は同じtool・stepと、コマンド全文またはPathがhook記録と一致することを必須とし、照合できない試行や拒否後の実行形跡は
   `command_hook_mismatch` として失敗する。末尾省略の復元はハーネスと同じ共通部品を使う。
   拒否があっても、以下のschema・不変条件・意味検査・照合付き書き込みの合格条件は変えない。
4. writerの挿入指示をschemaで検査し、Pythonでdetailsを生成して不変条件を調べる。通った本文だけを別会話のverifierへ渡す。
   `verdict=pass` かつ `differences=[]` だけが意味検査の合格。自由文やCLIの終了値から結果を補わない。
5. 不変条件か意味検査で落ちた場合、直前の挿入指示と具体的な理由をwriterへ渡して1回だけ修正する。
   schema不正・構造化出力なし・版/モデル/認証等の呼び出し失敗は修正を試さず失敗する。
   supplement / supplement_retry のjob上限は各60分。最大4呼び出し（writer/verifierと1回の修正/再検査）の40分、
   本文の安定待ち最大10分、checkout・依存準備・完全性検査・artifact処理の10分を確保する。
6. applyはartifactのschema、不変条件、verifier合格を再検査し、対象状態と合意記録も再取得する。
   元本文と現在の本文、入力時と現在の編集履歴が一致したときだけ本文をPATCHする。
   不一致なら書かず、applyの `needs_retry=true` outputを受けた静的な `supplement_retry`（self-hosted）と
   `apply_retry`（GitHub-hosted）が最新本文から1回だけ作り直す。attempt 2も不一致なら失敗する。
   dispatchは行わず、最終label以外のGitHub変更は手動再実行のfailed削除と本文PATCHに限る。
   本文PATCHの直前の最後の読み取りでも、合意・実行根拠の記録を確認してから書く。
7. PATCH後にGraphQLの `userContentEdits` を全ページ取得し、以前の履歴が変わっておらず、
   今のGitHub tokenのviewerによる編集が1件だけ増えたことと現在の本文を確認して `supplement:done` にする。
   GitHubが最初の編集時に作成時の版も追加する場合は、`includesCreatedEdit` のfalse→true、
   変更前の空履歴、作成時刻・作者が一致する一意の作成記録を照合して、その記録を区別する。
   同一作者の追加編集、履歴の欠損・取得失敗・曖昧さも不合格。
   編集者と作成者は `User` なら `login`、`Bot` なら `login + "[bot]"` に正規化し、viewerとの本人判定・creation recordの照合に使う。
   `editor=null` は不一致、未知の型・`__typename`/`login` の欠落は取得エラーにする。

## 原文保存と構造化出力

Writerは本文全体を返さず、`{supplements: [{after, content}], notes}` を返す。
`after` は原文中に1回だけ現れる文字列で、完全な段落・見出し・リスト・引用ブロックの末尾を指す。
すべてのanchorは原文を参照し、複数の補足は原文上の位置順に挿入する。
文・リスト・引用・コード・既存HTMLの内部、機械管理節への挿入、曖昧なanchor・重複位置は拒否する。
`markdown.py` がCommonMarkのblock境界を原文のoffsetへ対応付ける。CRLF・CR・LFを区別し、
Unicodeの行区切り文字を改行offsetへ誤計上しない。

`content` は補足Markdownだけで、wrapperはコードが次の固定形で生成する。

```html
<details>
<summary>Geminiによる補足・背景情報</summary>

確認した既存の事実と、その根拠ファイルへのリンク。

</details>
```

空の補足、raw HTML、details/summaryのdelimiter、未閉鎖fence、`#` / `##` 見出し、
タスクのcheckbox、原文のリンク先を変えうる参照定義は拒否する。
数値・参照・URLの追加は補足内で許可し、原文側は一切変更できない。
`check_invariants` は原文と宣言された補足から本文を再構築し、候補全体と完全一致で比較する。
原文のsubstring/subsequence照合や任意のdetailsを除去する方式は使わない。
既存のdetails・引用・コメント・codeも原文の一部として保存し、`validate_body` も検査する。
補足不要なら空配列を受理する。Verifier合格後、本文PATCHなしで `supplement:done` を付ける。

Verifierには原文、挿入指示、生成済みの `supplemented` を渡す。根拠を独立に読み、
原文の完全保持、補足の各主張の実在性、推測・仕様追加・要件の変更がないことを検証する。
差分の引用fieldは `original` / `supplemented`。`verdict=pass` かつ `differences=[]` のみ合格。
公開handoffのschema_versionは2で、検査済み本文と `supplements` を保持する。
apply側でもschema・不変条件を再検査し、挿入指示と一致しない本文や旧schemaのartifactは適用しない。

編集履歴APIのフィールドは[GitHubのIssueの仕様](https://docs.github.com/en/graphql/reference/issues#issue)と
[UserContentEditの仕様](https://docs.github.com/en/graphql/reference/users#usercontentedit)に従う。
RESTの本文PATCHは原子的なcompare-and-swapではない。PATCH後に割り込みを検出した場合は失敗を記録し、
**他の編集を上書きする巻き戻しは行わない**。原文と各変更はGitHubの編集履歴から確認する。
検査段階での失敗では本文を一度も書き換えない。

## agyの作業領域と完全性

workflowのsupplement・supplement_retryは履歴を全取得する。呼び出し前に検査側checkoutがcleanであることを確認し、
HEADを固定する。writer・verifier・修正の**呼び出しごと**に、run dirの隣の一時ディレクトリへ
`git worktree add --detach` でそのcommitを展開する。履歴・remoteは元repoを参照するので
`git log`・`git show`・`gh issue view` のための文脈を維持する。
呼び出しごとに作り直す理由は、writerが作業領域を変更してもverifierや修正の入力へ持ち越さないため。
作業領域内のコード・prompt・schemaは検査や合否判定に使わない。
hook設定だけを呼び出し中のworkspaceへ置き、実行プログラムはcontroller側を使う。
配置・片付けと記録の契約は[共通部品](../lib/README.md#agyの共通呼び出し部品)に従う。

使用するpromptの雛形とschemaは固定commitから読み、検査側のファイルとの一致を確認する。
入力JSONを加えたprompt、schemaのコピー、構造化出力、log、公開artifactは使い捨て領域の外へ置く。
privateなrun dirも検査側checkoutの外を必須とする。
各呼び出しの直後（agyが失敗した場合も）、検査側の追跡ファイルの内容・mode・index、未追跡ファイルの一覧
（ignoredも含む）、HEADが呼び出し前と一致し、使用したprompt・schemaが期待した内容と一致することを検査する。
ファイルのstat cacheだけに頼らず、内容のSHA-256も比較する。検査中にbytecodeは作らない。
絶対パス等から検査側に変更が入った場合も結果を破棄し、静的な理由を公開して停止する。
後続のGitHub-hostedのfailed jobが[本文と履歴を照合して確定する](#失敗skip手動の再実行)。

終了時は `finally` で自分が作ったworktreeだけを `git worktree remove --force` し、一時ディレクトリも削除する。
構造化出力の不正、agyの異常終了、完全性の違反でも片付ける。片付けの失敗も明示的な失敗とし、
他のworktreeの登録はpruneしない。作業領域内だけの変更は捨て、合否の理由にしない。

## 公開するデータ

runnerのHOME・agyの場所・導入と直接ログインの手順は[runner README](../../../scripts/github_actions/README.md#agyの導入と疎通)が正本。
supplement jobのGitHub tokenはcontents・issues・pull-requestsのreadだけ、checkoutはcredentialを永続化しない。
prepare・apply・apply_retry・failedはissuesのwriteとcheckoutのcontents readだけで、
agyの導入やGoogle認証情報を持たない。全jobからactionsのwriteを外している。
外部actionは公式release tagと照合した完全commit SHAと版コメントで固定する。

run dirは0700、保存ファイルは0600。入力・prompt・writerのnotes・verifierの説明・agy logはprivateに保つ。
各turnのlogに対応する `.stream.jsonl`・`.hooks.jsonl`・`.commands.json` も同じ非公開領域に残す。
失敗時の `failure.json` には `denied_actions` も保存する。成功・異常終了・timeoutともコマンド全文を公開しない。
読み取りの拒否パスも非公開記録に限り、公開するのは件数と固定形式だけとする。
artifactへ渡すのは `public/handoff.json` 1ファイルだけで、globやrun dir全体をuploadしない。
合格時は比較用の元本文、検査済み本文、履歴のID・時刻・作者、空差分の検証結果を含む。
失敗した候補本文やコメント、log、会話ID、ローカルパス、認証情報は含めない。

本文はartifact作成前にGitHub token、Google access/refresh token等の既知の形を検査する。
ローカルパス・会話/session IDの表記・UUIDも公開を拒否する。既知の形による保守的な検査なので、
原文の引用にこれらがある場合も自動処理を失敗させ、消した本文を黙って適用しない。
公開JSON全体も同じ検査を通し、applyで再検査する。
summaryには状態、検査理由、意味差分の種類と実際の本文から取れた引用、usage・所要時間を出す。
verifierの自由文説明や拒否コマンドの生文字列は出さない。usage未取得はnull/未取得のまま扱い、0に補わない。
コマンドの試行・拒否は各turnの件数と、共通部品が伏せ字化した形式だけをsummary/artifactへ渡す。
呼び出し失敗でも取得できた集計を残す。
過去revisionの不合格理由は、修正成功後もrevision付きで残す。

## 失敗、skip、手動の再実行

failed jobは本文と編集履歴を再取得し、未書き込みなら `supplement:failed`、候補本文への本人の編集1件を確認できれば `supplement:done`（`confirmed_after_apply_error`）にする。
他者だけの編集や割り込みがあれば、その本文を巻き戻さず `supplement:failed` として確定する。
取得失敗・履歴の不整合、新しい編集がないのに元本文と違う場合、またはbundle欠落・不正で本人の編集がある場合は、`write_state_unconfirmed` で非ゼロ終了し、確定ラベルを付けない。
実行記録の解析失敗時は、本人の書き込みなしを確認できた場合だけ `invalid_execution_record` でfailedにし、それ以外は `write_state_unconfirmed` としてdone・failedを付けない。
pendingのまま親CLI・dispatcherのgateが止めるため、人間が状態を確認して判断するか `supplement:skipped` を付ける。
`supplement:failed` では親CLI・dispatcherは現在の本文で進められる。
summaryの理由を確認してから、合意・実行根拠の記録がまだ無いfailed issueだけをownerが再実行できる。
Actions → **Issue supplement** → **Run workflow** でmain・issue番号を選ぶか、次を使う。

```bash
gh workflow run issue-supplement.yml --ref main -f issue_number=123
```

手動の再実行ではprepareが合意の確認より先にfailedを外し、親CLI・dispatcherの確定待ちgateを閉じる。
生成中もpendingを保つ。生成中やPATCH直前に合意を見つけた場合は、本文を変えずskippedにする。
確定ラベルは個別の追加APIだけを使い、ラベル全体を置換しない。付与直前に別の確定ラベルがあれば停止する。
追加APIの直後にも再確認し、残る競合は明示的な失敗として検出する。人間が付けたラベルを消して修復しない。
done/skippedは手動実行でも補足を再実行しない。

`supplement:skipped` は「この本文で進めてよいと人間が判断した」か、「既に合意・実行根拠があるため補足を省略した」を表す。
runner待ちを省略する判断は人間が行い、GitHubのissue画面でこのラベルを付ける。
成功や失敗と競合しないよう、他の確定ラベルとは同時に付けない。
実行中に人間がskipした場合、後続の失敗jobはそれをfailedへ置き換えない。
補足追加workflowはコメントを投稿しない。

識別子は `issue_supplement`、CLIは `supplement`、親の待機は `wait-supplement`、
共有APIは `issues.supplement_state` / `SupplementState` / `SUPPLEMENT_*` に統一する。
旧package・CLI・APIへのaliasは設けない。切替と保存済み入力の扱いは[移行手順](#名称設定ラベルの移行)を参照する。

CLIは `.agents/ops` を `PYTHONPATH` にして `python -m issue_supplement prepare|supplement|apply|fail` を使う。
`--repo`・`--issue-number`・`--enabled-since` が必須、prepare以外では `--bundle` も必須。
`--attempt` は内部の1か2。supplementには新しい `--run-dir` を渡す。
`--manual` はprepareだけが受理し、read-onlyのsupplement/applyからfailed処理を迂回できない。
`GITHUB_OUTPUT` へ `status` と `needs_retry` を出し、静的なjobの連鎖を制御する。
通常処理は0、検査・処理の失敗は1、引数不正は2。failコマンドの0は確定ラベルの処理成功を意味し、
上流のworkflow失敗を成功には変えない。実機の導入・有効化・1件目の確認は別タスクで行う。

## 検証

```bash
.venv/bin/python -m pytest tests/unit/agents_ops/test_issue_supplements.py \
  tests/e2e/agents_ops/test_issue_supplement.py \
  tests/e2e/agents_ops/test_issue_supplement_state.py \
  tests/e2e/agents_ops/test_issue_content_edits.py \
  tests/e2e/agents_ops/test_agy_headless.py tests/e2e/agents_ops/test_agents_docs.py -n 2 -q
```

偽agy・偽ghの実行ファイルと一時state、注入した時計を使う。本物のagy・Googleログイン・
GitHubへの変更・GPUは使わず、安定待ちを実時間のsleepで検証しない。
実行時は `HERDR_*` を環境から除く。一時的な小さなGit repoで、元checkoutへの改変・追加・HEAD移動を
拒否し、作業領域だけの変更は許可し、正常・失敗時ともworktreeが残らないことを確認する。
依存部品の版・モデル・schema・read-only設定は既存の `test_agy_headless.py`、
対象状態は `test_issue_supplement_state.py`、待機の接続は `test_parent_session.py::test_wait_supplement_*` も使って確認する。

## 名称・設定・ラベルの移行

設定名は `ISSUE_SUPPLEMENT_ENABLED_SINCE` と `ISSUE_SUPPLEMENT_AGY_MODEL`。
旧 `ISSUE_REWRITE_ENABLED_SINCE` のUTC値と旧モデルpinの値をそのまま移し、
有効化境界やモデル選択は変更しない。新変数が未設定ならworkflowは起動しない。
権限エラーや不正値を旧変数へfallbackしない。

切替は次の順で行う。

1. 旧workflowのin_progress・queued・waiting等の未完了runを完了させる。切替中に手動runを開始しない。
   旧履歴は旧workflow名のまま保存する。新workflowは `issue-supplement.yml` と
   `issue-supplement #<番号>`、同名接頭辞のconcurrency/artifact/一時領域を使う。
   旧runをdrainしてから切り替え、concurrency groupの変更で同時に実行される窓を作らない。
2. 新repository variableへ旧値をコピーし、両値の完全一致を確認する。
   PRのmergeまでは稼働中のmainが使う旧変数も保持する。新workflow・親CLI・dispatcherの
   切替後に旧変数を削除する。新コードが旧変数を読む経路はない。
3. 旧 `rewrite:done` / `rewrite:failed` / `rewrite:skipped` の付与issueをopen/closedとも列挙し、
   本文・作成日時・全ラベルを退避する。各issueの旧確定ラベルが一意で、新確定ラベルと
   競合しないことを全件確認する。未知の `rewrite:*` や競合があれば変更せず止める。
4. 新ラベルが存在しなければ旧ラベル自体をrenameできる。既に両方ある場合は、各issueへ
   同じ状態の `supplement:*` だけを個別追加し、再取得で追加を確認してから旧ラベルだけを
   個別削除する。全ラベルの置換や、done/failed/skippedをpendingに戻す方法は使わない。
   付与issueが0件であることを再取得して旧定義を削除する。
5. 移行前後で全本文・対象外ラベル・確定状態を照合する。有効化前のissueは引き続き
   `not_target`。旧方式の完了を新方式の本文へ再処理せず、本文に一度もPATCHしない。
   共有判定は移行漏れの旧ラベルを明示的に拒否する。

dispatcherの保存済み `input.json` / `start-input.json` に旧 `rewrite_enabled_since` がある場合、
新キーとの併記でも再利用を拒否し、`supplement_enabled_since` を持つ入力の再収集を求める。
元JSON・digest・既存の実行根拠を直接書き換えない。旧候補を予約・起動せず、判断を再実行し、
起動前の全入力・実行根拠の鮮度照合をやり直す。過去の完了記録は証拠として保持する。

旧語を意図的に残すのは、この移行元名、旧ラベル/旧schemaを拒否する負例、
過去run・artifact・実測記録である。`scripts/ci_metrics/cpu-file-times.json` の旧ファイル名/nodeidは
測定時点の証拠なので置換しない。改名したテストは未測定として扱い、次の実CIから
[profile生成](../../../scripts/ci_metrics/sharding.md#regenerate-measured-weights)で測定の来歴付きで更新する。
Git履歴書換えやNext.jsの `rewrites()` 等、補足機能と関係のない用語も変更しない。

## 実機検証

`agy-smoke.yml` の `--supplement-contract` は同じpinのmarkdown-it-py・本番schema・
原文保存検査をモデル呼び出しなしで確認する。実Issueによる検証では、テスト用親issueの
原文を保存し、ownerの手動workflowからself-hostedのWriter/VerifierとGitHub-hostedのapplyを
実行する。handoffとGitHubの本文・編集履歴・新ラベルを照合し、原文完全保存と補足の出典を確認する。
本番workflowのowner/main制限は維持する。merge前の検証に限り、同じ処理を実行する
検証branchでref条件をそのbranchへ固定して手動起動し、検証後に条件を本番へ戻す。
