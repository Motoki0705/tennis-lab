# 親issue本文の書き直し

[`issue-rewrite.yml`](../../../.github/workflows/issue-rewrite.yml) は、作成直後の親issueの本文を
Geminiで書き直す。何を書くかは起票者、表現・言い換え・説明構造はGeminiが担当する。
用語・参照先の短い説明だけを補足できる。起票者への書き方の指示は追加しない。

## 起動と状態

repository ownerによる `issues.opened`（`issue:parent` がある場合）、
`issues.labeled`（`issue:parent` を付けた場合）、mainへの `workflow_dispatch` だけを受け付ける。
`ISSUE_REWRITE_ENABLED_SINCE` が未設定ならjobを起動しない。
対象・確定状態・有効化時刻・ラベル定数は
[`issues.rewrite_state`](../issues/README.md#親issueの書き直し状態)を使う。
workflowは `vars.ISSUE_REWRITE_ENABLED_SINCE` の値をCLIの `--enabled-since` へ明示的に渡し、
共通の `rewrite_state` で日時と状態を検証する。空値は未設定として渡す。
Variables APIの追加権限を持たないjobでは `read_enabled_since` を呼ばず、権限エラー後のfallbackも行わない。
workflow以外の親CLI・dispatcherの取得契約は変更しない。
有効化前のissue、親ラベルのないissue、処理済みの自動イベントには何も書き込まない。
子も兼ねる親は親・子両方の本文書式を検査する。

同じissueのrunをconcurrencyで直列化し、実行中のrunは取り消さない。
workflowファイル名 `issue-rewrite.yml` と表示名 `issue-rewrite #<番号>` は
[`parent wait-rewrite`](../parent/README.md#書き直しの確定待ち)との共有契約。
本文競合によるattempt 2も同じrun内で実行し、runの表示名とowner起点の認可を保持する。
wait-rewriteはこの1つのrunが全jobを終えるまで待つ。別runの起動や追跡は行わない。

## 処理と責務

| 部品 | 処理 |
|---|---|
| `pipeline.prepare` / job `prepare` | GitHub-hosted runnerで手動のfailed再実行をpendingに戻してから合意を確認する |
| `pipeline.generate` / job `rewrite`・`rewrite_retry` | self-hosted runnerで入力を保存し、安定待ち・writer・不変条件・verifierを実行する。GitHubへの変更権限は持たない |
| `invariants.py` | 元の本文から保持すべき情報を抽出し、違反した項目と理由を列挙する |
| `github.py` | 既存の合意・実行根拠の解析、GitHubへの要求、編集履歴の全ページ取得と照合 |
| `publication.py` | 公開する項目の選別、token等の検査、summaryの出力。生のagent診断は公開しない |
| `pipeline.apply` / job `apply`・`apply_retry` | GitHub-hosted runnerでartifactを再検査し、最新入力との照合・PATCH・編集履歴の検査・確定ラベルの付与を行う |
| `pipeline.fail` / job `failed` | 上流jobの失敗時に、GitHub-hosted runnerで現在の対象・合意・確定状態を再取得して失敗を確定する |

1. ownerによる手動再実行でfailedの場合、prepareが現在の状態を読み直してfailedラベルだけを個別に削除し、
   pendingを再取得してから合意・実行根拠を確認する。記録済みならskippedを付けて終了する。
   done/skippedや非対象は再実行しない。通常のpendingでも生成前に記録を確認する。
   title・body・labels・created_at・作者・全コメントをprivateなrun dirへ保存する。
   本文が60秒変わらなくなるまで10秒以下の間隔で確認する。10分以内に安定しなければ失敗。
2. repository ownerをpublisherとして `parent.policy.events` の合意・実行根拠と、
   `parent.judgement_comments.owned_comment` が完全検証した `startable` の判定記録を調べる。
   既に記録があれば本文を変更せず `rewrite:skipped` にする。引用・別publisher・保留判定は実行根拠にしない。
   壊れた所有コメントは解析失敗とし、書き直さず `rewrite:failed` にする。
3. writerとverifierは、[`agy_headless.py` の公開API](../lib/README.md#agyの共通呼び出し部品)を使う。
   版・モデルは同部品と `agents.conf` のpinだけに従う。呼び出しごとに別の会話とlogを作り、
   repo・関連issue/PRはGemini自身が読み取る。1呼び出しの上限は300秒。
4. writerの構造化出力をschemaで検査してから不変条件を調べる。通った本文だけを別会話のverifierへ渡す。
   `verdict=pass` かつ `differences=[]` だけが意味検査の合格。自由文やCLIの終了値から結果を補わない。
5. 不変条件か意味検査で落ちた場合、直前の本文と具体的な理由をwriterへ渡して1回だけ修正する。
   schema不正・構造化出力なし・版/モデル/認証等の呼び出し失敗は修正を試さず失敗する。
6. applyはartifactのschema、不変条件、verifier合格を再検査し、対象状態と合意記録も再取得する。
   元本文と現在の本文、入力時と現在の編集履歴が一致したときだけ本文をPATCHする。
   不一致なら書かず、applyの `needs_retry=true` outputを受けた静的な `rewrite_retry`（self-hosted）と
   `apply_retry`（GitHub-hosted）が最新本文から1回だけ作り直す。attempt 2も不一致なら失敗する。
   dispatchは行わず、最終label以外のGitHub変更は手動再実行のfailed削除と本文PATCHに限る。
   本文PATCHの直前の最後の読み取りでも、合意・実行根拠の記録を確認してから書く。
7. PATCH後にGraphQLの `userContentEdits` を全ページ取得し、以前の履歴が変わっておらず、
   今のGitHub tokenのviewerによる編集が1件だけ増えたことと現在の本文を確認して `rewrite:done` にする。
   GitHubが最初の編集時に作成時の版も追加する場合は、`includesCreatedEdit` のfalse→true、
   変更前の空履歴、作成時刻・作者が一致する一意の作成記録を照合して、その記録を区別する。
   同一作者の追加編集、履歴の欠損・取得失敗・曖昧さも不合格。

不変条件は、必須 `##` 節の集合と順序、`validate_body`、
`parent.execution.PROGRESS_SECTIONS` の節内の原文（前後の空白を含む）、
引用ブロック全体（`>`を省略したlazy continuationを含む）、HTMLコメント、fenced code、inline codeの原文と順序を保持する。
`#番号`・HTTP(S) URL・`@mention`・数値は多重集合で比較する。
引用範囲とURLはmarkdown-it-pyのCommonMark解析を使い、括弧を含むリンク先を完全に取得する。
参照リンクの使用回数をURL出現数へ重複計上せず、強調や句読点をURLへ取り込まない。
workflowの隔離Pythonにも `markdown-it-py==4.0.0` を明示する。
数値の字面も保持するが、行頭のMarkdownの番号付きリストの番号は除く。
checkboxの数とchecked/uncheckedは節ごとに比較する。段落・箇条書き・表・小見出し等の表現はGeminiに任せる。

編集履歴APIのフィールドは[GitHubのIssueの仕様](https://docs.github.com/en/graphql/reference/issues#issue)と
[UserContentEditの仕様](https://docs.github.com/en/graphql/reference/users#usercontentedit)に従う。
RESTの本文PATCHは原子的なcompare-and-swapではない。PATCH後に割り込みを検出した場合は失敗を記録し、
**他の編集を上書きする巻き戻しは行わない**。原文と各変更はGitHubの編集履歴から確認する。
検査段階での失敗では本文を一度も書き換えない。

## 公開するデータ

runnerのHOME・agyの場所・導入と直接ログインの手順は[runner README](../../../scripts/github_actions/README.md#agyの導入と疎通)が正本。
rewrite jobのGitHub tokenはcontents・issues・pull-requestsのreadだけ、checkoutはcredentialを永続化しない。
prepare・apply・apply_retry・failedはissuesのwriteとcheckoutのcontents readだけで、
agyの導入やGoogle認証情報を持たない。全jobからactionsのwriteを外している。
外部actionは公式release tagと照合した完全commit SHAと版コメントで固定する。

run dirは0700、保存ファイルは0600。入力・prompt・writerのnotes・verifierの説明・agy logはprivateに保つ。
artifactへ渡すのは `public/handoff.json` 1ファイルだけで、globやrun dir全体をuploadしない。
合格時は比較用の元本文、検査済み本文、履歴のID・時刻・作者、空差分の検証結果を含む。
失敗した候補本文やコメント、log、会話ID、ローカルパス、認証情報は含めない。

本文はartifact作成前にGitHub token、Google access/refresh token等の既知の形を検査する。
ローカルパス・会話/session IDの表記・UUIDも公開を拒否する。既知の形による保守的な検査なので、
原文の引用にこれらがある場合も自動処理を失敗させ、消した本文を黙って適用しない。
公開JSON全体も同じ検査を通し、applyで再検査する。
summaryには状態、検査理由、意味差分の種類と実際の本文から取れた引用、usage・所要時間を出す。
verifierの自由文説明や拒否コマンドの生文字列は出さない。usage未取得はnull/未取得のまま扱い、0に補わない。
過去revisionの不合格理由は、修正成功後もrevision付きで残す。

## 失敗、skip、手動の再実行

`rewrite:failed` は「原文で確定」の意味で、親CLI・dispatcherはこの確定状態で進められる。
summaryの理由を確認してから、合意・実行根拠の記録がまだ無いfailed issueだけをownerが再実行できる。
Actions → **Issue rewrite** → **Run workflow** でmain・issue番号を選ぶか、次を使う。

```bash
gh workflow run issue-rewrite.yml --ref main -f issue_number=123
```

手動の再実行ではprepareが合意の確認より先にfailedを外し、親CLI・dispatcherの確定待ちgateを閉じる。
生成中もpendingを保つ。生成中やPATCH直前に合意を見つけた場合は、本文を変えずskippedにする。
確定ラベルは個別の追加APIだけを使い、ラベル全体を置換しない。付与直前に別の確定ラベルがあれば停止する。
追加APIの直後にも再確認し、残る競合は明示的な失敗として検出する。人間が付けたラベルを消して修復しない。
done/skippedは手動実行でも再度書き直さない。

`rewrite:skipped` は「この本文で進めてよいと人間が判断した」か、「既に合意・実行根拠があるため書き直さなかった」を表す。
runner待ちを省略する判断は人間が行い、GitHubのissue画面でこのラベルを付ける。
成功や失敗と競合しないよう、他の確定ラベルとは同時に付けない。
実行中に人間がskipした場合、後続の失敗jobはそれをfailedへ置き換えない。
書き直しworkflowはコメントを投稿しない。

CLIは `.agents/ops` を `PYTHONPATH` にして `python -m issue_rewrite prepare|rewrite|apply|fail` を使う。
`--repo`・`--issue-number`・`--enabled-since` が必須、prepare以外では `--bundle` も必須。
`--attempt` は内部の1か2。rewriteには新しい `--run-dir` を渡す。
`--manual` はprepareだけが受理し、read-onlyのrewrite/applyからfailed処理を迂回できない。
`GITHUB_OUTPUT` へ `status` と `needs_retry` を出し、静的なjobの連鎖を制御する。
通常処理は0、検査・処理の失敗は1、引数不正は2。failコマンドの0は確定ラベルの処理成功を意味し、
上流のworkflow失敗を成功には変えない。実機の導入・有効化・1件目の確認は別タスクで行う。

## 検証

```bash
.venv/bin/python -m pytest tests/e2e/agents_ops/test_issue_rewrite.py \
  tests/e2e/agents_ops/test_parent_session.py -k 'rewrite or parent_body_update' -n 2 -q
```

偽agy・偽ghの実行ファイルと一時state、注入した時計を使う。本物のagy・Googleログイン・
GitHubへの変更・GPUは使わず、安定待ちを実時間のsleepで検証しない。
依存部品の版・モデル・schema・read-only設定は既存の `test_agy_headless.py`、
対象状態は `test_issue_rewrite_state.py`、待機の接続は `test_parent_session.py::test_wait_rewrite_*` も使って確認する。
