# ci_watch: merge後のmain CIの収集と分類

#1349 の観測入力（#1450）。GitHubからの取得は **`gh api --method GET` だけ**で行う。
agent、issue作成、rerun、workflow_dispatch、workflow変更、systemd操作は含まない。
issueの生成/再利用と修正後の回復判断は #1452、全体調査からの呼出しは #1447 の担当。
既存の [ops-report](../ops-report/README.md) や [review](../review/README.md) の状態は変更しない。

## 実行

```bash
# 初回: 観測開始のmain SHAを明示する（このSHAを含む）。省略するとエラー。
PYTHONPATH=.agents/ops .venv/bin/python -m ci_watch collect \
  --repo Motoki0705/tennis-lab --start-sha <full-main-SHA>

# 2回目以降: 新しいSHAの発見と、既存SHAのrun/attempt/jobの更新を1回行う。
PYTHONPATH=.agents/ops .venv/bin/python -m ci_watch collect --repo Motoki0705/tennis-lab

# workflow追加は初回に指定する。既定はci.yml。ファイル名だけ、複数指定可能。
PYTHONPATH=.agents/ops .venv/bin/python -m ci_watch --state-dir <別の状態ディレクトリ> \
  collect --repo Motoki0705/tennis-lab --start-sha <full-main-SHA> \
  --workflow ci.yml --workflow cuda.yml
```

既定の状態先は `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/ci-watch/`。
`--state-dir` はサブコマンドの前に置く。checkoutに依存しないローカル状態を使い、
同じ状態への収集は `collect.lock` で排他する。競合した起動は待たず非ゼロ終了する。
repository・workflow集合・開始SHA・schema versionの不一致やSQLite破損はエラーで止まり、
既存状態を作り直さない。観測範囲/設定を変える場合は明示的に別の状態先を使う。
GitHub CLIの認証は呼出し側で準備する。対象hostはgithub.comに固定。

`collect` はJSONを返す。`complete` は必要な結果の取得が完了したことを表し、CI成功を意味しない。
CIの失敗自体では終了コードを1にしない。未実行/進行中も終了コード0で `complete: false` と
`pending` のSHA一覧を返す。API失敗・JSON不正・件数不一致・権限不足は終了コード1で `errors` に
endpointと理由を記録する。中断後は同じ引数で再開できる。

## SHAと取得の境界

- 毎回 `refs/heads/main` をGETで固定し、first-parentを前回の発見cursorまで辿る。
  初回は指定の開始SHAまで辿る。mergeの他方のparentや、PRの途中commitは対象にしない。
  SHAの順序は日時に依存しない。cursorが祖先に見つからなければ履歴の書換え等として止める。
- 新しい連続SHAの登録と各SHAの `not_collected` 記録は同じtransactionで保存する。
  cursorは発見位置であり、成功済みCIの位置ではない。取得前の中断でも未取得SHAが残る。
  祖先探索途中の失敗ではcursorを進めず、SHA未特定の失敗も `collection` 記録に残す。
- 各SHAについて指定workflowのrun一覧を100件ずつ取得し、`total_count` と重複IDを照合する。
  件数変動、途中ページの欠落、workflow run検索の1000件上限への到達は取得失敗にする。
  `main` の `push` / `workflow_dispatch` / `schedule` の既存runを観測し、PR用runは除く。
- 各runの全attemptと、それぞれのjobの全ページを取得する。SHA・run・attempt・workflow・jobの
  所属を照合する。同名jobが同じattemptに複数ある場合は再試行の対応が曖昧なためエラーにする。
  失敗jobのログもGETで取得する。ログ期限切れ等も未取得として記録し、原因を推測しない。
- 完了済みSHAも毎回照会する。古いSHAの遅延runや、人間による後日の再試行を期限で切り捨てない。
  取得済みの完了jobログはローカル原本を検査して再利用し、API失敗を挟んでも保持する。
  API呼出し回数は履歴に比例するため、初回に必要な開始SHAを指定する。

## 保存形式と後続からの取得

`observations.sqlite3` に設定、重複のないSHAの連鎖、最後に取得した事実、追記式のイベントを保存する。
最後に検査した時刻と内容の変わった時刻を区別する。同じ事実・分類の再収集では観測イベントを重複追加しない。
再試行、取得失敗からの復帰、後続SHAによる分類の変化は新しい版として追記し、過去の版を保持する。
APIレスポンスとログのUTF-8テキスト原本は `objects/<object ID>` に保存し、内容からIDを計算する。
原本の読み出し時は内容を再検査する。欠落・改変はエラーになり、自動復元しない。

後続はSQLiteへの直接書込みをせず、次のローカル読み取りを使う。いずれもGitHubへのアクセスはない。

```bash
# 追記記録をsequence順に範囲取得。next_afterを次の--afterに使う（境界はexclusive）。
PYTHONPATH=.agents/ops .venv/bin/python -m ci_watch records --after 0 --limit 100
# 特定SHAの履歴/現在の版
PYTHONPATH=.agents/ops .venv/bin/python -m ci_watch records --sha <full-SHA> --limit 10
PYTHONPATH=.agents/ops .venv/bin/python -m ci_watch show --sha <full-SHA>
# 原本参照のobjectから必要な行だけを取得。start-lineは1始まり。
PYTHONPATH=.agents/ops .venv/bin/python -m ci_watch evidence --object <object-ID> \
  --start-line 1 --limit 100
```

`records` の返却は `schema_version: 1`、`config`、`records[]`、`next_after`、`has_more`。
`--limit` は1〜1000。各記録は次の形で、`kind` は `observation` または `collection`。

| フィールド | 内容 |
|---|---|
| `sequence` | 状態内の単調増加ID。後続が処理済みcursorとして使う |
| `kind`, `sha`, `observed_at` | 記録種別、対象SHA（collectionはnull）、版の保存時刻（UTC） |
| `observation` | SHAの観測内容、またはcollection全体の状態/取得エラー |

`observation` のSHA観測内容:

| フィールド | 内容 |
|---|---|
| `sha`, `parent_sha`, `url`, `committed_at`, `source` | first-parent上のcommitとGET原本参照 |
| `status` | `not_collected` / `pending` / `fetch_error` / `complete` |
| `result` | 未完了/取得失敗はnull、全対象の最新attemptが成功なら `success`、それ以外は `non_success` |
| `workflows[]` | `workflow`、`status`（`not_collected` / `no_run` / `incomplete` / `fetch_error` / `complete`）、`sources`、`runs`、`failures`、`errors` |
| `workflows[].runs[]` | `run_id`、`attempt`、`event`、`status`、`conclusion`、`url`、`source`、`job_sources`、`jobs`。過去attemptも含む |
| `runs[].jobs[]` | `id`、matrixを含む`name`、`status`、`conclusion`、`url`、`source`、`steps`、`log`（未取得はnull）、`failures` |
| `errors[]` | 取得失敗の `endpoint` と `reason`。既知の結果を併記してもSHA全体は成功にしない |
| `failures[]` | 全workflow/attemptの失敗発生と分類（下記）。同じ署名の別発生を保持 |

原本参照は `endpoint` と `object`。ログ未取得の参照はendpointだけで、取得失敗の理由は `errors` にある。
`show` は最新の観測版に加え `sequence`、`observed_at`、`checked_at`（最後のSHA取得時刻）を返す。
未取得の `checked_at` はnull。存在しないSHA/原本を「対象なし」の成功に変換しない。
`collection` はhead/cursor/取得件数/未完了SHA/エラーの全体状態を記録する。
祖先取得失敗では既存SHAの結果を消さないため、後続はSHA観測と併せてcollectionのエラーも確認する。
一時失敗から復帰した収集もcollectionの新しい版になり、未解消の取得障害と区別できる。

## 失敗署名と分類

署名のversionは `ci-failure-v1`。workflowファイル名、matrixを含むjob名、失敗したpytest node ID、
エラー要約から作る。ログの時刻/ANSI、メモリアドレス、Pythonの行番号は正規化する。
pytestの短いsummaryがあればテスト単位、それ以外はjob単位にする。
同一署名の発生IDは `(sha, run_id, attempt, job_id, signature)`。
後続 #1452 は署名で照合し、発生IDとsequenceで二重処理を防げる。issueへの書込みはこのCLIでは行わない。

各失敗は `signature`、`workflow`、`job`、`test_id`、`error`、`failed_steps`、`sha`、`run_id`、
`attempt`、`job_id`、`job_url`、`job_conclusion`、`evidence`、`infrastructure_evidence`、`classification` を持つ。
根拠には原本参照・1始まりのログ行番号・抜粋がある。行番号nullはjobの結果から生成した要約であり、
ログを取得した証拠ではない。対応する失敗jobがないrun失敗はjob名 `<workflow>`、job ID nullとして残す。

| `classification.category` | 判定根拠 | 確信度 |
|---|---|---|
| `runner_infrastructure` | 明示的なrunner通信/停止エラー、または失敗した依存取得/設定stepと通信/容量エラーの併存 | high |
| `flaky` | 同SHA・同runの後のattemptで同名jobがsuccess。skipped/neutralや別runのsuccessは使わない | medium |
| `code_regression` | 隣接するmain SHAで、同じテスト/具体的なコードエラーの署名が継続 | medium |
| `unclassified` | 不足・未知・一般的なexit codeだけ等 | low |

`classification` は `confidence`、`assessment: hypothesis`、`reasons[]` を持ち、
規則名と根拠のSHA/run/attempt/job/原本を残す。明示的な基盤エラー、同SHAの再試行成功の順に優先する。
テストjob内の一般的なnetwork例外だけでは基盤障害にしない。run欠落のSHAを飛び越して継続失敗と判定しない。
隣接SHAの同じ失敗に再試行成功の反証がある場合も、回帰と決めず分類不能として反証を残す。
同じjobの成功は個別テストの実行や外部状態の一致を証明しないため、flakyと回帰は確定原因にしない。
分類不能を空配列や成功に読み替えない。回復の確定や関連変更の調査は後続へ渡す。

## 周期・能動的な実行

定期実行は既定で無効で、unit/timer自体を追加していない。周期値も持たない。
人間が周期・起動方式を決めた後、必要なら [install_units.sh](../install_units.sh) に沿って
`systemd/` のservice/timerを別途追加し、有効化を人間が行う。workflow_dispatchとrerunも判断待ち。
この実装は既存runを読み取るだけで、それらを能動的に起動する設定/コマンドを持たない。

## 検証

`tests/e2e/agents_ops/test_ci_watch.py` と専用fake ghで、連続SHA、ページ境界、中断/再開、
取得失敗、同SHAの再試行、4分類、原本参照とGET限定を検証する。実GitHub/agentはテストで起動しない。

```bash
bash .agents/ops/pytest_locked.sh tests/e2e/agents_ops/test_ci_watch.py -q
```
