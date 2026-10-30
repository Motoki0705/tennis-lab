# 子workerの実行状況（読み取り専用）

親registryの子taskと、共通ランナーの保存記録から公開可能なJSONを作る。
Web・FastAPI依存はなく、掲示板の実行状況タブとCLIが同じPython APIを使う。
担当・委任・承認・実行状態を変更する操作、agent起動、停止、再開、mergeは持たない。

## Python APIとCLI

repo rootをcwdとして実行する。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m worker_status --parent 1445
PYTHONPATH=.agents/ops .venv/bin/python -m worker_status --parent 1445 --github
PYTHONPATH=.agents/ops .venv/bin/python -m worker_status --include-inactive --stale-after 600
```

```python
from worker_status import StatusError, collect_status

local = collect_status(parent=1445, github=False)
refreshed = collect_status(parent=1445, github=True)
children = refreshed["parents"][0]["children"]
```

`collect_status` はJSON化できるdictを返す。失敗は公開可能なコードだけを持つ
`StatusError`。CLIは成功JSONだけをstdoutへ出し、失敗はstderrの`{"error":"code"}`と
終了値2で返す。途中まで取得したJSONを成功として返さない。CLIの引数構文エラーも終了値2。

| 引数（CLIはハイフン表記） | 意味・既定値 |
|---|---|
| `parent` / `repo` | 親番号 / `owner/repo`の絞り込み。未指定は全repoの対象親。repo特定のためにghを呼ばない |
| `state_dir` | 親stateのディレクトリ。未指定は`parent.state.state_directory()`のXDG state |
| `dispatcher_state_dir` | dispatcher state。未指定は親stateと同じ親ディレクトリの`dispatcher` |
| `include_inactive` | 既定false。通常はactive・handoff_pendingのみ。trueならended・releasedも含む |
| `github` | 既定false。trueの呼出しだけGitHubをGETする。キャッシュは1回の取得内だけ |
| `stale_after` | 更新途絶の閾値、既定300秒。有限の正数。CLIは`--stale-after` |
| `tail_bytes` | イベントと受信時刻の各ファイルの末尾を読む上限、既定262144 bytes。1024〜1048576 |
| `tool_limit` | 直近ツール数、既定5。1〜20 |
| `now` | Python APIだけ。timezone付きdatetimeで観測時計を注入できる。既定はUTC現在時刻 |

ローカル表示の更新は`github=False`、人間のGitHub更新操作だけ`github=True`にする。
自動polling・daemon・Webの表示周期はこのパッケージでは実装しない。

## 入力と読み取り境界

- 親registryは`parent.state.Store.read_data()`を使う。version 1/2を読み、後者は親別JSONと
  所有索引を照合する。operation・registry・session lockを取らず、作成・初期化・saveを呼ばない。
  照合対象は出力に含める各親の行だけ。repo・parent・担当状態のフィルタで対象外となる親の
  更新は取得を失敗させない。破損・未知schemaなどregistryそのものの検証失敗は従来どおりエラー。
  対象候補は呼出し開始時に選び、各親の取得直前にその行を読み直す。直前に担当終了した親は
  `include_inactive=False`なら除外し、新たに追加された親は次の呼出しから対象にする。
- GitHubは各親の取得直後にその親の行を取得前と比較する。変更があれば、その親だけ最新行を
  読み直して**1回だけ**GitHubを取り直し、旧結果は捨てる。取り直しも変化した場合や対象行が
  消失した場合は`registry_changed_retry`で全体を失敗させ、部分成功を返さない。
  再取得ではGETキャッシュを引き継がず、新旧担当の私的な識別子はどちらも秘匿検査に保持する。
  各親の担当世代・task・GitHub結果を一組として確定し、別の親の取得中に以前の親が更新されても
  確定済みの観測を無効にしない。複数親の状態は同一瞬間のsnapshotではない。
- `github=False`も各親のローカル投影の直前・直後にその行だけ照合する。対象行が変化したら
  `registry_changed_retry`で全体を失敗させる。ローカル取得には自動再試行を加えない。
- 手動workerはtaskの`worker_runs`を読む。形式・保存側の契約は#1548。
  現在の`delegation_id`と一致する記録の最後の要素を選び、古い委任へ戻らない。
  相対runパスは親state内、外部runは絶対パスでなければならない。
  `worker_runs`キーがある委任で現在の委任IDに対応するrunがなければ、旧委任のrunが残っていても
  未起動とする。キーのない旧委任は「記録なし」であり、読み取りでキーを補わない。
- dispatcherのscheduleが存在すれば`Schedule.read()`で検証し、registryの
  `dispatch.jobs`から現在の委任・子に一致する最大sequenceのjobの`runtime`を読む。
  手動の`worker_runs`があればそちらを優先する。scheduleがなければ
  `dispatcher.status=unused`・「自律dispatcher未使用」。壊れたscheduleを未使用扱いにしない。
- 選択したrunの`result.json`の`contract.task_id/cwd/branch`を、現在の委任ID・子taskの
  worktree・branchと照合する。dispatcherのjobの場合は委任IDの代わりにそのjobのtask_idを使う。
  cwdは絶対パスの実体、残る2項目は文字列の完全一致。取得側・期待値側の欠落も一致とみなさず、
  状態不明と`contract_missing`または`contract_mismatch`、不一致フィールド名を返す。
  resultがない場合だけinvocation.jsonの同じ契約を照合する。照合できてもinvocationだけから
  実行中とは推測しない。破損したresultをinvocationで置き換えない。
- 各runの`result.json`と初回runの`session.json`はそれぞれ最大1 MiB。
  `events.jsonl`と`event-receipts.jsonl`は末尾だけ有限byteで読む。
  上限境界にかかる最初の断片は捨て、省略を`summary.tail_truncated`で示す。
  stdout原文と受信時刻の対応はbyte範囲とdigestで照合する。
  記録形式は[共通ランナーREADME](../lib/README.md#記録と結果jsonschema_version-1)が正本。
- prompt・stderr・final全文・Codex transcript・CODEX_HOMEは読まない。
  要約はCodexのユーザー向け`agent_message`と許可したツールのフィールドだけ。
  reasoning本文・ツール出力・MCP引数・変更ファイルのパスは出さない。

## 状態と根拠

| `children[].status` / `label` | 判定 |
|---|---|
| `running` / 実行中 | `result=running`かつrunnerのpid・boot ID・start ticksが`lib.opsproc.process_identity`と全一致 |
| `turn_completed` / ターン終了 | `result=completed`・`turn_end_reason=completed`。タスクの合格・Issue完了ではない |
| `failed` / 失敗 | `result=failed`。終了理由は`run.turn_end_reason` |
| `stopped` / 停止中 | taskの明示停止記録、または明示GitHub取得で確認した停止マーカー。プロセス終了を意味しない |
| `dependency_wait` / 依存待ち | 未起動・未委任の子で、ローカル依存先に停止記録、またはGitHubの既存launch_planに依存待ち理由がある |
| `not_started` / 委任予約済み・未起動 | 委任済みで`worker_runs`キーがあり現在の委任のrunがない（旧委任のrunだけ残る場合も含む）、またはdispatcherの未起動予約 |
| `not_delegated` / 未委任 | 委任予約がない |
| `unknown` / 状態不明 | run記録方式のキーがない旧委任、runner契約の不一致・欠落、runner喪失・identity不一致、JSON破損・未知schema・記録間不一致など |

既存の委任に`worker_runs`キー自体がなければ、未起動と推定せず「記録なし」の状態不明にする。
最新runが壊れていても古い正常runへ戻らない。現在のrun/sessionの対応も照合する。
既に起動したrunの状態を、依存先が現在未完了であることだけで依存待ちに置き換えない。

`evidence[]`は`source`（parent_registry.task / parent_registry.worker_runs /
parent_registry.dispatch.jobs / runner.result / runner.invocation / runner.session / runner.records / github.issue /
github.launch_plan）、`fields`（判断に使った入力フィールド名）、`code`（判断理由）を持つ。
入力フィールド名は説明用で、フィールドの私的な値をコピーしない。

## 出力フィールド

時刻はすべてtimezone付きUTC ISO8601。未取得を現在時刻やmtimeで補完しない。

| フィールド | 意味 |
|---|---|
| `schema_version` | この公開JSONのversion、現在1 |
| `observed_at` | ローカルの観測時刻 |
| `stale_after_seconds` | この観測で使った閾値 |
| `dispatcher.{status,label}` | scheduleの有無と説明。availableでも個々のjobの実行中を意味しない |
| `github.{status,fetched_at}` | not_requestedなら未取得・時刻null。observedなら今回のGETが全て成功した取得時刻 |
| `parents[]` | repo・親番号順の親。`repo`、`parent`、`assignment`（担当状態）、`generation`（担当世代）、`source=parent_registry`、`children` |
| `children[].child` | 子issue番号。ローカルはregistryの子、GitHub指定時は正式な子も含む |
| `status` / `label` / `evidence` | 上表の状態コード・日本語表示名・判定根拠 |
| `delegation` | 表示用の別名。生delegation IDの一方向digestから作り、再委任で変わる。IDのない旧委任には子番号の別名。委任なしならnull |
| `run` | 最新runの公開記録。未取得ならnull |
| `run.{alias,kind,recorded_at,source}` | 表示別名、initial/resume/registered、registry記録時刻、対応の取得元。dispatcherのrecorded_atは未取得null |
| `run.{result,turn_end_reason}` | ランナーのrunning/completed/failedと終了理由。task_successを推定しない |
| `run.identity_verified` / `identity_status` | 保存したrunner identityとの生存照合結果。statusはidentity_verified / identity_unavailable / runner_absent / identity_mismatch。PIDや識別情報そのものは含まない |
| `run.{started_at,finished_at}` | ランナーの開始・終了時刻。保存値なしはnull |
| `run.files_updated_at` | result/session/eventsのファイル更新時刻。イベントの受信時刻とは別 |
| `run.receipt_status` | observed（受信時刻記録あり）、unavailable（旧runでなし）、partial（末尾書込み途中） |
| `observed_at` / `last_activity_at` | 子の観測時刻 / 受信時刻記録の最終時刻。本文が読取上限を超えるイベントも時刻を取得できる |
| `activity_time_status` | observedまたはunavailable。旧runの最終活動時刻は未取得null |
| `updates_stalled` | 実行中かつ受信時刻があり、無更新時間が閾値を**超えた**場合true。閾値ちょうどはfalse。判定不能・実行中以外はnull。停止や失敗に読み替えない |
| `dependencies[]` | ローカル委任記録の依存子番号`child`と`status`。停止記録ありはstopped、それ以外はunobserved。ローカルだけではGitHubの完了・利用できる依存PRを推定しない |
| `stop_record` | 明示停止のreason/facts/impact/resume_condition/next_action。値は下記の秘匿済みtext object。通常は空object |
| `summary.message` | 読めた範囲の最後のユーザー向け発言先頭300字。text objectにreceived_at・time_statusを付ける |
| `summary.tools[]` | 直近ツール。kind・received_at・time_status。command_executionはcommand先頭行、mcp_tool_callはserver/tool、file_changeはchange_countだけ |
| `summary.tail_truncated` | 末尾読取り上限により省略された可能性がある |
| `github`（子） | 未指定時は`{status:not_requested}`。指定時は下記のGitHub観測 |
| 子`github.issue` | state・state_reason。ターン終了とは独立 |
| 子`github.pull_requests[]` | closing指示で子に対応するPRのnumber・state・merged・url。単なるRefsやbranch名では対応付けない |
| 子`github.launch` | 既存parent launch_planのaction・state・dependencies・reasons。理由はcodeと秘匿済みmessage。起動の許可ではない |
| 子`github.{stopped,stop_record}` | GitHubの最新停止マーカーによる停止状態と理由・再開条件・次の対応など。取得時刻はトップレベルのgithub.fetched_atを共有 |

text objectは`{status:visible|hidden|unavailable, text:string|null, truncated:bool}`。
`hidden`は秘匿検査により**非表示**、`unavailable`は**未取得**。どちらも本文はnull。
messageとツールには`received_at`と`time_status=observed|unavailable`を付け、
旧runのツール時刻は未取得と明示する。同じitem IDのツール更新は最新の1件へまとめる。
ユーザー向け発言が末尾範囲に無ければmessageは未取得。

`command_execution.command`は、展開なしに「シェル・オプション・1引数」と確定できる
`/bin/bash -lc`・`bash -lc`・`/bin/sh -c`・`sh -c`だけを1段外す。
外側の引用符はbashの単語分割・引用符除去と一致する限定した規則で解く。
単一引用符内は文字どおり。二重引用符内のバックスラッシュは`$`・backtick・`"`・`\`の前だけで
取り除き、バックスラッシュ直後の改行は両方を取り除く。それ以外の前のバックスラッシュは保持する。
単一・二重引用符で囲んだ部分の連結は許可するが、二重引用符内の未エスケープの`$`・backtick、
引用符外の展開文字（`$`・backtick・glob・tilde・brace）、バックスラッシュ・コメント・演算子・改行は
外さない。未知のシェル・オプション、引数の過不足、閉じていない引用符も元の文字列をそのまま検査する。
コマンドの実行・環境変数展開・再帰的なラッパー除去はしない。

引用符を解いた内側コマンドの**全文**を従来の秘匿基準で検査する。コマンド表示だけには追加で、
`-`から始まる語の中で英数字の直後に`/`が続くオプション直結パス
（`-C/path`・`-I/path`・`-L/path`・`-o/path`・`-xfoo/path`など）もhiddenにする。
通常の相対パス引数（`src/utils`・`tests/e2e/...`）とHTTP(S) URLはこの追加検査の対象外。
URLの除外は引用符とシェル演算子の境界を考慮し、後続の別コマンドのパスを覆い隠さない。
全文の検査後、先頭行を最大300字で表示する。text objectの`status`・`text`・`truncated`と時刻の契約は変えない。

## 安全契約と失敗

[掲示板の既存public検査](../board/safety.py)を直接再利用する。
構造検査と自由文の非表示検査は、次のように対象を分ける。

| 対象 | 検査する値 |
|---|---|
| `Guard.check(output)`：キー・状態コード・ラベル・URLを含む出力全体 | boardのpublic検査（保存済み親のsession、3つのCLI session/thread環境値、UUID/ID表記・絶対パス・制御文字）に加え、観測済みのsession/turn/run/delegation/job ID・worktree/runパス、下記identity系環境値、既存の認証token表記 |
| `Guard.text(...)`：発言・コマンド・ツール名・停止理由などの自由文 | 上記の構造検査と、起動環境の**8文字以上の全環境値**。全文を検査してから先頭300字・コマンド先頭行へ切り詰める |

identity系環境値は`CLAUDE_CODE_SESSION_ID`、`CLAUDE_CODE_PARENT_SESSION_ID`、
`CLAUDE_CODE_AGENT_ID`、`CODEX_SESSION_ID`、`CODEX_THREAD_ID`、
`AGENT_RUNTIME_RUN_DIR`、`AGENT_RUNTIME_TASK_ID`。空でなければ長さによらず両方で検査する。
再取得前後の担当・run参照も検査対象へ保持する。

一般の`GITHUB_EVENT_NAME`・`USER`・`LOGNAME`・`WSL_DISTRO_NAME`・`LANG`などの環境値は、
自由文の非表示判定だけに使う。コードで定義したキー・状態コード・ラベルとの部分一致によって
出力全体を拒否しない（例：`GITHUB_EVENT_NAME=pull_request`と`pull_requests`キー）。
不合格の自由文は本文全体をhiddenにし、明示的な私的情報を含むキーやURLなどの構造化フィールドは
`unsafe_public_field`で取得全体を失敗させる。通常の出力にもエラーにも私的な原文を出さない。
これは許可したフィールドの最小投影と既存秘匿検査による防御であり、任意の秘密情報の完全な分類器ではない。

主なエラーはinvalid_query / invalid_repository / clock_requires_timezone / registry_missing /
invalid_registry（ほかregistry構造の分類）/ registry_changed_retry / unsafe_public_field /
local_observation_failed / github_observation_failed。壊れた単独runは子をunknownとして根拠を返す。
registry・schedule・GitHubの取得失敗は空の成功へ変えない。

GitHubは`parent.github.GitHub`のsnapshot/family取得と`ParentSession.launch_plan`を再利用する。
呼出し内のGETだけをキャッシュし、writeは明示拒否する。POST・PATCH・GraphQL mutationを呼ばない。
GitHub認証・OSのstate読取権限は既存実行ユーザーのものを使い、権限を拡張しない。
実装側の権限不足・HTTP失敗・未知schemaは、私的なstderrを出さず取得失敗になる。

## 検証

```bash
bash .agents/ops/pytest_locked.sh tests/e2e/agents_ops/test_worker_status.py tests/e2e/agents_ops/test_worker_status_manual_runs.py tests/e2e/agents_ops/test_manual_worker_runs.py tests/e2e/agents_ops/test_agent_runtime.py -n 4 -q
```

一時registry/run、fake CLI/fake gh、固定時計、現在のテストプロセスのidentityを使うCPUテスト。
実agentを起動せず、実registry・run・GitHubは変更しない。
結合テストはparent CLIのdelegate・run-worker・record-worker-runが作った記録を使い、
未起動→実行中→ターン終了/失敗、外部runの登録、旧runを保持した再委任後の未起動を確認する。
fake runnerの実行中区間はsocketで同期し、worker_runsをテストから手書きしない。
