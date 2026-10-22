# Headless agent共通ランナー

`agent_exec.sh` はこのcheckoutの `.venv/bin/python` で `agent_runtime.py` を実行する。
Linux（`/proc`、flock、child subreaper）を前提に、Claude/Codexの1ターンを実行・保存する。
担当registry、実行枠、GitHub操作、タスクの合否判定は呼出元が管理する。

| ファイル | 責務 |
|---|---|
| `identity.py` | parent・board・Herdr表示・レビューが共有する環境由来のroot session識別。IDの引数指定や推測をせず、混在と子agentを拒否する |
| `codex_queue.py` | boardとcodex-scheduled-followupが共有するqueue投入と実行証拠の照合。timerや担当操作は持たない |
| `agents.conf` | 役割の既定値・実装子の代替モデル・用途別の固定プロファイル |
| `agent_settings.py` | 設定の検証・優先順位・実CLIの引数構築 |
| `agent_events.py` | 構造化イベントの検証と実効値・使用量の抽出 |
| `agent_runtime.py` | run/session対応、同一sessionの排他・継続・停止、ローカル記録 |
| `opsproc.py` | 外部コマンド、プロセス識別・グループ停止・離脱した子孫の回収 |

## Codex queueの共通部品

`codex_queue.py`は、呼出元が固定したthread/cwd/rollout/実行環境を使う。
`enqueue(task, marker, persist=..., command=...)`は保存済みの完全なrollout境界を前提とし、
uncertainを永続化してからqueueを1回だけ実行する。受理行を得ればqueued、
timeout・異常終了でも受理行があればその証拠を保持する。
`reconcile(task, persist)`はmarkerをuser入力で照合し、対応turnのstarted/completedを記録する。
ファイル置換・切詰め・境界付近の改変・不正なturn対応はblocked_reasonを残す。
保存方法・排他・宛先の発見・再送を許す条件は呼出元の責務。
timer・systemd操作やagent起動は含まない。

呼出元は[codex-scheduled-followup](../../skills/codex-scheduled-followup/references/cli-heartbeat.md)と
[board](../board/README.md#呼び鈴の実経路と受信hook)。
既存helperのhistory/acknowledge/reconcile公開関数とstate versionは保持する。

## モデルと設定の優先順位

`--role parent` は `--agent claude`、`--role implementation` は `--agent codex` と同じ。
両方の指定は拒否する。既定値の正本は [`agents.conf`](agents.conf)。
親はOpus / xhigh、実装子はSol / max。実装子は `--model gpt-6-astra` で代替モデルを明示選択でき、
effortはmaxを必須とする。親effortは設定または `--effort` で変更できる。

- プロファイルなし: 完全な役割既定値 → タスクの `--model` / `--effort` の順。
  overrideがあっても不足・不正な既定設定を許容しない。
- `--profile review` / `ops_report`: そのプロファイルの完全な固定値だけを使う。
  通常の既定値へ戻らず、`--model` / `--effort` / `--role` との併用を拒否する。
  reviewとops_reportの従来のmodel/effort/service tierは維持する。
  ops_reportだけnative multi_agentと設定されたthreads/depthを有効にする。
- configは隣の `agents.conf`。明示設定は `--config-file FILE`。内容はリテラルな
  `NAME=value` とコメントだけで、shellとして評価しない。環境変数の同名pinは読まない。
  CLI自身のmodel/effort/service tier設定より明示引数を優先し、実効値も別途照合する。

利用可能モデルへの事前アクセス確認は行わない。指定モデルを1回起動し、失敗・異なる実効値を
非ゼロ終了にする。他モデルへの振替や自動再試行はしない。

## 起動・継続

呼出元は既存worker/PR/担当と実行枠を先に照合し、各ターンに新しい `--run-dir` を割り当てる。
`--task-id` は同じ作業を表す安定した呼出元のキー。指定時はgit worktreeを必須とする。
`--branch` は任意の期待branchで、不一致なら起動前に拒否する。detached checkoutは
`(detached)` とそのHEADを記録し、再開時に照合する。パスは呼出元のcwdを基準に解決する。

```bash
runner=.agents/ops/lib/agent_exec.sh
"$runner" --role implementation --mode write --task-id ball-detector \
  --cwd "$worktree" --branch codex/ball-detector \
  --prompt-file "$task_dir/task.md" --run-dir "$task_dir/initial"

"$runner" resume --role implementation --mode write --task-id ball-detector \
  --cwd "$worktree" --branch codex/ball-detector \
  --resume-from "$task_dir/initial" --prompt-file "$task_dir/fix.md" \
  --run-dir "$task_dir/fix"
```

`--prompt-file -` はstdinのUTF-8本文を読む。promptは引数へ埋め込まない。
最終文は常にrun dirの `final.md` に保存し、`--out FILE` 指定時は正常終了後にそのパスにもコピーする。
同じoutを次ターンで使っても過去の最終文は残す。従来の `--agent --cwd --prompt-file --out`
呼出しも使え、`--run-dir` 省略時は `--log-dir/<run UUID>/` に保存する。
log dir既定は `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/logs/`。
task ID省略時は初回promptのSHA-256をキーにする。後続workflowは明示キーとrun dirを使う。

再開では初回と同じagent/profile/mode/task/worktree/branch/read_dirs/configを指定する。
前回の明示model/effort overrideは省略すれば保持し、異なる値の指定は拒否する。
configはパスとファイル全体のSHA-256も照合するため、コメントだけの変更でも再開を止める。
初回sessionのlockを全ターンで共有し、稼働中・古いrunからの再開を拒否する。
Codex transcriptに別経路の継続が見つかった場合も拒否する。
同一branch上の実装commitによるHEADの前進は許容する。

Codexは `exec resume <session-id> -`、Claudeは `-p --resume <session-id>` を使う。
確認したCLIはCodex 0.163.0（`exec --help` / `exec resume --help`）とClaude Code 2.1.307（`--help`）。
Codexは初回・再開とも `--disable daemon_auto_start` を渡す。
WSLでCODEX_HOMEがdrvfs上にあるとdaemonのUnix socketが作れないため、この指定を外さない。
Codexのread-only再開は `exec --sandbox read-only ... resume` の親引数でsandboxを指定する。
`--last`、fork、新sessionへの自動振替は使わない。
JSONL/再開の公式説明は [Non-interactive mode](https://developers.openai.com/codex/noninteractive)。

### checkout外の入力ディレクトリ

`--read-dir DIR` は複数回指定できる。各値は呼出元のcwd基準で絶対実体パスへ解決し、
存在するディレクトリでなければ終了値2でCLI起動・run dir作成前に拒否する。
同じ実体への重複指定（symlinkを含む）は1件にまとめ、パス順に並べる。
`--cwd` の変更は相対read-dirの解決基準を変えない。

```bash
"$runner" --role parent --mode read-only --task-id assignment-judgement \
  --cwd "$checkout" --read-dir "$input_dir" --read-dir "$context_dir" \
  --prompt-file "$task_dir/prompt.md" --run-dir "$task_dir/agent-runtime"
```

Claudeはread-only/writeの両方で各ディレクトリを `--add-dir DIR` として渡す。
Claude Code 2.1.308の `claude --help` でツールアクセス用の追加ディレクトリであることを確認した。
公式の [CLI reference](https://code.claude.com/docs/en/cli-reference#cli-flags) はファイルアクセスの付与、
[Working directories](https://code.claude.com/docs/en/permissions#working-directories) は追加先も
作業ディレクトリと同じ規則でプロンプトなしに読めることを説明している。
[Permission modes](https://code.claude.com/docs/en/permissions#permission-modes) にも、
`dontAsk` では作業ディレクトリ内の読み取りを実行できると明記されている。
この組合せで、既存の `--tools Read Grep Glob --permission-mode dontAsk` のまま追加先を読む。
明示的なdeny/ask規則やhooks等による拒否はCLIの規則に従う。writeでの編集許可も既存modeに従う。

Codexはread-only/writeとも追加引数を付けない。
read-only sandboxはOSの権限で読めるファイルシステム全体を読み取れる前提を使う。
公式 [Sandbox read access](https://learn.chatgpt.com/docs/app-server#sandbox-read-access-readonlyaccess)
では `readOnly.access` の既定は `fullAccess`、限定する場合は `restricted` と読取rootを指定する契約である。
read-dirによってCodexのwriteに追加の書込権限は与えない。
read-dir省略時は空リストで、review / ops_reportを含む既存のCLI引数は変わらない。

`invocation.json` と `result.json` の `contract.read_dirs` に同じ正規化済みリストを保存する。
resumeでも同じ集合を明示指定する（順序・重複は問わない）。追加・削除・symlink先の変更は
cwd/branch等と同様に契約不一致として終了値2で拒否する。read_dirsのない旧記録も不一致になる。

子へ渡す環境から `CLAUDE_CODE_SESSION_ID`、`CLAUDE_CODE_PARENT_SESSION_ID`、
`CLAUDE_CODE_AGENT_ID`、`CLAUDECODE`、`CODEX_SESSION_ID`、`CODEX_THREAD_ID` を除く。
子の担当identityは子CLI自身が設定する値を使い、親の値やPIDを代用しない。
`AGENT_RUNTIME_RUN_DIR` と `AGENT_RUNTIME_TASK_ID` は子のローカル記録への参照であり、
担当identityや起動権限ではない。実CLIが環境を生成すること自体の検証は実機検証で行う。

## 記録と結果JSON（schema_version 1）

run dirは新規0700、ファイルは0600（umask 077）。JSONはfsync・renameで保存する。
既存runを上書きせず、不正JSON・重複key・未知versionを空状態へ置換しない。

| 保存先 | 内容 |
|---|---|
| `prompt.md` | そのターンの入力原文。`prompt_sha256` と対応 |
| `invocation.json` | contract、実引数と除去した環境変数の名前。環境の値は保存しない |
| `events.jsonl` / `stderr.log` | CLIの構造化stdout原文 / stderr原文。失敗・途中切れも残す |
| `final.md` | 最終文。CodexではCLIのlast-messageファイルとイベント本文を照合。正常終了時は `--out` にもコピー |
| `result.json` | run ID、contract、runtime情報、process情報、終了理由 |
| 初回runの `session.json` / `session.lock` | runtime session、最後に観測したturn、最新runの対応と排他 |

`result.json` の主要フィールド:

- `result`: `running` / `completed` / `failed`。`completed` はCLIターンの正常終了だけ。
  `task_success` は常に `null`。親が差分・受入条件・テスト・PRなどを照合する。
- `contract`: `task_id`, `cwd`, `read_dirs`, `branch`, `detached_head`, `mode`, `runtime_home`, `settings`。
  settingsはconfig出典/hash、role/agent/profile、`defaults`、`overrides`、`selected`、
  `selection_sources`、ops_reportのthreadsを区別する。
- `runtime`: 実session/turn ID、再開要求のsession ID（観測値とは別）、`effective`、`usage`、`runtime_errors`。
  effective/usageは `status=observed` と `value` / `source`、または
  `status=unavailable` と `value=null` / `reason` を持つ。欠測を既定値や0へ変換しない。
- Codexのmodel/effort/turn/worktreeは、返されたsession IDと一致するCODEX_HOMEの
  `sessions/` または `archived_sessions/` の新しい `turn_context` から確認する。
  transcript欠落・破損・曖昧さ・不一致は `metadata_error`。CLIのJSONLだけから推測しない。
  service tierは現行transcriptにないため、実効値未取得の理由を残す。
- Claudeのmodel/sessionは `system.init` と `result` を照合する。
  effective effortとturn IDはstream-jsonが提供しないため理由付き未取得。
  usage/modelUsage/costは取得できたresultの値を保存し、請求確定額へ読み替えない。
- `runner` / `process`: PID、boot ID、start ticks。生存確認はPIDだけで判定しない。
  `cli_exit_code` はCLIの実終了、`exit_code` はwrapperの終了、`terminated_pids` は回収対象。
  `cleanup_complete` が未確認、または前のCLI/groupが残る場合は再開を拒否する。
  session IDはイベント受信時に保存する。強制終了直前の未保存情報は推定しない。
- `transcript_before`: Codex継続の起動前transcriptのパス・byte数・SHA-256・turn ID。
  初回とClaudeでは `null`。runner喪失時に、他経路のturnを中断runとして取り込まないために使う。
- `runner_lost` / `cleanup_confirmation` / `recovery_error`: runner喪失からの復旧時に追加する。
  外側supervisorによる全子孫の回収確認と、session/turnの証拠復旧を別々に記録する。
  証拠の復旧エラーがあるrunは、回収確認済みでもresumeを拒否する。

stdoutは `schema_version`, `run_id`, `result`, `turn_end_reason`, `exit_code`, `task_success`
だけのJSON。status/stopは生存確認のbooleanを追加する。session ID・絶対パス・本文を含めない。
ローカル記録には秘密情報・生session ID・ローカルパスが入り得るため、そのまま公開しない。
`--out` の本文を公開する既存呼出元も、自身の出力schema/公開検証を続ける。

| 終了理由 | wrapper終了値 | 扱い |
|---|---|---|
| `completed` | 0 | 正常なsession/完了イベント、非空最終文、必須実効情報を確認 |
| `invalid_output` / `metadata_error` | 1 | 空・破損・未完了イベント、最終文不一致、実効値取得失敗 |
| `runtime_error` / `usage_limit` | 非0（CLIが0なら1） | runtime失敗。枠不足の識別は構造化errorの文言による補助分類 |
| `cli_error` / `startup_failed` | CLI終了値 / 127 | 起動・認証・モデル利用不能など。詳細はローカルstderr |
| `timeout` | 124 | 呼出元が明示した有限正数の `--timeout` 到達。既定は上限なし |
| `stopped` | 130 / 143 | SIGINT / SIGTERM、またはstopで中断を照合 |
| `descendants_reaped` | 1 | CLI終了後も残ったtoolsを回収。タスク完了にはしない |
| `cleanup_failed` | 1 | プロセス回収未確認。supervisorが照合するまで再開しない |

設定・入力・再開契約の事前拒否は終了値2でCLIを起動しない。run dir作成前の拒否は
result.jsonを生成しないので、呼出元は終了値とJSONの有無を両方確認する。

## 状態確認・停止・中断からの復旧

```bash
.agents/ops/lib/agent_exec.sh status --run-dir "$task_dir/initial"
.agents/ops/lib/agent_exec.sh stop --run-dir "$task_dir/initial"
```

statusは保存結果とrunner/CLI/CLI process groupの生存を返す。読み取りだけで状態を変えない。
通常のstopは同じboot/start ticksのrunnerへTERMを送り、終了記録を待つ。
runnerはCLI groupをTERM→KILLし、subreaperで引き受けた別sessionの子孫も回収する。
stdoutを先に閉じたCLIも、停止・timeoutを監視し続ける。

runnerがSIGKILL等で失われた場合、`running` の残存は成功でも稼働の証拠でもない。
stopはsession lock下で、生存するCLI leaderのidentityが一致すればそのgroupを回収する。
CLI/groupが消滅済みでも `cleanup_complete=false`、`cleanup_failed` として記録し、resumeを拒否する。
stop自身の終了値0は照会・記録の成功であり、回収の確認は保存結果で判断する。
PID再利用・leader不在で残るgroupなど、
identityを確認できない残存プロセスは強制停止せず、supervisorの照合を要求する。
wrapper自身のSIGKILL後に既に離脱していた子孫の回収は、外側supervisorの責務。
supervisorは自身の隔離scope等で**全子孫**の回収を照合し、その後で次の確認JSONを作る。
CLI/groupや既知PIDの消滅だけを根拠にしない。

- `schema_version`: 整数 `1`。
- `run_id` / `runner` / `process`: 対象 `result.json` の同名フィールドの値をそのまま照合に使う。
- `cleanup_complete`: boolean `true`。
- `evidence`: supervisorの回収scopeと照合根拠を記した非空文字列。

この6キーだけを持つJSONを、次の追加オプションで渡す。確認内容はローカルの `result.json` に保存する。
確認JSONはsupervisorの明示的な申告であり、runnerが失った子孫一覧を復元するものではない。
対象run・process identityの不一致、既知CLI/groupの生存、最新run以外への確認は拒否する。

```bash
.agents/ops/lib/agent_exec.sh stop --run-dir "$task_dir/interrupted" \
  --cleanup-confirmation "$task_dir/supervisor-cleanup.json"
```

stopは保存済みイベントを再読し、Codexでは起動前transcriptのprefix・追記された唯一のturn・
保存promptのhash/本文・session・worktree・実効設定を照合して、`runtime` と `session.json` を復旧する。
外部turnの混入・改変・破損・継続runのcheckpoint欠落では `recovery_error` を残し、
全子孫の回収を確認しても `metadata_error` としてresumeを拒否する。古い記録から対応を推測しない。
stopの再実行は途中の保存から復旧でき、以前のrunから最新sessionを上書きしない。
同じsessionのローカル記録・実プロセス・差分を確認し、回収と証拠の両方を確定してからresumeする。
session IDが得られなかったrunは再開できず、別sessionを黙って作成しない。

既存 `opsproc.run_process_group` からも呼べる。runnerはTERM/INTを処理して自身のCLI groupを
回収する。外側のgraceは内側の回収待ち（通常最大約4秒）より長く保つ。既定は5秒。
review / ops-reportは `agent-runtime/` を指定して従来のprofileと `--out` を使い、
それぞれの既存の受理・レビュー・公開処理がタスク合否を判断する。
