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
| `agent_output.py` | 出力JSON Schemaの固定と構造化結果の検証 |
| `agy_headless.py` | issue書き直し用agyの読み取り専用1回実行。固定版・logのモデル・構造化結果を照合 |
| `agy_permissions.py` | コマンド・引数・フラグの定義から、両端を固定したagyの権限規則を生成 |
| `agent_runtime.py` | run/session対応、同一sessionの排他・継続・停止、ローカル記録 |
| `opsproc.py` | 外部コマンド、プロセス識別・グループ停止・離脱した子孫の回収 |

固定commitとGitHub GETの取得・隔離CPUテストは [review_tools](review_tools/README.md) を正本とする。

## agyの共通呼び出し部品

`agy_headless.py` は `agent_exec.sh` とは独立した、issue書き直しの writer/verifier 用の同期Python API。
導入・ログイン・更新・手動疎通は [runner README](../../../scripts/github_actions/README.md#agyの導入と疎通) にある。
`sys.path` にこの `lib/` を加え、次の公開APIを使う。`jsonschema` が必要で、schemaの方言と
外部参照の拒否は既存の `OutputSchema` に従う。

```python
from pathlib import Path
from agy_headless import DEFAULT_AGY, AgyError, AgyResult, run_headless

try:
    result: AgyResult = run_headless(
        agy=DEFAULT_AGY,
        cwd=Path.cwd(),
        prompt_file=Path("writer-prompt.md"),
        json_schema=Path("writer.schema.json"),
        log_file=Path("private-run/writer.log"),
        timeout=300,
    )
except AgyError as error:
    reason = error.reason
    denied_actions = error.denied_actions
else:
    rewritten = result.structured_output
```

`run_headless(*, agy: Path, cwd: Path, prompt_file: Path, json_schema: Path,
log_file: Path, timeout: float, model: str | None = None,
config_file: Path = DEFAULT_CONFIG) -> AgyResult` が公開する呼び出し契約。
相対パスは呼出元のcwdを基準に解決する。prompt/schemaはUTF-8ファイル。
schemaの原文をprivateな一時ファイルへ固定し、同じ内容でCLIとローカル検証を行う。
`timeout` は有限の正数で、実行前後の `--version` とheadless呼び出し全体の上限秒数。
timeoutでは既存の `opsproc.run_process_group` がCLIと同じprocess groupのtoolを回収する。

- `load_config(config_file: Path = DEFAULT_CONFIG) -> AgyConfig` は `agents.conf` の4キーだけを設定元とする。
  `AgyConfig` は `version: str`, `url: str`, `sha256: str`, `model: str` を持つfrozen dataclass。
  `DEFAULT_CONFIG` はこのモジュールの隣の設定ファイル。環境変数の同名キーは読まない。
  4キーは `AGY_VERSION`, `AGY_URL`, `AGY_SHA256`, `ISSUE_REWRITE_AGY_MODEL`。
  導入テスト用の明示configでは `file://` のtarballも使える。通常は公式GitHub Releaseの版付きasset URLと
  release APIが返したSHA-256 digestを固定する。
- `model` の省略時は `ISSUE_REWRITE_AGY_MODEL` を使い、明示しても固定値と異なれば起動前に拒否する。
  `AGY_CLI_DISABLE_AUTO_UPDATE=true` を子環境へ設定し、実行前後の版が `AGY_VERSION` と一致することを確かめる。
  logの選択モデル表示は、既知のslugに対応するlabelと全件一致しなければ拒否する。
  新しいモデルを採用するときは設定と `MODEL_LABELS` の対応を一緒に更新する。
- `readonly_settings() -> dict[str, Any]` が生成する固定modeと `permissions` をJSON文字列として `--settings` へ渡す。
  コマンドの許可形式・拒否・実機での検証は[読み取り権限と実機プローブ](#agyの読み取り権限と実機プローブ)を正本とする。
  `--mode plan` を指定する。このpermission規則はrunnerのOS隔離を置き換えない。
- `log_file` は呼び出しごとの未作成パスが必須。親ディレクトリがなければ0700で作成し、logは0600にする。
  存在するファイルやsymlinkを再利用しない。writer、verifier、修正呼び出しごとに別パスを渡す。
  毎回新しい会話を作り、自動再試行・resume・別モデルへの振替は行わない。

`AgyResult` はfrozen dataclassで、フィールドは次のとおり。

| フィールド | 型・意味 |
|---|---|
| `structured_output` | `Any`。指定schemaを通った非nullのJSON値。自由文の `response` から補完しない |
| `usage` | `dict[str, int] \| None`。CLIが返した非負整数の使用量。欠測は `None` で、0を補わない |
| `denied_actions` | `list[dict[str, str]]`。CLIが返した拒否一覧。フィールド省略時だけ空リスト |
| `observed_model` | `str`。今回のlogで照合したlabel（現行は `Gemini 3.8 Flash (High)`） |
| `observed_version` | `str`。実行前後で固定値との一致を確認した版 |
| `duration_seconds` | `float \| None`。CLIの所要秒数。欠測は `None` |
| `elapsed_seconds` | `float`。版の確認を含め、部品が計測した実行秒数 |

`AgyError(RuntimeError)` は `reason: str` と `denied_actions: list[dict[str, str]]` を持つ。
`reason` は `invalid_input`（設定・入力・既存log）、`version_mismatch`（版欠落・相違）、
`cli_error`（起動失敗・非0終了）、`timeout`、`invalid_output`（不正JSON・status・構造化出力・schema・観測値）、
`model_mismatch`（今回のlogのlabel欠落・相違）、`unsafe_user_settings`（永続設定の事前検査で拒否）。
呼出元はこの型をcatchし、成功結果を捏造しない。既存のreasonと関数の引数・結果の型は維持する。
例外の文字列にはstderrやpromptが含まれ得るため、そのまま公開しない。
拒否一覧はparseできた出力から得たものだけを保持する。非0終了やparse不能時は空の場合がある。
有効な構造化出力もある拒否は `AgyResult.denied_actions` に残し、タスクとしての合否は呼出元が判断する。
exit 0・`status: SUCCESS` でも、構造化出力欠落やschema違反なら例外になる。
[1.4.11の公式release notes](https://github.com/google-antigravity/antigravity-cli/releases/tag/1.4.11)に従い、
途中終了を示す `finished_reason` がある出力も拒否する。

この部品は本文の不変条件・意味の同等性・秘密情報の検査・GitHubへの書き込みを行わない。
writerとverifierの結果を受け取った呼出元が、これらの条件を判定してから公開する。

### agyの読み取り権限と実機プローブ

`readonly_settings()` は[公式のCLI設定](https://antigravity.google/docs/settings?tab=cli)に従い、
次の値を明示する。`--mode plan` だけを権限の固定とは扱わない。

| 設定 | 固定値 | 理由 |
|---|---|---|
| `toolPermission` | `request-review` | allowにない操作は確認を要求し、headlessでは拒否になる。永続設定の自動承認を引き継がない |
| `agentMode` | `plan` | 同時に渡す `--mode plan` と一致させ、auto/accept-editsを有効にしない |
| `allowNonWorkspaceAccess` | `false` | workspace外の読み取り・編集を減らす。コマンド引数への適用は実機で別途観測する |
| `enableTerminalSandbox` | `false` | runnerでの利用可否を確認していないため、sandboxを有効化せず、利用可能性を前提にしない |

`--settings` とuserの `settings.json` の置換・併合・優先関係は公式文書に無い。
そのため、`run_headless` は `--version` を含む各agyプロセスの起動前に、実行userの
`$HOME/.gemini/antigravity-cli/settings.json` を読み、次の場合は `unsafe_user_settings` で止める。
ファイルが無い場合は許可する。設定ファイルを自動修正・削除したり、認証情報をコピーしたりはしない。

- `toolPermission` が `request-review` / `strict` 以外。未知の値・不正な型も拒否する。
- `agentMode` が `auto` / `accept-edits`、または未知の値・型。
  省略・`""`・`default`・`plan` だけを許す（公式CLI referenceで `""` はdefault mode）。
  拒否メッセージは、userの `agentMode` を削除するか `plan` に直すよう案内する。
- `permissions.allow` が空リスト以外。`permissions` はobject、子キーは `allow` / `ask` / `deny` だけとし、
  `ask` / `deny` は非空文字列のリストだけを許す。省略はいずれも空リストとして検査する。
- `enableTerminalSandbox` / `allowNonWorkspaceAccess` が明示され、その値がboolean `false` と異なる。
  省略は公式の既定値 `false` として許す。数値0や文字列はbooleanとして扱わない。
- 読めない、通常のファイルではない、UTF-8/JSONとして不正、rootがobjectでない、重複キーがある。
  dangling symlinkや、不明・相対・空の `HOME` も拒否する。
- `model` が文字列以外、または `trustedWorkspaces` が文字列の配列以外。
  ログイン後にCLIが保存するこの2キーは許す。`model` は表示名を含む任意の文字列でよい。
  各呼び出しで `--model` を明示し、今回のlogの表示名でも固定モデルを照合するため、保存値にモデル選択を任せない。
- 実行する `cwd` が `trustedWorkspaces` のいずれかと同じ、またはその内側にある。
  文字列の前方一致は使わず、パスの要素で同一・祖先を判定する。`~` を展開し、相対entryは実行する `cwd` が基準。
  `..` を正規化した絶対パスとsymlinkを解決したパスの両方で検査する。パスを解決できない場合も拒否する。
- 上記と `permissions` 以外の未知のtop-levelキー。未検証のキーが将来権限を広げないよう拒否し、
  表示設定なども勝手に無視しない。例外は公式にCLIが無視すると明記された文字列の `$schema` だけで、URLは取得しない。

これは、併合されても権限を広げない永続設定だけを受け入れる検査である。
userの `strict` やask/denyが優先されて読み取りまで拒否される可能性は残り、実機smokeで確認する。

`trustedWorkspaces` を許すのは、#1607が呼び出しごとに新しい一時ディレクトリへ使い捨てworktreeを作り、
この検査がその領域と祖先のtrustを拒否するためである。trustの追加や設定ファイルの書き換えは行わない。
[公式の設定ファイル一覧](https://antigravity.google/docs/settings?tab=cli#other-configuration-files)では、workspaceの拡張設定は
`.agents/hooks.json`、`.agents/mcp_config.json`、`.agents/plugins/` に置く。
このrepoの `.agents/` はskills・memory・ops・READMEなどを持つが、これらの拡張設定は持たない。
**untrustedでこれらを読み込まないこと、headlessがtrust確認待ちで止まらないことは未確認**であり、
設定検査だけで保証しない。#1599で[手動疎通の確認事項](../../../scripts/github_actions/README.md#手動の疎通)を確かめる。

コマンド名だけの前方一致は使わない。`agy_permissions.py` の形式定義を `_form` / `_option` /
`_argument` で正規表現へ変換し、allowをすべて `command(regex:^...$)` にする。
[公式のCLI権限仕様](https://antigravity.google/docs/permissions#supported-cli-actions-and-matching-rules)は
正規表現の行全体照合と部分照合の詳細を確定していないため、どちらでも後続の引数を無制限に許さないよう両端を固定する。
未知のフラグを通常の引数として受け入れず、引数先頭の `-` は引用されていても拒否する。

すべての引数で、制御文字（改行・CR・tab等）、`;`・`&`・`|`・`<`・`>`・`$`・バッククォート・
`(`・`)`・`{`・`}`・バックスラッシュを禁止する。単一/二重引用符は対になった引数全体だけで使え、
引用中にも同じ禁止文字の制約を適用する。スペースとglobの `* ? [ ]` は引用中に限る。
引用しない先頭 `~` とglob、引用の連結、環境変数の代入や `env` 等のラッパーも許可しない。
禁止文字は行全体のcommand denyでも先に判定する。これはPythonの `re.search` で `$` が末尾LF直前にも
一致し得る場合も拒否するためで、テストは `re.fullmatch` / `re.search` の両方でDeny優先の最終判定を確認する。
lookaroundやPython固有の絶対終端anchorには依存しない。

許可するプログラムとフラグは次のとおり。表にない略記・結合フラグ・オプションは拒否する。
`N` は非負整数で、短い数値フラグは `-n10` の形も使える。長い値付きフラグは空白区切りまたは `=` を使う。
`--` は表の該当プログラムで使えるが、その後も引数の禁止文字・先頭 `-` の検査を省略しない。

| 形式 | 許可するフラグ・引数 |
|---|---|
| `cat FILE...` | `-n`, `--` |
| `ls [PATH...]` | `-a`, `-l`, `-d`, `-h`, `-1`, `--` |
| `head FILE...` / `tail FILE...` | `-n N`, `--`。`tail -f` は拒否 |
| `wc FILE...` | `-l`, `-w`, `-m`, `--` |
| `grep PATTERN FILE...` | `-n`, `-i`, `-l`, `-F`, `-w` |
| `rg PATTERN [PATH...]` | `-n`, `-i`, `-l`, `-F`, `-w`, `-g ARG`, `--glob ARG`, `--hidden`, `-C/-A/-B N`, `--` |
| `rg --files [PATH...]` | `-g ARG`, `--glob ARG`, `--hidden` |
| `git log [REV/PATH...]` | `--oneline`, `-n N`, `--follow`, `--` |
| `git log --no-ext-diff --no-textconv [REV/PATH...]` | 上のlogフラグに加え `-p`, `--stat` |
| `git show --no-ext-diff --no-textconv [REV/PATH...]` | `-p`, `--stat`, `--name-only`, `--name-status`, `--` |
| `git diff --no-ext-diff --no-textconv [REV/PATH...]` | 上のshowフラグに加え `--cached` |
| `git grep PATTERN [PATH...]` | `-n`, `-i`, `-l`, `-F`, `-w`, `--no-textconv`, `--` |
| `gh issue view ID` / `gh pr view ID` | `--comments`, `--json FIELD,FIELD`, `--repo OWNER/REPO`, `-R OWNER/REPO` |
| `gh issue list` | `--json FIELD,FIELD`, `--repo OWNER/REPO`, `-R OWNER/REPO`, `--limit N`, `--state open/closed/all` |

Gitの各形式では `git --no-pager ...` も使える。patchを生成する形式は、通常のユーザー設定からの
[外部diff・textconv実行](https://git-scm.com/docs/git-diff#Documentation/git-diff.txt---no-ext-diff)も避けるため、
`--no-ext-diff --no-textconv` をこの順でサブコマンドの直後に必須とする。
子環境では `GIT_OPTIONAL_LOCKS=0`、`GIT_PAGER=cat`、`GH_PAGER=cat`、`RIPGREP_CONFIG_PATH=/dev/null` を固定し、
Gitの任意index更新、pagerの実行、rgの設定経由のフラグ追加を避ける。

書き込み/外部操作のdenyは `write_file(*)`, `read_url(*)`, `execute_url(*)`, `unsandboxed(*)`, `mcp(*)`。
`edit_file` は公式CLIのactionではないため置かない。編集も `write_file` の拒否で扱う。
workspace内の `read_file` はCLIの暗黙の読み取り許可を維持し、全ファイルへの読み取り許可は追加しない。
`--output`, `-o`, `--pre`, `--pre-glob`, `-O`, `--open-files-in-pager`, `--ext-diff`, `--textconv`,
`-c`, `--exec-path`, `--git-dir`, `--work-tree`, `--web`, `--jq`, `--template` は許可形式に存在しない。

静的テストだけで実CLIの照合を確認したとは扱わない。#1599で人間がデスクトップから **AGY smoke** を手動実行し、
通常のテンプレート読み取りに続いて `agy_smoke.py --permissions-probe --timeout 120` の段階を通す。
この段階は使い捨てgit repositoryを作り、`read_file` と9個の読み取りコマンドを依頼し、
privateな入力nonceとHEADのcommit subjectに一致する出力、拒否なし、ファイル不変を確認する。
その後、リダイレクト・連結・実行フラグ・展開・改行と、許可外の単独コマンドを含む27形式を、
新規会話で1形式ずつそのまま試すよう依頼する。単独コマンドは `touch b`, `rm x`, `mkdir new-directory`,
`cp a copied`, `mv a moved`, `python3 -c pass`, `sh -c true`。特殊文字のdenyに頼らず拒否されることを確認する。
各形式の `denied_actions` にcommandの拒否が必要。exit 0でも拒否がなければ失敗する。
構造化出力を伴わない拒否も、版・今回のモデルlogを照合した場合だけ期待した拒否として受け取る。

さらに、外側のcanaryファイルを絶対パスの `cat` で読む観測を1件行い、`outside_read_denied` をsummaryに記録する。
commandまたはread_fileの拒否があれば `true`、正常な構造化結果が返り拒否が無ければ `false`。
**この読み取りが通ったことだけではsmokeを失敗にしない。** コマンド引数経由の領域外読み取りは
#1597の既知の改善候補であり、この変更で解消したとは仮定しない。実機での拒否が確認できたときだけ改善の証拠にする。
観測でも、runtime/モデル/版の失敗やファイル変更は引き続き失敗とする。外側のパス・内容・nonceは公開しない。

呼び出しごとにwork全体（`.git`を含む）と、その外側の専用canary directoryの
ファイル一覧・種類・mode・inode・size・mtime・内容のdigestを比較し、作成・削除・変更があれば失敗する。
外側の実行canaryはrg/git/browser経由の実行を記録する。`BROWSER` と `GH_BROWSER` はこのcanaryへ向け、
誤って許可されても本物のbrowserは開かない。検査範囲はこの使い捨てworkと外側のcanary directoryであり、
agy自身の認証/cache/logが保存されるHOME全体の変更を主張するものではない。
制御用prompt/schema/logは別directoryに置き、summaryには公開しない。
summaryに残すのは、照合した版・モデル、数値usage・時間、固定probe識別子、拒否件数、検査結果だけ。
最大29会話（読み取り1・拒否27・領域外読み取り観測1）、各120秒、workflow全体75分。
必須の拒否・読み取り・不変検査が1件でも失敗すればjobも失敗し、有効化の根拠にしない。

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

### JSON Schemaによる構造化出力

`--mcp-config FILE` はread-onlyのCodex review/ops_reportだけで使うホスト提供MCP接続。
JSONはserver名をキーにcommand/args/env/tool_timeout_sec/startup_timeout_secを指定する。
原文をrun dirへ保存し、SHA-256をcontractへ固定する。resume時の変更は拒否する。
この指定でsandboxや追加agentの設定を変更しない。

`--json-schema FILE` を指定すると、そのUTF-8 JSON Schemaを起動前に検査し、原文をrun dirの
`output-schema.json`へ固定する。検証にはproject依存の`jsonschema`を使う（導入時は`uv sync --locked`）。
未指定のreview / ops_reportを含む既存呼出しの引数・テキスト出力・profile設定は変わらない。

Claude/Codex共通の方言はdraft-07とし、`$schema`省略時もローカル検証には`Draft7Validator`を使う。
Claude Code 2.1.308はAjv既定のmeta-schemaだけを受理するため、draft-07以外の宣言を起動前に終了値2で拒否する。
2020-12の宣言を削除するだけで新しいkeywordの意味を保てるとは限らない。
[Ajvの方言](https://ajv.js.org/json-schema.html#json-schema-versions)と
[strictTypes](https://ajv.js.org/strict-mode.html#require-applicable-types)に合わせ、配列制約には`type: array`を明示する。
[draft-07の参照](https://json-schema.org/understanding-json-schema/structuring#ref)では`$ref`の兄弟制約を無視するため、
参照に制約を加えるときは`allOf`を使う。Codex向けに別の方言を明示した場合は、宣言に対応するvalidatorで検証する。

- Claude Code 2.1.308の`--help`で`--json-schema <schema>`を確認した。ファイル名ではなく
  固定schemaのJSON文字列を渡す。stream-jsonの最終`result`イベントにある`structured_output`だけを
  検証して`final.md`へJSONとして保存する。`result`の自由文やtool_useから補完しない。
  公式の[CLI出力](https://code.claude.com/docs/en/headless#get-structured-output)と
  [構造化結果の取得](https://code.claude.com/docs/en/agent-sdk/structured-outputs#recognize-the-structuredoutput-tool-call)
  がこの取得先を説明している。
- Codex 0.163.0の`exec --help` / `exec resume --help`で`--output-schema <FILE>`を確認した。
  初回・再開とも固定ファイルを渡し、既存のlast-messageと完了イベントの本文照合後に、JSONを厳格にparseして検証する。
  [公式のschemaと最終ファイルの使用例](https://learn.chatgpt.com/docs/non-interactive-mode#create-structured-outputs-with-a-schema)に従う。

構造化結果の欠落、JSON不正、schema不一致、Claudeの`error_max_structured_output_retries`は
`invalid_output`で失敗し、`--out`へコピーしない。前置きやfenceを読み飛ばさない。
schema内の参照は固定ファイル内だけで解決し、外部ファイル・URLのschemaを取得しない。
出力との照合エラーと生イベントはprivateなrun記録に残す。
事前検査を通ってもCLIがschema等を拒否した場合は`cli_error`で失敗し、`result.json`の`error`へ
stderr先頭1024 bytes（UTF-8の不正byteは置換）を保存する。stdoutの短い要約へは本文を出さない。

指定時だけ`contract.json_schema_sha256`にschema原文のSHA-256を記録する。
resumeでも同じ内容のschemaを明示指定する。別パスの同一バイト列は許容するが、
内容・空白の変更、指定の追加・削除はCLI起動前に終了値2で拒否する。

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

Codex CLI 0.163.0 の `web_search` は、`ThreadItem.id` と flatten された
`WebSearchItem.id` を同じitem objectへ出力する。CLIイベントの
`item.started` / `item.updated` / `item.completed` に限り、先頭の `item_数字` を
`id`、2つ目の非空文字列を内部の `web_search_id` として区別する。
この例外はitem直下のidがちょうど2件の場合だけで、他の重複キー・不正型は拒否する。
`events.jsonl` は正規化せず受信した原文を保持し、中断後の復旧も同じ境界処理を使う。
Claudeイベント・保存JSON・Codex transcript・最終構造化出力の重複キー検査は緩めない。

| 保存先 | 内容 |
|---|---|
| `prompt.md` | そのターンの入力原文。`prompt_sha256` と対応 |
| `output-schema.json` | `--json-schema`指定時のschema原文。`contract.json_schema_sha256`と対応 |
| `invocation.json` | contract、実引数と除去した環境変数の名前。環境の値は保存しない |
| `events.jsonl` / `stderr.log` | CLIの構造化stdout原文 / stderr原文。失敗・途中切れも残す |
| `event-receipts.jsonl` | 非空イベント行ごとのUTC受信時刻。schema_version=1、event_start/event_endは原文のbyte範囲（改行を含む）、sha256は改行を除いた行のdigest、received_atはUTC ISO8601。原文は変更しない。旧runには存在しない |
| `final.md` | 最終文（schema指定時は検証済みJSON）。CodexではCLIのlast-messageファイルとイベント本文を照合。正常終了時は `--out` にもコピー |
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
