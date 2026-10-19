# DeepSeek Harness

`@deepseek-ai/dsh@0.2.0-rc.2` の導入と実行環境の確認を扱う。
[公式リポジトリ](https://github.com/deepseek-ai/deepseek-harness) は developer preview
として互換性の破壊を予告しているため、`latest` やglobalの `dsh` は使わない。
固定runtime、headlessのモデル・安全設定、read-only疎通、
任意タスクの実行・継続・停止と結果記録を行う共通wrapperを扱う。

## 導入

Node.js **`^22.19.0 || >=24.0.0`** とnpmが必要。22.18以前、23系、prereleaseは
対応しない。runtimeは対応範囲を検査して停止し、Nodeのインストールや更新は行わない。

repo rootで実行する。

```bash
node --version
npm --version
node tools/dsh/runtime.mjs install
node tools/dsh/runtime.mjs check
```

`install` は自身の `tools/dsh/` で
`npm ci --no-audit --no-fund` を実行する。
`package.json` はdshの版、`package-lock.json` は間接依存と配布物のintegrityを固定する。
固定された依存のlifecycle scriptも実行し、native依存の準備を行う。
Node 24.14.1 / npm 11.11.0で固定版の `-V` とheadless helpを確認した。
`node_modules/` は生成物であり、worktreeごとにこの手順で再作成する。

`install` / `check` はローカルpackageのmetadataと実CLIの `-V` を照合し、成功時に
次のJSONを1行出力する（`install` ではnpmの通常出力が先に出る）。パスは実行環境で決まる。

```json
{"node_executable":"/path/to/node","node_version":"24.14.1","node_requirement":"^22.19.0 || >=24.0.0","dsh_executable":"/path/to/tools/dsh/node_modules/@deepseek-ai/dsh/lib/bin.js","dsh_version":"0.2.0-rc.2"}
```

`check` の版検査は呼出元の `DEEPSEEK_API_KEY` を除外し、空の一時ディレクトリを
cwdと `DSH_HOME` に設定して、`DSH_TELEMETRY_DISABLED=1` で実行する。一時領域は
検査後に削除する。CLIが15秒以内に応答しない場合も停止する。
APIキーの発行・探索・保存、有料API呼び出しはこの手順に含まない。

## 後続から使う契約

公開入口は **`node <runtimeを所有するworktree>/tools/dsh/runtime.mjs`**。
`runtime.mjs` と同じディレクトリの `node_modules/@deepseek-ai/dsh/lib/bin.js` だけを
選ぶ。選択は呼出元のcwd、`PATH` 上の `dsh`、親ディレクトリの `node_modules` に依存しない。
`node_modules/.bin/dsh`、`npx`、globalインストールへの切替も行わない。

| コマンド | 動作 |
|---|---|
| `install` | Nodeを検査し、lockから導入、実CLIの版を検査 |
| `check` | Node・導入物・実CLIの版を検査し、選択した実体をJSONで返す |
| `exec -- <dsh引数...>` | 同じ検査後、同じNodeで選択したdshを起動 |

`exec` は呼出元のcwd・環境・stdin/stdout/stderrを保持し、dshの終了コードと
SIGINT/SIGTERM/SIGHUPを伝える。検査の成功メッセージをstdoutへ追加しないので、
後続wrapperはdshのNDJSONをそのまま取得できる。引数の区切り `--` とdsh引数は必須。
実タスク用の安全設定・APIキー・ログ設定は下記の共通wrapperが指定する。

Node未対応、未導入、metadataまたは実CLIの版不一致、CLI起動失敗は終了コード1。
引数の誤りは2で、導入・実行を開始しない。実行時に依存を自動取得しない。

別の専用worktreeから既存の導入物を使う場合は、入口の絶対パスを明示する。
たとえば `runtime_owner` を導入済みworktreeに設定してから、作業worktreeで以下を実行する。
`exec` のcwdは作業worktreeのままになる。

```bash
node "$runtime_owner/tools/dsh/runtime.mjs" check
# 実タスクでは下記run.shがこのexec入口を呼ぶ。
node "$runtime_owner/tools/dsh/runtime.mjs" exec -- -V
```

各worktreeで `install` しても同じ契約を使える。runtimeの選択と作業cwdは明示的に
決め、共有の `node_modules/` を複数worktreeから同時に再インストールしない。

## キー不要のCLI確認

空の一時ディレクトリを使い、呼出元の `.env` や既存の `DSH_HOME` を参照させない。
次はrepo rootで実行する。headlessのタスクは渡さずhelpだけを表示する。

```bash
runtime_entry="$PWD/tools/dsh/runtime.mjs"
check_dir="$(mktemp -d)"
(
  cd "$check_dir" || exit 1
  env -u DEEPSEEK_API_KEY DSH_HOME="$check_dir" DSH_TELEMETRY_DISABLED=1 \
    node "$runtime_entry" exec -- -V &&
  env -u DEEPSEEK_API_KEY DSH_HOME="$check_dir" DSH_TELEMETRY_DISABLED=1 \
    node "$runtime_entry" exec -- --profile headless --help
)
check_status=$?
rm -rf -- "$check_dir"
test "$check_status" -eq 0
```

## 検証と更新

ネットワーク・APIキー・実dshの導入を必要としないテスト:

```bash
.venv/bin/python -m pytest -n 4 tests/e2e/dsh/
node --check tools/dsh/runtime.mjs
```

テストは一時runtimeと偽のCLIで、Node境界、導入欠落、版不一致、globalへの切替禁止、
キーを除外した版検査、別cwdからの実行、引数/stdin/終了コード/シグナルを確認する。
ローカルにNodeが無い場合、CLIテストはskipになる。

固定版を更新する場合は、`package.json` と `package-lock.json` を同時に更新し、
空のディレクトリにこの2ファイルと `runtime.mjs` をコピーして `install` から
`-V` / headless helpまで再検証する。Node対応範囲やCLI仕様の変更も同じ変更で扱う。

## headlessの固定設定

設定検査・スモークにはrepoの `.venv/bin/python`（PyYAMLを含む）も使う。
`headless.patch.yml` でprovider `deepseek-official`、model `deepseek-flash`、
`reasoningEffort: high` を固定する。公式Messages endpointとthinking有効も明示し、
呼出環境の `DEEPSEEK_BASE_URL` や既存のprovider設定による切替を防ぐ。
`session-log-deepseek.config.enabled: false` でセッションログ送信を止め、
`session-telemetry-otel.disabled: true` と **`DSH_TELEMETRY_DISABLED=1`** を併用する。
公式モデルへの通常の推論リクエストはこの設定でも発生する。

`DSH_PERMISSION_MODE` は **`read-only` / `workspace-write`** のみ。
`read-only.patch.yml` / `workspace-write.patch.yml` の対応する方を共有patchの後に適用する。
各patchはsandbox mode・approval `ask`・選択可能なpresetを指定モードに固定する。
`danger-full-access` と不明なモードは検査入口で拒否する（環境に設定済みの場合も拒否）。
headlessには承認応答者がなく、昇格要求は拒否になる。
workspace-writeは起動cwd配下とdshが許可する一時領域への変更を許可する。
どちらのモードも一般の読み取りとネットワークを制限するものではない。

固定版の `dsh-agent-default-model` / `dsh-session-log-deepseek` /
`dsh-llm-deepseek-api-key` / `dsh-sandbox-policy` / `dsh-permission-presets` のREADME・
`lib/*.js` のConfig定義、および実CLIのdumpで確認した。
patchの `config` は行全体の置換であり、部分mergeではない。
`--dump-config` の `!!js` は未評価なので、権限patchにはmodeをリテラルで書く。

### キー不要の有効設定検査

repo rootで次を実行する。API呼び出しは行わない。

```bash
.venv/bin/python tools/dsh/headless_config.py --mode read-only
.venv/bin/python tools/dsh/headless_config.py --mode workspace-write
```

検査はキー等を引き継がない環境変数のallowlistと、空の一時 `DSH_HOME`・cwdを使い、
`runtime.mjs exec -- --profile headless --patch ... --patch ... --dump-config` を実行する。
モデル・effort・provider・アップロード無効・telemetry行の無効化・権限・承認・pluginの実体を
検査し、成功時だけ秘密情報や絶対パスを含まないJSONを出す。
`!!js` はデータとして読み、実行しない。stderrの警告（不明なpatch targetを含む）、
非ゼロ終了、欠落・重複行、モデルや安全値の違い、不正なpatchは非ゼロで停止する。
一時領域は検査後に削除する。`--patch <file>` は代替ファイルの検査用で、
受け付ける内容はレビュー済みの固定policyと同じリテラル行だけである。

これは空のhome上の設定合成を検査する。任意の既存 `DSH_HOME` に追加されたpluginや
継続sessionの保存値まで検査するものではない。スモークは新しいhomeで実行し、
共通wrapper側も専用homeと継続sessionの安全性を管理する。

### 後続wrapperが再利用するもの

`tools.dsh.headless_config` の `check_config(mode)` が上記検査、
`headless_arguments(mode)` が固定のprofileと2つのpatch引数、
`headless_environment(mode, home, source)` がhome・telemetry・modeの環境設定を返す。
Pythonのimportはrepo rootから行う。実プロセスは引き続き `runtime.mjs exec --` で起動し、
検査後に別のpatch/profileを追加しない。helperはAPIキーを探索・保存せず、
呼出側が明示した `source` 環境のキーを実行先へ渡す。
共通wrapperは呼出元のキーを除き、専用ファイルの値だけを子環境に加える。

## 共通wrapper: 実行・継続・停止

公開入口は **`tools/dsh/run.sh`**。そのworktreeの `.venv/bin/python` と固定runtimeを使う。
停止管理にはLinuxの `/proc` とchild subreaper (`prctl`) を使い、利用できなければ起動を拒否する。
起動元cwdに依存せず、task/worktreeの相対パスは呼出元cwdから解決する。
入力タスクは非空のUTF-8 Markdown (`.md`) で、改行・引用符・長文を変えず
`runtime.mjs exec -- --profile headless --patch ... --patch ... --json -` のstdinに渡す。
CLI引数や記録にタスク本文を追加しない（metadataにはbyte数とSHA-256だけ）。

```bash
runner="$PWD/tools/dsh/run.sh"
"$runner" --task /path/to/task.md --worktree /path/to/repo/.claude/worktrees/task \
  --mode workspace-write --timeout 600 > /path/to/private-summary.json
# 調査だけなら --mode read-only。同じroot・同じmodeの会話を続ける:
"$runner" --task /path/to/followup.md --worktree /path/to/repo/.claude/worktrees/task \
  --mode workspace-write --timeout 600 --session-id session-... \
  > /path/to/private-followup-summary.json
```

必須引数は `--task`, `--worktree`, `--mode`, `--timeout`。
タイムアウトは正の有限秒数（小数可）で、実タスクの子プロセス起動から計測する。
設定dump・session検査は別に各30秒を上限とする。
メインtree・存在しないパス・root以外・mainの `.claude/worktrees/` 配下でないlinked worktreeを拒否する。
未許可modeは、引数だけでなく呼出元の `DSH_PERMISSION_MODE` にある場合も拒否する。
wrapperが付ける `TENNIS_LAB_DSH_ACTIVE`、または固定版がshell toolに付ける
`DSH_SHELL=1` がある場合は再帰起動を拒否する。

### APIキー: 専用ファイルだけを読む

ユーザーがrepo外の **`~/.config/tennis-lab/deepseek-api-key`** を一度だけ作る。
以下はユーザー自身が実行し、キー値をシェル履歴・タスク・GitHubに書かない。
**既存ファイルに `install` を再実行すると内容が空になるため、新規作成時だけ使う。**

```bash
install -d -m 700 "$HOME/.config/tennis-lab"
install -m 600 /dev/null "$HOME/.config/tennis-lab/deepseek-api-key"
# エディタ等で上のファイルを開き、内容をキーのみ（末尾改行可）にする。
```

場所の変更手段は **`--key-file <external-file>` だけ**。
通常ファイル・本人所有・0600または0400・非空を確認し、不備ならdshを起動せず作成方法を返す。
symlink/FIFO/ディレクトリ、権限に余計なbitがあるもの、複数行・空白を含む値も拒否する。
ファイルの妥当性を設定検査前に確認し、実タスク起動直前にも読み直す。
キーはその子プロセスの環境の `DEEPSEEK_API_KEY` にだけ入れ、呼出元環境は変更しない。
**呼出元の同名変数は引き継がず、`.env` や `$DSH_HOME/.credentials.yaml` に切り替えない。**
固定版ネイティブの解決順は環境 → `.credentials.yaml` → cwd `.env` → home `.env` だが、
wrapperは有効な専用ファイルを必須とし、home内の資格情報ファイル自体を拒否する。
stdout/stderrのキー値はpipe境界をまたぐ場合も保存前に伏せる。キー値やキーのhashはmetadataに残さない。

### 永続homeと継続session

wrapper専用の **`~/.local/state/tennis-lab-agents/dsh-home`** (0700) を使う。
変更手段は `--dsh-home <external-directory>` のみで、呼出元の `DSH_HOME` は使わない。
home・keyはメインrepoとそのworktreeの外に置く。既存homeの権限が緩い場合は自動修正せず拒否する。
同じhomeの同時実行は `.runner.lock` で拒否し、終了・timeout・signal後に解放する。
PATH/HOME/LANG/LC_ALL/TERM以外の呼出元変数はdshへ引き継がず、
`NODE_OPTIONS`, `NODE_PATH`, `DSH_TOOLS_MODE`, providerの上書き、他のAPIキー、Gitのcwd上書きも除く。
そのうえでhome・mode・telemetry無効と専用tmpを明示する。

毎回 `check_config` による空homeでの検査と、実行先home/cwdに同じpatchを適用した
キーなしの `--dump-config` 検査を行う。dumpは `profiles/headless/cordis.yml` を生成・更新するため、
生成後のprofileの内容hashを `.runner-profile.json` に保存し、次回に照合する。
資格情報・home patch・旧settings・管理外profileや、その後のprofile改変は拒否する。
このhomeは対話dshや他のwrapperと共用せず、手動カスタマイズしない。

`--session-id` は `.runner-sessions.json` に記録済みのsessionだけを受け付ける。
未知ID・cwd不一致はrunの失敗記録を残す。再開前に `session_state.mjs` が同じ固定版の
persistence backendで圧縮ログを読み、semantic replay、cwd・sandbox mode・approval `ask`・
permission preset・過去turn番号を照合する。ログはread handleで読み、書き換えない。
固定版は再開時に保存権限を維持するので、**modeを変える場合は新規sessionを開始する**。
保存モデルではなく今回の固定 `agentDefaultModel` を新しい呼び出しへ渡す固定版の経路を使う。
外部で作ったsessionの取り込みや、履歴不整合を無視する再開は行わない。

### 出力と終了値

stdoutは1行の要約JSONのみ。`exit_code` はwrapperの終了値、`dsh_exit_code` はruntime側の終了値。
`result`, `session_id`, `turn`, `turn_end_reason`, `records_dir`, `final_path`,
`diff_summary`, `usage_totals`, `observed_token_estimate_range_usd`, `usage_missing` を含む。
`truncated_events`（件数）、`truncated_event_types`（種別ごとの件数）、
`truncated_missing_fields`（短縮イベントで欠けた必須フィールド名ごとの件数）も返し、
`summary.json` / `metadata.json` に保存する。フィールド名は `tool_call.input` 等の形式。
生の最終回答・providerのエラーメッセージはstdoutへ出さない。
**exit 0はdshのターン完了だけを示し、タスク成功を意味しない。** `task_success` は常に `null`。
親が最終回答・tracked/untracked差分・指定テストを検証し、取り込みを判断する。
ツール失敗があってもターンがcompletedで終わることがあるため、親の検証は必要である。

| result | wrapper終了値 | 意味 |
|---|---:|---|
| `completed` | 0 | 整合したNDJSONとcompleted turn、dsh exit 0 |
| `api_error` | 10 | terminal turn_endのreasonがerror |
| `invalid_ndjson` | 11 | 不正JSON・未完の行・session/final/turn/stepの欠落や順序違い、または短縮で実行状態を確定できない |
| `dsh_failed` | 12 | dsh/runtimeの非ゼロ終了 |
| `aborted` | 13 | dshがabortしたturn |
| `turn_stopped` | 13 | blocked / max-tokens / interrupted / forkedで終了したturn |
| `timeout` | 124 | 指定秒数を超過 |
| `interrupted` | 130 / 143 | SIGINT / SIGTERM |
| `rejected` / `execution_failed` / `recording_failed` | 2 / 3 | 入力・前提の拒否、起動や記録の失敗 |

timeout・signal・通常終了のどの経路でも独立process groupと追跡した子孫のgroupへTERM→KILLを送り、
leader終了後やsetsidで別groupになった子孫もsubreaperで引き受けて停止・回収する。
打ち切り時も完全に読めたprefixのstep_end usageを保存し、
欠けた続きがあることを明示する。

### 固定版のイベント短縮

固定版 `dsh-headless` の `lib/json-stream-*.js`（`boundJsonLine` / `projectJsonRun`）と
`lib/index.js`（`fail`）の出力形式に従う。`final` 以外では、文字列とオブジェクトのキーを
UTF-8で8KiBに制限し、深さ64以上の配列・オブジェクトを文字列 `[truncated: depth]` にする。
変更があればイベント直下に **`truncated: true`** を付ける。
改行込みで32KiBを超える場合は全ての構造化フィールド（配列・オブジェクト）を削除して
スカラーを残し、それでも超える場合は **`{"type":"...","truncated":true}` だけ**を出す。
JSONでエスケープが増える文字列では、8KiBに収まっていてもこの最終形になり得る。

| 種別 | 通常のフィールドと短縮時の扱い |
|---|---|
| `session` | `sessionId`, `cwd` は文字列。短縮またはtypeのみの形になる |
| `status` | `phase`, `turn`, stepイベントの `step` はスカラー。`turn_end.reason` と任意の `step_end.usage` は構造化フィールドなので行長超過時に削除される |
| `tool_call` | `callId`, `tool` は文字列。`input` は任意のJSON値（不正な引数JSONはraw文字列）で、入れ子の短縮や、配列・オブジェクトの場合の削除がある |
| `tool_result` | `callId`, `status`, `result` は文字列。個別に削除されず、文字列短縮またはtypeのみの形になる |
| `text`, `thinking` | `text` を短縮、またはtypeのみの形になる |
| `error` | `message` を短縮、またはtypeのみの形になる。turn外の失敗で、後続finalを保証しない |
| `final` | `finish()` が直接出力し、本文・行長とも無制限。短縮印は付かない。`final.md` には受信した全文を保存する |

wrapperは短縮印のある正規の省略形を受理して後続イベントを解析する。
印なしの必須フィールド欠落、不正な印、残存フィールドの型違い、短縮印の付いたfinalは破損として扱う。
`truncated_missing_fields` で本文の省略を明示し、本文を補完しない。任意のusageが無い場合は、
短縮前からの欠測と削除を区別できないため、省略フィールド件数には含めず `usage_missing` で示す。
イベント単位の短縮はNDJSON途中切れと区別し、短縮後も終端まで構文が正しければ
`metadata.ndjson_complete: true` となる。
session識別やturn/step、終了理由が失われた場合は `invalid_ndjson` にして完了を推測せず、
後続のfinalと観測できたusageは保存する。本文だけの短縮はcompleted判定・usage集計を妨げない。
固定read-onlyスモークは引き続き全ての短縮印を拒否し、tool inputも辞書を必須とする。

### ローカル記録と使用量の区切り

実行先のignore対象 **`.dsh-runs/<UTC>-<id>/`** を必須とする。
ディレクトリ0700・ファイル0600で作り、公開先へ自動送信しない。
不正worktree等、記録先を安全に決められない拒否では `records_dir: null` を返す。
安全なworktree上でのその他の拒否（未知sessionを含む）は空のevents/finalとmetadata/usageも保存する。

| ファイル | 内容 |
|---|---|
| `events.ndjson`, `stderr.log`, `final.md` | キー値を伏せたイベント・診断と最終回答 |
| `metadata.json`, `summary.json` | 起動引数・実行条件・日時・設定・run/session/turn・状態・要約 |
| `git-before.status`, `git-after.status` | NUL区切りporcelain、tracked/untracked一覧 |
| `git-before.json`, `git-after.json` | HEAD、各差分パス・status・内容hash |
| `git-before.diff`, `git-after.diff`, `git-*.index.diff` | HEADからのtracked差分とindex差分（binaryも含む） |
| `usage.json` | このrunのstep/turn内訳、sessionのrun一覧と別集計、単価・式・欠測・推定範囲 |

既にdirtyなworktreeも受け付け、実行前の差分を保存する。tracked/untrackedの内容hashで、
status記号が同じまま内容だけ変わった場合も検出する。
キー以外の機密内容やローカルパスを含みうるため、記録はローカル確認に使い、そのまま公開しない。

`usage.json` のトップレベルtotalsは**今回のrunだけ**。
`turns` は今回観測したturnごとの集計、`session_runs` は過去runへの対応、
`session` はそのsessionで観測した全runの別集計で、継続前の使用量を今回のtotalsへ混ぜない。
`stream_complete` / `usage_complete` / `missing_billable_buckets` で欠測を示す。
推定単価・時間帯・未計上項目の契約は下記「使用量と推定費用」を参照する。

## APIキー: 固定read-onlyスモーク

以下の旧来スモークは**起動環境のキーのみ必須**とし、既存のDSH_HOMEや資格情報ファイルを探さない。
共通wrapperの専用ファイル読み取りとは別の、固定タスクの検証入口である。
既存の受入条件を維持するため、このスモークの引数・キー解決は変更していない。
ユーザーがシェル履歴にキーを残さず入力する例:

```bash
read -r -s -p 'DeepSeek API key: ' DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY
printf '\n'
```

新規homeを記録先の `home/` に作り、キーの永続保存は行わない。
キーなしなら `MISSING_CREDENTIAL` で起動前に停止する。
`AUTH` / `INVALID_CREDENTIAL` はキー・アカウント設定の確認を案内し、成功扱いしない。
ツールの権限拒否も、dshがexit 0 / completedを返していてもスモーク失敗とする。

## ユーザー設定後のread-only実APIスモーク

**キー設定と実API利用の許可がある親セッションが実行する。** この実装時点では実API未確認。
実行先はREADME.mdのある、変更をcommit済みの専用linked worktreeのrootとする。
runtimeを所有するworktreeのrootから、`smoke_worktree` に実行先を明示する。
同じworktreeで実行するなら `smoke_worktree="$PWD"` でよい。

```bash
smoke_worktree="$PWD"
.venv/bin/python -m tools.dsh.smoke --worktree "$smoke_worktree"
smoke_status=$?
unset DEEPSEEK_API_KEY
test "$smoke_status" -eq 0
```

実行入口は固定の読み取りタスク1件だけをstdinで渡して `--json` で起動する。
`read` でREADME.mdの冒頭を読み、`DSH_READ_ONLY_OK` と短い要約を返す。
常にread-only、新規session、新規home、telemetry無効。既存のNODE_OPTIONS/NODE_PATHと
DSH_TOOLS_MODEは引き継がない。固定180秒で停止し、異常終了時は子プロセス群も止める。
ログは既定で実行先の `.dsh-runs/smoke-<UTC>-<id>/`（gitignore対象）に保存する。
`--records-dir <new-directory>` でrepo外またはgitignore対象の新規ディレクトリも指定できる。
ディレクトリ700・ファイル600で保存するが、**raw記録はローカルに保管し公開しない**。

| 記録 | 内容 |
|---|---|
| `events.ndjson`, `stderr.log` | dshの生イベント・診断。秘密情報や内部パスを含む可能性がある |
| `final.md`, `task.md` | 最終回答と固定タスク |
| `git-before.status`, `git-after.status` | `git status --porcelain=v1 --untracked-files=all -z` の前後記録 |
| `metadata.json` | cwd・起動引数・home・日時・session・turn_end・終了値・判定 |
| `usage.json` | stepごとのusage、欠測、単価・式・推定額・未計上項目 |
| `summary.json` とstdout | 固定値・判定・usage合計・最終回答のhashのみ。任意のモデル本文や絶対パスは含めない |

合格条件は、設定検査成功、exit 0、1つのsessionとcompletedの `turn_end`、完全なstepと
terminal `final`、READMEを読むツールの正常完了、期待markerと空でない要約、
実行前後のcleanなgit status一致。1つでも違えば入口は非ゼロを返す。
親はローカルの最終回答・要約内容と記録を確認して実APIの受入条件を確定する。
このスモークの合格は任意の実装タスクの成功判定には使わない。

### 使用量と推定費用

固定版は `status/step_end.usage` に `inputTokens`, `outputTokens` と任意のcache/total/reasoning項目を出す。
**`inputTokens` は未キャッシュ入力**で、`cacheReadTokens` / `cacheWriteTokens` は別枠。
stepのusageはdshが報告した試行分の合計であり、NDJSONのtext長からは推定しない。
不足するbucketやstepは `null` にし、0へ補完しない。reasoningはoutputへ二重加算しない。

単価の正本は[DeepSeek公式価格表](https://api-docs.deepseek.com/quick_start/pricing/)。
確認日・USD単価・計算式は共有 `accounting.py` の `PRICE` に1つだけ置き、
wrapperとスモークのどちらも毎回 `usage.json` に保存する。2026-10-19に公式表を再確認した。
課金時刻・中国の休日を解決しないため、取得済みtokenについてoff-peak/peak両方の推定範囲を記録する。
必要なbucketが欠測、または別単価を確認していないcacheWriteが非ゼロなら推定額も `null`。
補助リクエスト（session titleやcompaction等）・Search/Fetch・その他の料金は未計上として残す。
`complete: false` / `billing_total_usd: null` は請求確定額でないことを示す。
親の実API実行時に公式価格表を再確認し、変更があればこのsnapshotを更新する。
ピークは中国祝日を除く平日01:00–04:00 / 06:00–10:00 UTC。
時刻や休日を確定して片側の価格を選ばず、両価格による範囲を出す。

### headless検証の範囲

`test_headless.py` はネットワーク・キー不要の偽CLIで、空の検査環境、危険モード、
patchと有効設定の食い違い、認証エラー、権限拒否、NDJSON異常、usage欠測、
git状態変化、ローカル記録の権限・公開要約への秘密情報混入防止を検証する。
ローカル固定版を導入済みなら、実物のdumpとread-onlyのファイル権限拒否もAPIなしで検証する。
未導入ならその3ケースだけskipする。実APIスモークは自動テストに含めない。


`test_runner.py` はfake CLIと使い捨てlinked worktreeでstdin、キー専用ファイル、
拒否・失敗状態、継続、権限/履歴の不一致、usage区切り、差分、キー伏せ、記録権限、
timeout/SIGINT/SIGTERM後の子孫停止を検証する。`test_session_state.py` は導入済みなら
実backendのzstd fixtureを読み、内容を変更しないことと永続profileの反復dumpをAPIなしで確認する。
未導入なら実backendの3ケースだけskipする。

```bash
.venv/bin/python -m pytest -n 4 tests/e2e/dsh/
bash -n tools/dsh/run.sh
node --check tools/dsh/session_state.mjs
```
