# DeepSeek Harness

`@deepseek-ai/dsh@0.2.0-rc.2` の導入と実行環境の確認を扱う。
[公式リポジトリ](https://github.com/deepseek-ai/deepseek-harness) は developer preview
として互換性の破壊を予告しているため、`latest` やglobalの `dsh` は使わない。
固定runtimeとheadlessのモデル・安全設定、および1回のread-only疎通を扱う。
任意タスクの実行・継続・差分管理を行う共通wrapperは後続の対象（#1264）。

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
実タスク用の安全設定・APIキー・ログ設定は下記のheadless設定と後続wrapper側で指定する。

Node未対応、未導入、metadataまたは実CLIの版不一致、CLI起動失敗は終了コード1。
引数の誤りは2で、導入・実行を開始しない。実行時に依存を自動取得しない。

別の専用worktreeから既存の導入物を使う場合は、入口の絶対パスを明示する。
たとえば `runtime_owner` を導入済みworktreeに設定してから、作業worktreeで以下を実行する。
`exec` のcwdは作業worktreeのままになる。

```bash
node "$runtime_owner/tools/dsh/runtime.mjs" check
# 実タスクでは後続の共通wrapperがこのexec入口を呼ぶ。
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
任意タスク、session継続、汎用timeout、差分保存、run/session単位の課金共通化は#1264の責務。

## APIキーとDSH_HOME

固定版のキー解決順は、起動環境の **`DEEPSEEK_API_KEY`** →
`$DSH_HOME/.credentials.yaml` → 起動cwdの `.env` → `$DSH_HOME/.env`。
キー値をpatchやタスク本文に書かない。repoの `.env` も使用しない。
推奨はシェルへの一時入力かシークレット管理からの環境変数注入。
シェル履歴にキーを残さず入力する例（ユーザー自身が実行する）:

```bash
read -r -s -p 'DeepSeek API key: ' DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY
printf '\n'
```

ネイティブdshは、repo外の専用 `DSH_HOME`（未指定時は `~/.dsh`）の
`.credentials.yaml` も利用できる。固定版の形式は次のとおりで、実際の値はこのファイルやrepoに書かない。
保存する場合は親ディレクトリを700、ファイルを600にし、共有しない。

```yaml
version: 1
refs:
  DEEPSEEK_API_KEY: <user-supplied-secret>
```

以下のスモークは**起動環境のキーのみ必須**とし、既存の `DSH_HOME` や上記ファイルを探さない。
新規homeをローカル記録先の `home/` に作り、キーの永続保存は行わない。
キーなしなら `MISSING_CREDENTIAL` で起動前に停止する。
`AUTH` / `INVALID_CREDENTIAL` はキー・アカウント設定の確認を案内し、成功扱いしない。
ツールの権限拒否も、dshがexit 0 / completedを返していてもスモーク失敗とする。
公開出力に生のエラーメッセージや最終回答を転載しない。

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

### 使用量とスモーク1回分の推定費用

固定版は `status/step_end.usage` に `inputTokens`, `outputTokens` と任意のcache/total/reasoning項目を出す。
**`inputTokens` は未キャッシュ入力**で、`cacheReadTokens` / `cacheWriteTokens` は別枠。
stepのusageはdshが報告した試行分の合計であり、NDJSONのtext長からは推定しない。
不足するbucketやstepは `null` にし、0へ補完しない。reasoningはoutputへ二重加算しない。

単価の正本は[DeepSeek公式価格表](https://api-docs.deepseek.com/quick_start/pricing/)。
スモーク用の確認日・USD単価・計算式は `smoke.py` の `PRICE` に1つだけ置き、毎回 `usage.json` に保存する。
課金時刻・中国の休日を解決しないため、取得済みtokenについてoff-peak/peak両方の推定範囲を記録する。
必要なbucketが欠測、または別単価を確認していないcacheWriteが非ゼロなら推定額も `null`。
補助リクエスト（session titleやcompaction等）・Search/Fetch・その他の料金は未計上として残す。
`complete: false` / `billing_total_usd: null` は請求確定額でないことを示す。
親の実API実行時に公式価格表を再確認し、変更があればこのsnapshotを更新する。
run/session/turn全体の集計と単価管理の共通化は#1264へ引き継ぐ。

### headless検証の範囲

`test_headless.py` はネットワーク・キー不要の偽CLIで、空の検査環境、危険モード、
patchと有効設定の食い違い、認証エラー、権限拒否、NDJSON異常、usage欠測、
git状態変化、ローカル記録の権限・公開要約への秘密情報混入防止を検証する。
ローカル固定版を導入済みなら、実物のdumpとread-onlyのファイル権限拒否もAPIなしで検証する。
未導入ならその3ケースだけskipする。実APIスモークは自動テストに含めない。
