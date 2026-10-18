# DeepSeek Harness runtime

`@deepseek-ai/dsh@0.2.0-rc.2` の導入と実行環境の確認を扱う。
[公式リポジトリ](https://github.com/deepseek-ai/deepseek-harness) は developer preview
として互換性の破壊を予告しているため、`latest` やglobalの `dsh` は使わない。
headlessのモデル・権限設定と実タスクの実行wrapperは後続の対象（親issue #1261）。

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
実タスク用の安全設定・APIキー・ログ設定は後続のheadless設定とwrapper側で指定する。

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
.venv/bin/python -m pytest tests/e2e/dsh/test_runtime.py
node --check tools/dsh/runtime.mjs
```

テストは一時runtimeと偽のCLIで、Node境界、導入欠落、版不一致、globalへの切替禁止、
キーを除外した版検査、別cwdからの実行、引数/stdin/終了コード/シグナルを確認する。
ローカルにNodeが無い場合、CLIテストはskipになる。

固定版を更新する場合は、`package.json` と `package-lock.json` を同時に更新し、
空のディレクトリにこの2ファイルと `runtime.mjs` をコピーして `install` から
`-V` / headless helpまで再検証する。Node対応範囲やCLI仕様の変更も同じ変更で扱う。
