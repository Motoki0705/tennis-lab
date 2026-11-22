親Issue #2384 配下の子Issue #2019 の実装担当として、専用worktreeで実装・検証・PR作成まで行ってください。

## 対象Issueと目的
- 担当子Issue: #2019（親 #2384）
- 目的: 状態ファイルの書込み中断で一時ファイルが残存しても `Store.write_json` および同種実装が恒久的な `FileExistsError` で停止・無限再起動しないようにし、symlink・別所有者の保護を維持したうえで dispatcher の失敗原因と復旧手順を journal 出力から特定できるようにする。

## 事前調査で特定した対象箇所（最新コードを再確認すること）
1. `.agents/ops/parent/state.py` (`Store.write_json` L418-L431):
   - 現在 `temporary = path.with_name(path.name + ".tmp")` の固定名 + `O_WRONLY | O_CREAT | O_EXCL` を使用しており、書込み中断で `.tmp` が残ると以後ずっと `FileExistsError` で失敗する。
   - `progression.py`・`dispatcher/schedule.py`・`dispatcher/supervisor.py`・`dispatcher/judgement.py`・`parent/state.py` 等がすべて `Store.write_json` を使用している。
2. `.agents/ops/dispatcher/agent_input.py` (L34付近):
   - 同様に `temporary = path.with_name(path.name + ".tmp")` を使っているため、同じ安全契約で共通化または修正すること。他にも `.agents/ops/` 配下で同じ固定名 `.tmp` + `O_EXCL` パターンがないか念のため点検すること。
3. `.agents/ops/dispatcher/cli_impl.py` (L250-L258付近):
   - 例外時に stdout JSON が汎用の `"dispatcher input/state/runtime error; inspect private records"` のみになっている。状態書込み失敗・残存一時ファイル（symlink・別所有者・危険な状態含む）等の区別可能な原因と復旧手順が、秘匿情報を漏らさずに journal（stderr/stdout）へ出るように改善すること。

## 設計上の必須要件
- 一時ファイルは固有の一時ファイル名（PID・乱数等）を `O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW` で作成し、`fsync` 後に `os.replace` で原子置換する。失敗時は自身が作成した一時ファイルのみ unlink する。
- 対象パス自体や既存一時ファイルが symlink・別所有者等の危険な状態である場合は、決して symlink を辿らず明示的に拒否する（`lstat` / `O_NOFOLLOW` 等で確実に保護し、権限 `0o600` / ディレクトリ `0o700` を維持する）。
- 既存の残存固定名 `<name>.tmp`（通常ファイル）が残っていた場合でも、固有 tmp 名での次回書込みが安全に成功することを確認する。さらに、危険な残存物（symlink・別所有者等）や書込み失敗時は区別可能な例外・診断・復旧手順を返すこと。

## 受入条件（#2019 の Acceptance checklist をすべて満たすこと）
1. 書込み中断で残った通常ファイルの一時ファイルがあっても次回書込みが安全に成功する回帰テスト（`Store.write_json`, `dispatcher/agent_input.py`, `dispatcher/schedule.py`）。
2. 一時ファイルや対象パスが symlink・別所有者等の場合は辿らずに拒否され、リンク先が書き換わらないテスト。
3. dispatcher 失敗時の出力（journal 相当の stdout/stderr）に、区別可能な原因と復旧手順が含まれ、秘匿情報を漏らさないテスト。

## 検証・PR作成・終了ルール
- 関連する既存テスト（`tests/unit/agents_ops/`, `tests/e2e/agents_ops/` の parent/dispatcher 関連）＋新規追加テストを実行し全通過を確認する。
- pre-commit（ruff / mypy）を通してコミット・push し、PR 本文に `Closes #2019`、変更概要、受入条件ごとの検証証拠、テスト結果を日本語で記載する。
- push・PR作成が終わったら、このターン内で CI 待ちループを行わず、最終メッセージ末尾に以下の形式で 1 行だけ出力して終了する（`<PR番号>` と `<40桁HEAD SHA>` は実際の値に置換）：
`PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":<PR番号>,"head":"<40桁HEAD SHA>","resume_on_success":false}}`
