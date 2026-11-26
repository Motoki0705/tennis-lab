# gotcha — 落とし穴

エントリから自動生成する。更新手順は [memory skill](../../skills/memory/SKILL.md#形式) を参照。

- [親CLIのローカルE2Eは実セッションのAGY識別子を引き継ぐと登録に失敗する](agents-ops-local-failures-not-regressions.md)
- [合意の改訂は進行中のレビュー・マージ後確認を止める](agreement-revision-strands-progression.md)
- [実行中のbashスクリプトをその場で書き換えると、実行中のプロセスが新しい内容を途中から実行する](bash-script-edit-while-running.md)
- [互換表の対象ファイルを触るPRは、PR CIが通ってもマージ後のmain CIを落とす](cache-compat-table-cross-pr-break.md)
- [Claude Codeのbackgroundの監視は、会話のプロセスが終わると一緒に止まり、再開した会話へ通知は届かない](claude-background-monitor-dies-with-session.md) (claude)
- [codex-scheduled-followupのtimerは登録時のcli_heartbeat.pyを絶対パスで参照する](codex-followup-timer-script-path.md)
- [注釈workerは途中終了条件が曖昧だとレビューを早く打ち切ることがある](codex-worker-defers-without-numeric-budget.md)
- [アイドル中のColab VMが回収され、取得済みの入力が消えた](colab-idle-vm-reclaimed.md)
- [.gitignore の汎用パターン（lib/, runs/, *.json）が新しいディレクトリを黙って無視する](gitignore-generic-rules-hide-new-paths.md)
- [記録専用PRは開き直しでなくmain取り込みで比較を更新する](knowledge-pr-stale-test-merge.md)
- [NHTのsubmodule checkoutが古いと学習・描画で古いコードを使うことがある](main-submodule-checkout-lags-pointer.md)
- [merge commit の main CI が開始前に取り消されるとマージ後確認が進まない](post-merge-main-ci-cancelled.md)
- [pre-commit の mypy は staged ファイルだけを --follow-imports=skip で検査する](precommit-mypy-follow-imports-skip.md)
- [spin setup-nht は、実行したtreeの third_party/nht に全体共通の nht-render を結び付ける](setup-nht-binds-global-tool-to-checkout.md)
- [メインtreeの共有 .venv は pyproject.toml / uv.lock の依存追加に自動で追従しない](shared-venv-lags-lockfile.md)
- [共有venvのpython -m srcは起動したcheckoutのソースを読む](src-imported-from-cwd.md)
- [共有venvで import tests は外部の同名パッケージを指すことがある](tests-dir-not-a-package.md)
