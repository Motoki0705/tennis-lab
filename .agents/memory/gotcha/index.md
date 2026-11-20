# gotcha — 落とし穴

エントリから自動生成する。更新手順は [memory skill](../../skills/memory/SKILL.md#形式) を参照。

- [親CLIのローカルE2Eは実セッションのAGY識別子を引き継ぐと登録に失敗する](agents-ops-local-failures-not-regressions.md)
- [実行中のbashスクリプトをその場で書き換えると、実行中のプロセスが新しい内容を途中から実行する](bash-script-edit-while-running.md)
- [Claude Codeのbackgroundの監視は、会話のプロセスが終わると一緒に止まり、再開した会話へ通知は届かない](claude-background-monitor-dies-with-session.md) (claude)
- [codex-scheduled-followup の timer は、作成時の cli_heartbeat.py を絶対パスで実行し続ける](codex-followup-timer-script-path.md)
- [Codex workerのpromptで途中終了を数値なしで許すと、時間が残っていても早々に途中終了する](codex-worker-defers-without-numeric-budget.md)
- [アイドル中のColab VMが回収され、取得済みの入力が消えた](colab-idle-vm-reclaimed.md)
- [.gitignore の汎用パターン（lib/, runs/, *.json）が新しいディレクトリを黙って無視する](gitignore-generic-rules-hide-new-paths.md)
- [PRで submodule の pointer が進んでも、メインtreeの submodule の checkout は古いまま残る](main-submodule-checkout-lags-pointer.md)
- [メインcheckoutの定期同期はlockや作業中の変更で延期される](main-tree-main-lags-origin.md)
- [pre-commit の mypy は staged ファイルだけを --follow-imports=skip で検査する](precommit-mypy-follow-imports-skip.md)
- [spin setup-nht は、実行したtreeの third_party/nht に全体共通の nht-render を結び付ける](setup-nht-binds-global-tool-to-checkout.md)
- [メインtreeの共有 .venv は pyproject.toml / uv.lock の依存追加に自動で追従しない](shared-venv-lags-lockfile.md)
- [共有venvで実行するsrcはカレントディレクトリのcheckoutから読まれる](src-imported-from-cwd.md)
- [共有venvで import tests は外部の同名パッケージを指すことがある](tests-dir-not-a-package.md)
