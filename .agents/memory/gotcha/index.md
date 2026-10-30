# gotcha — 落とし穴

気づきにくい落とし穴、静かに失敗する挙動。1行に1エントリを書き、リンク文字列はエントリの `title` と一致させる。形式は [memory skill](../../skills/memory/SKILL.md#形式) を参照。

- [実行中のbashスクリプトをその場で書き換えると、実行中のプロセスが新しい内容を途中から実行する](bash-script-edit-while-running.md)
- [codex-scheduled-followup の timer は、作成時の cli_heartbeat.py を絶対パスで実行し続ける](codex-followup-timer-script-path.md)
- [.gitignore の汎用パターン（lib/, runs/, *.json）が新しいディレクトリを黙って無視する](gitignore-generic-rules-hide-new-paths.md)
- [PRで submodule の pointer が進んでも、メインtreeの submodule の checkout は古いまま残る](main-submodule-checkout-lags-pointer.md)
- [pre-commit の mypy は staged ファイルだけを --follow-imports=skip で検査する](precommit-mypy-follow-imports-skip.md)
- [spin setup-nht は、実行したtreeの third_party/nht に全体共通の nht-render を結び付ける](setup-nht-binds-global-tool-to-checkout.md)
- [メインtreeの共有 .venv は pyproject.toml / uv.lock の依存追加に自動で追従しない](shared-venv-lags-lockfile.md)
- [src パッケージはインストールされておらず、カレントディレクトリから import される](src-imported-from-cwd.md)
- [tests/ はPythonパッケージではないので、テストモジュール同士で import できない](tests-dir-not-a-package.md)
