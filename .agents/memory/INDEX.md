# 共有memory 索引

memoryは規則ではなく、検証待ちの観察である。形式と、追加・更新・削除の手順は [README.md](README.md) を参照。1行に1エントリを書き、リンク文字列はエントリの `title` と一致させる。

- [src パッケージはインストールされておらず、カレントディレクトリから import される](src-imported-from-cwd.md) — gotcha
- [新しいworktreeには .venv がないので、メインtreeの .venv へのsymlinkを張る](worktree-needs-venv-symlink.md) — workflow
- [pre-commit の mypy は staged ファイルだけを --follow-imports=skip で検査する](precommit-mypy-follow-imports-skip.md) — gotcha
- [.gitignore の汎用パターン（lib/, runs/, *.json）が新しいディレクトリを黙って無視する](gitignore-generic-rules-hide-new-paths.md) — gotcha
- [tests/ はPythonパッケージではないので、テストモジュール同士で import できない](tests-dir-not-a-package.md) — gotcha
- [実行中のbashスクリプトをその場で書き換えると、実行中のプロセスが新しい内容を途中から実行する](bash-script-edit-while-running.md) — gotcha
- [codex-scheduled-followup の timer は、作成時の cli_heartbeat.py を絶対パスで実行し続ける](codex-followup-timer-script-path.md) — gotcha
- [Claude Code の sandbox 内で起動した training-queue worker は、セッション終了時に殺される](claude-sandbox-kills-queue-worker.md) — gotcha (claude)
- [WSL2 で 16GB の VRAM を使い切ると、エラーにならずホスト全体が周期的に固まる](wsl2-vram-overflow-freezes-host.md) — environment
- [WSL2 のホストRAM不足では、学習プロセスがtracebackなしで黙って殺される](wsl2-host-ram-silent-kill.md) — environment
- [test推論をbackfillできない古いckptは、結果がknowledgeノードにあれば削除してよい（ユーザー判断）](ckpt-force-delete-policy.md) — decision
