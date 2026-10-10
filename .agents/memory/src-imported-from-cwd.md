---
title: "src パッケージはインストールされておらず、カレントディレクトリから import される"
type: gotcha
applies_to: all
source: "#525 asym 実験の無効化（2026-06-20）、#507、#576"
created: 2026-06-20
last_verified: 2026-10-10
evidence: "repo外のディレクトリで `.venv/bin/python -c \"import src\"` を実行すると ModuleNotFoundError になる。worktree内で実行すると src.__file__ はそのworktreeを指す"
---

`.venv` に tennis-lab の editable install はない。`python -m src....` は、カレントディレクトリの `src/` を import する。どの `.venv/bin/python` を使うかではなく、**どこで実行したか**で、使われるコードが決まる。

- メインtreeで実行すると、worktreeで追加したオプションは存在しない。新しいconfigキーが既存コードに黙って無視されることがある（#535 の asym 実験は、メインtreeで実行されて無効になった）。
- training-queue は、enqueue したときのカレントディレクトリをjobに記録する。worktreeのコードで学習したいときは、worktreeをカレントディレクトリにしてenqueueする。

**使い方:** worktreeのコードを試すときは、worktreeに `cd` してから実行する。学習の開始時ログ（パラメータ数など）で、意図したコードが動いているかを確かめる。新しいconfigキーは Hydra の struct モードのおかげで、古いコードでは読み込みエラーになる。これが古いコードの検出に使える。
