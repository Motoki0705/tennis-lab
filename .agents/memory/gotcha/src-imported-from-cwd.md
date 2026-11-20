---
title: "共有venvで実行するsrcはカレントディレクトリのcheckoutから読まれる"
summary: "共有.venvで使われるsrcは実行時のカレントディレクトリで決まる。試すworktreeから実行・enqueueする。"
applies_to: all
source: "#525 asym 実験の無効化（2026-06-20）、#507、#576"
created: 2026-06-20
last_verified: 2026-11-20
evidence: "repo内外で共有venvの importlib.util.find_spec('src') を比較し、storage_tier prepare_worktree とtraining queueのcwd保存を確認する。"
citations:
  - path: ".agents/ops/storage_tier/layout.py"
    symbol: "def prepare_worktree"
  - path: ".agents/ops/storage_tier/README.md"
    symbol: "#worktrees-and-service"
  - path: ".agents/skills/training-queue/scripts/training_queue.sh"
    symbol: "_json_str \"$PWD\""
  - path: "AGENTS.md"
    symbol: "#学習の実行"
---

共有 `.venv` を使って `python -m src....` を実行すると、カレントディレクトリの `src/` を import する。2026-11-20の確認では、共有 `.venv` に対してrepo外から `src` を検索すると見つからず、作業worktreeからの `src.__file__` はそのworktreeを指した。

- #535のasym実験では、メインtreeから実行したためにworktreeの変更が使われなかった。
- training queueはenqueue時のカレントディレクトリをjobに記録する。worktreeから投入し、AGENTS.md に従って `TRAINING_QUEUE_DIR` はメインtreeのqueueを指定する。worktreeの `outputs/`・`data/`・`ckpt/`・`.venv` のリンクは [prepare-worktree](../../ops/storage_tier/README.md#worktrees-and-service) でまとめて準備する。

**使い方:** 試すコードのworktreeに `cd` してから実行する。開始時ログのパラメータ数や `src.__file__` で、意図した版が動いているかを確かめる。
