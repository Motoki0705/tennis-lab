---
title: "codex-scheduled-followup の timer は、作成時の cli_heartbeat.py を絶対パスで実行し続ける"
type: gotcha
applies_to: all
source: "PR #1054 作業中の観察（2026-10-10）"
created: 2026-10-10
last_verified: 2026-10-10
evidence: ".agents/skills/codex-scheduled-followup/scripts/cli_heartbeat.py が timer の起動コマンドに Path(__file__).resolve() を書き込む"
---

`codex-followup-*` の user systemd timer は、`create` を実行したときの `cli_heartbeat.py` の絶対パスで `tick` を実行する。ふつうはメインtreeのパスなので、メインtreeにhelperの変更を取り込むと、動いている follow-up の挙動がすぐに変わる。worktreeから作った timer は、worktreeのコピーを指す（worktreeを消すと壊れる）。

**使い方:** `cli_heartbeat.py` の state の形式やサブコマンドを変える前に、`systemctl --user list-timers | grep codex-followup` と `~/.local/state/codex-followups/` で、動いている task を確かめる。互換性のない変更には migration を用意し、動いている task dir のコピーで試す（コピーに対して `tick` は実行しない）。
