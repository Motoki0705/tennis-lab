---
title: codex-scheduled-followupのtimerは登録時のcli_heartbeat.pyを絶対パスで参照する
summary: codex-followupのtimerは作成時のスクリプトを絶対パスで参照する。timerが動いている間はそのcheckoutの変更や削除に注意する。
applies_to: all
source: 'PR #1054 作業中の観察（2026-10-10）'
created: 2026-10-10
last_verified: 2026-11-22
evidence: cli_heartbeat.pyのregister_timerはPath(__file__).resolve()をtickのスクリプトに設定する。createとresumeがregister_timerを呼ぶ。
citations:
- path: .agents/skills/codex-scheduled-followup/scripts/cli_heartbeat.py
  symbol: def register_timer
- path: .agents/skills/codex-scheduled-followup/references/cli-heartbeat.md
  symbol: re-registers a paused task's timer
---

`codex-followup-*` のuser systemd timerは、作成または再登録したときに使った `cli_heartbeat.py` の絶対パスで `tick` を呼ぶ。受信セッションのcwdは `--working-directory` に別途設定される。稼働中に参照先のcheckoutを更新すると次のtickから新しいコードを使い、そのcheckoutを消すと起動できなくなる。

helperの形式やcheckoutを変更する前に、アクティブなtimerが呼ぶserviceの `ExecStart` と対象taskの状態を照合する。保存状態に互換性がない変更を検証するときは、コピーした状態を用い、配送する `tick` は実行しない。
