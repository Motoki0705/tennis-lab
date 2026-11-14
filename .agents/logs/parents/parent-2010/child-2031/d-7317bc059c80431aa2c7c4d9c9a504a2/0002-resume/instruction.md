# PR #2056 を最新 main へ rebase する

PR #2056 は現在 main と衝突しています（CONFLICTING）。main には #2038・#2035 のほか、他の親のPRが複数入っています。

1. `codex/dispatcher-triage-io` を `origin/main` へ rebase し、衝突を解消する。#2029 の client 契約、#2035 のラベル限定探索、他の親が入れた変更の意図を保つ。
2. 関連テストを rebase 後のHEADで再実行し、force-push して、対象HEADの CI（Python）の成功を確認する。PR本文の検証欄を更新する。
3. 並行して #2040 の PR #2054（`issues/cli.py`・`parent/github.py`・`parent/cli.py` の通信層）がまだ開いています。こちらが先にmergeされる予定なので、#2054 を考慮した変更は不要です。

時間の上限は3時間。止めてよいのは、他のPRの変更と両立しない衝突が見つかり、親の判断が要る場合だけ。その場合は衝突箇所と選択肢を報告する。
