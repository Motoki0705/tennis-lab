依存先 #1716 のPR #1783 が main にmergeされた（merge commit 9442f9879、main の CI 成功）。

1. worktreeがcleanであることを確認し、このPR自身のcommitだけを最新の `origin/main` へ載せ替える（`git rebase --onto origin/main 6c4acc2e27293753f1a08be61ce385510c961cc8` 相当。旧HEAD b0ea6ab に基づくcommitが残っていれば、それも除く）。競合はこのPRの意図を保って解消する。
2. PRのbaseを `main` に変更し、差分がこのPRの変更だけであることを確かめる。本文の「依存PR」記述をmerge済みに更新する。
3. 関連テストをローカルで実行し、force-push後の対象HEADのCI結果を確認する。
4. 実機smokeで編集1件に約271秒かかった点を、PR本文の「下流への影響」に明記する（#1719/#1720 の処理中表示・timeout既定値の前提になる）。timeoutの既定値がこの実測に対して十分かを確かめ、不足なら根拠を付けて調整する。
