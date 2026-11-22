依存先 #1716 のPR #1783 が main にmergeされた（merge commit 9442f9879、main の CI 成功）。

1. worktreeがcleanであることを確認し、このPR自身のcommitだけを最新の `origin/main` へ載せ替える（`git rebase --onto origin/main 6c4acc2e27293753f1a08be61ce385510c961cc8` 相当。旧HEAD b0ea6ab に基づくcommitが残っていれば、それも除く）。競合はこのPRの意図を保って解消する。
2. PRのbaseを `main` に変更し、差分がこのPRの変更だけであることを確かめる。本文の「依存PR」記述をmerge済みに更新する。
3. 関連テストをローカルで実行し、force-push後の対象HEADのCI結果を確認する。
4. 人間の判断（2026-11-09、親issue #1710 のnote）: AI生成画像は初版の作成時だけCodex imagegenで生成してimportする。閲覧中の新規画像要求は `image_generation_unavailable` として理由付きの未完了にし、現行版を保持する。これで「閲覧中の画像生成経路の実現可能性」は確定したものとして扱う。PR本文のAcceptance表と「必要な判断」をこの決定に合わせて更新し、READMEの記述が決定と食い違っていれば直す。
