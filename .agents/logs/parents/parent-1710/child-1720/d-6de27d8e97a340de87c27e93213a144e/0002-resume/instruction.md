依存先がすべて main にmergeされた: #1716・#1717・#1718、#1719（PR #1793、merge commit 3f11ab019）。#1793 はmerge前にrebaseと修正（undo・exportの公開照合）が入ったため、このbranchの土台 8817935 とはhashが異なる。

1. worktreeがcleanであることを確認し、このPR自身の3コミットだけを最新の `origin/main` へ載せ替える（`git rebase --onto origin/main 88179353ef7af6eaa2b5caf818b024cae2518578` 相当）。競合はこのPRの意図を保って解消する。特に、このPRで加えたserviceの閲覧view分岐とファイル名列挙の追加が、main上のserviceの変更（#1786 のvisual接続、undo・exportの照合）と整合するかを確かめる。README 3言語版の二重記載を作らない。
2. PRのbaseを `main` に変更し、差分がこのPRの変更だけであることを確かめる。本文の依存PRの記述をmerge済みに更新する。
3. main には #1786 の managed renderer（v1/v2）と `StrictHTMLParser` が入っている。実service＋偽editorで、図（宣言的グラフのslider）を含む文書を表示し、iframe内でrendererが動き、sliderの入力状態が更新後も保持されること、CSPが `sandbox="allow-scripts"` のiframeでも効くことを、ブラウザ検証に1シナリオ加えて確かめる。画像要求の `image_generation_unavailable` がUIで分かる表示になるかも確かめる。
4. 関連テストをローカルで実行し、force-push後の対象HEADのCI結果を確認する。

最終報告: HEAD・CI結果、載せ替えの境界と競合解消の方針、追加したブラウザ検証と結果、serviceへの変更（行数と理由）。目安は最大3時間。
