依存先がすべて main にmergeされた: #1716（PR #1783）、#1718（PR #1786、merge commit 9ad3e934c）、#1717（PR #1787、merge commit f8b3f244c）。#1787 はmerge前にrebaseされたため、このbranchの土台の旧commitとはhashが異なる。

1. worktreeがcleanであることを確認し、このPR自身のcommitだけを最新の `origin/main` へ載せ替える（例: `git rebase --onto origin/main 9b4ec527a1df3af7bbaac87aa08dfcc06e08e2f8`。土台が別のcommitなら正しい境界を確かめる）。競合はこのPRの意図を保って解消し、README 3言語版の二重記載を作らない。
2. PRのbaseを `main` に変更し、差分がこのPRの変更だけであることを確かめる。本文の依存PRの記述をmerge済みに更新する。
3. main に入った #1786 と接続する（最終報告で「統合側に残る」とした点のうち、serviceの責務に入るもの）:
   - serviceが `GeminiEditor(compile_document=...)` に、#1786 の `visuals.render_document`（図slotの展開・CSP・managed script）を渡す。呼出側で恒等変換に落とさない。
   - 画像要求を `image_generation_unavailable` に分類するAssetPreparer（またはそれに当たる注入点）を、#1786 の `ImageGenerator` の利用不可adapterで接続する。人間の判断（2026-11-09）により、閲覧中の新規画像は理由付き未完了で現行版を保持するのが仕様。
   - 偽Geminiと実 `render_document`・実storeを使う統合テストを1件追加し、図を含む編集がcommitまで通ること、画像要求が `image_generation_unavailable` で現行版を変えないことを確かめる。実API呼出しはしない。
4. 関連テストをローカルで実行し、force-push後の対象HEADのCI結果を確認する。

#1720 のworkerがこのbranchの旧HEAD（8817935）の上で作業中なので、HTTP入口・状態名・token受渡しの公開面は変えない。変えた場合は最終報告に列挙する。
最終報告: HEAD・CI結果、載せ替えの境界と競合解消の方針、#1786 接続の内容と追加テストの結果、公開面の変更有無。目安は最大3時間。
