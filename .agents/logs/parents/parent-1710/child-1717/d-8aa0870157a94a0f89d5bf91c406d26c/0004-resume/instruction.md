PR #1787 は再レビュー（run parent-1710-c71e35487c2a48ada37aa6c6d8d2335c）で approve になった。ただし、その後 #1718 の PR #1786 が main にmergeされ（merge commit 9ad3e934c）、#1787 は競合している。

1. worktreeがcleanであることを確認し、最新の `origin/main` へrebaseして競合を解消する。README 3言語版は両方の内容を保ち、二重記載を作らない。#1786 で `document.py` が `parsing.StrictHTMLParser` を継承する形に変わっている。editor側の検査がこれと整合するか確かめる。
2. #1786 の実際の compiler（`visuals.render_document` など、図slotを展開する公開関数）を `GeminiEditor(compile_document=...)` に渡したとき、最終候補の digest・意味判定・store.commit が一致することを、偽Geminiを使う統合テスト1件で確かめる。実API呼出しはしない。#1786 の公開関数が DocumentCompiler の型に直接合わない場合は、薄い接続関数をどこに置くべきかを判断し（editor側か、#1719 のservice側か）、理由をPR本文に書く。
3. rebaseと競合解消で、承認済みの内容から意図しない変更が入っていないことを確かめる（`git range-diff` 等）。force-push後の対象HEADのCI結果を確認する。

最終報告: HEAD・CI結果、競合を解消したファイルと方針、range-diffで見た差分の要約、追加した統合テストと結果。目安は最大2時間。
