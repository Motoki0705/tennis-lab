# #2540: knowledge Web UI の論文フィルタのテストを直す

担当Issue #2540 の本文を読み、受入条件に沿って直す。親は #2492。

## 状況
- main の `de2cf2436`（#2538 のマージ）から、CI の `knowledge-webui` が失敗している。直前の main `59faa32a2` では成功していた。
- 失敗しているのは `knowledge/webui/tests/browser/library.spec.ts` の「selected paper and experiment paper filter survive reload independently」。`paperFilter=paper-2024-gvhmr` の実験一覧を3件と決め打ちしているが、#2538 で `paper-2024-gvhmr` を参照する実験記録が増え、11件になった（CI ログ: Expected 3, Received 11）。

## 進め方
1. main の最新から作業する。まず手元で失敗を再現する。knowledge/webui の README と CI の workflow を読み、CI と同じ手順で動かす。
2. テストの意図（選択中の論文と論文フィルタが、reload 後もそれぞれ保たれること）を保ったまま、件数の決め打ちをやめる。期待値は実際の knowledge データから求める。フィルタが効いていることを確かめる assertion（フィルタ後の件数がフィルタ前より少ない、一覧の各行が対象論文を参照している、など）は残すか、強める。
3. 同じファイルの他のテストにも、実データの件数を決め打ちしていて記録の追加で壊れるものがあれば、同じ方針で直す。範囲はこのファイルに限る。
4. Web UI 本体の不具合が見つかったら、直さずに根拠付きで報告する。knowledge の記録（nodes・runs・papers）は変えない。
5. 手元でブラウザテストを通し、pre-commit を通してから、PR を作る（本文は日本語、`Closes #2540`）。push 後は CI の結果を待つ。

## 最終回答
直した内容、手元の検証、PR 番号、push した HEAD を書き、CI を待つ `PARENT_RESULT`（`waiting` / `ci`）を1行の正しい JSON で返す。
