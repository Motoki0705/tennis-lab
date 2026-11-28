# #2714 本番公開（一時比較ページ削除含む）と公開後実表示確認の指示

Tier 1 代表 6 ページの本番 GitHub Wiki 公開（および役割を終えた一時比較ページ 3 件・一時アセット 44 件の削除）について、ユーザーから承認（「承認します」）を取得しました。
先行して `#2700`（`Meiji-4cam-Benchmarks.md`）の公開が完了しています。以下の手順で人物モデル比較 5 ページ群と 68 アセットファイルを本番 GitHub Wiki へ公開し、あわせて一時比較ページを削除して、公開後の実レンダリング検証を完了してください。

## 手順
1. `tmp/wiki/` で最新の Wiki `master` を取り込む（`git -C tmp/wiki pull --rebase`）。`Meiji-4cam-Benchmarks.md` の最新コミットを取り込んだ状態にする。
2. 役割を終えた一時比較ページ 3 件と一時アセットを削除する：
   - `_Preview-Wiki-Style-Comparison.md`
   - `_Preview-Wiki-Style-Comparison-AsciiDoc.asciidoc`
   - `_Preview-Wiki-Style-Comparison-RST.rst`
   - `assets/_preview-style-comparison/` 配下すべて
3. 改訂対象 5 ページ（`Person-Model-Comparison-Overview.md`, `Person-Model-Comparison-Methods.md`, `Person-Detector-and-ECPose-Comparison.md`, `Top-Down-Pose-Model-Comparison.md`, `Body-Recovery-FastHMR-vs-GVHMR.md`）と `assets/<page-slug>[local path]` 配下 68 ファイルに対して、`tools.wiki_assets verify --output tmp/person-revision/bundle --wiki-page ...` を再実行して全 5 ページ PASS を確認する。
4. 差分が「対象 5 ページ変更 + `assets/<page-slug>[local path]` 68 ファイル追加 + 一時比較 3 ページ・一時アセット 44 ファイル削除」のみであることを `git -C tmp/wiki status` で確認し、日本語コミットメッセージ（`Co-Authored-By` なし）で commit・`git push origin master` する。
5. 公開後の本番 5 ページすべてについて、実 GitHub Wiki 上で以下を検証・記録する（保存先: `tmp/person-revision/published-check.md` および `screenshots-published/`）：
   - 全 5 ページの匿名 HTTP 200 確認
   - 全 68 画像（SVG 34 件・PNG 34 件）の配信 HTTP 200・MIME・バイト一致確認
   - 削除した一時比較 3 ページが非公開/不在（リダイレクトまたは新規作成画面）であることの確認
   - Chromium 4 条件（1440×980 ライト/ダーク、390×844 ライト/ダーク）での全 5 ページ実表示確認（既定表示の横スクロール数、全 SVG デコード確認、折りたたみ開閉、Mermaid 3 図描画、代表スクリーンショット目視確認）
6. **本体リポジトリの変更・commit・PR 作成・Issue コメント投稿は行わないでください。**

## 報告・終了形式
最終メッセージに、公開した Wiki commit hash、公開 5 ページ URL と全 68 アセット配信・一時比較ページ削除・Chromium 4 条件の実測・目視結果を簡潔に報告し、末尾に次の 1 行を出力してください：
`PARENT_RESULT {"schema_version":1,"state":"done","wait":null}`
