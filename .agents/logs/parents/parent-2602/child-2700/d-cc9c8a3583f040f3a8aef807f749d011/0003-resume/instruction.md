# #2700 本番公開と公開後実表示確認の指示

Tier 1 代表 6 ページの本番 GitHub Wiki 公開について、ユーザーから承認（「承認します」）を取得しました。
以下の手順で `Meiji-4cam-Benchmarks.md` を本番 GitHub Wiki へ公開し、公開後の実レンダリング検証を完了してください。

## 手順
1. `tmp/wiki/` で `git status` を確認し、差分が `Meiji-4cam-Benchmarks.md` の 1 ファイルのみであることを確かめる。
2. `git -C tmp/wiki pull --rebase` で最新の Wiki `master` を取り込む。
3. 公開前最終チェック（公開禁止情報 0 件・旧版 62 リンク維持・`<details>` 空行ルール）を確認したうえ、`Meiji-4cam-Benchmarks.md` を 1 コミットにまとめる（日本語コミットメッセージ、`Co-Authored-By` は付けない）。
4. `git -C tmp/wiki push origin master` で公開する。
5. 公開後の本番ページ `https://github.com/Motoki0705/tennis-lab/wiki/Meiji-4cam-Benchmarks` に対して以下を実測・記録する（保存先: `tmp/meiji-revision/published-check.md` および `screenshots-published/`）：
   - 匿名 HTTP 200 確認
   - Chromium 4 条件（1440×980 ライト/ダーク、390×844 ライト/ダーク）での実ページ描画確認（横スクロールブロック数、Mermaid 描画、3 件の `<details>` 折りたたみ開閉、`#hugging-face-配布版` および短縮アンカーの挙動、サイドバー導線）
   - 公開スクリーンショットの目視確認（文字欠落・崩れ・意図しない表示がないこと）
6. **本体リポジトリの変更・commit・PR 作成・Issue コメント投稿は行わないでください。**

## 報告・終了形式
最終メッセージに、公開した Wiki commit hash、公開 URL の HTTP 200 と Chromium 4 条件の実測結果・目視結果、および確認したアンカー挙動を報告し、末尾に次の 1 行を出力してください：
`PARENT_RESULT {"schema_version":1,"state":"done","wait":null}`
