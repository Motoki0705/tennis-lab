# #2602 子Issue 担当: 人物モデル比較 5 ページ群の高密度 Typst 表・Matplotlib グラフ適用と公開前検証

`main` にマージされた共通生成基盤（`tools/wiki_assets/`）と執筆ガイドライン（`docs/wiki/README.md`）を用いて、人物モデル比較 5 ページ群を改訂し、本番 Wiki 公開前の全検査（正本照合・転記検証・情報欠落・リンク・ローカル 4 条件実表示比較）を完了してください。

## 厳守事項・境界
- **本番 GitHub Wiki への `git push` はまだ行わないでください！** 本番公開は親レビューおよび人間確認ゲート（Tier 1 計 6 ページの公開前確認）の承認後に行います。
- **本体リポジトリのファイルは変更・commit・PR 作成しないでください。** Issue コメント投稿も行わず、親への報告は最終メッセージと worktree 内 `tmp/person-revision/` の成果物にまとめてください。
- GitHub Wiki（`https://github.com/Motoki0705/tennis-lab.wiki.git`）はこの worktree の `tmp/wiki/` に clone し、対象 5 ページと `tmp/wiki/assets/<page-slug>[local path]` のみを更新してください（他の既存公開ページ・`_Sidebar.md`・`_Footer.md` は変更しない）。

## 対象 5 ページと参照ガイド
- 対象ページ:
  1. `Person-Model-Comparison-Overview.md` (`person-model-comparison-overview`)
  2. `Person-Model-Comparison-Methods.md` (`person-model-comparison-methods`)
  3. `Person-Detector-and-ECPose-Comparison.md` (`person-detector-and-ecpose-comparison`)
  4. `Top-Down-Pose-Model-Comparison.md` (`top-down-pose-model-comparison`)
  5. `Body-Recovery-FastHMR-vs-GVHMR.md` (`body-recovery-fasthmr-vs-gvhmr`)
- 必ず最初に読む文書:
  - `docs/wiki/README.md`（GitHub Wiki 執筆ガイドライン）
  - `tools/wiki_assets/README.md`（生成・検証 CLI の使い方）
  - `[local path]`（人物モデル比較 5 ページの設計方針）

## 実施手順

### 1. 共通生成基盤によるアセット＆スニペット生成
- 既存のキャッシュまたは明示パス（例: `[local path]` があれば参照可、または `tools/wiki_assets` の標準キャッシュ取得）を使って、5 ページ分のバンドルを `tmp/person-revision/bundle/` へ生成し、`verify --rerender` で自己検証してください：
  ```bash
  .venv/bin/python -m tools.wiki_assets generate --output tmp/person-revision/bundle
  .venv/bin/python -m tools.wiki_assets verify --output tmp/person-revision/bundle
  ```
- 生成された `tmp/person-revision/bundle/assets/<page-slug>[local path],png}` を `tmp/wiki/assets/<page-slug>[local path]` へコピーしてください。

### 2. 対象 5 ページの改訂（`docs/wiki/README.md` 準拠）
5 ページすべてで以下を満たしてください：
- **冒頭定型メタデータ欄の完全廃止**:
  - 冒頭の「項目／内容」表（「想定読者」「検索キーワード」「正本ファイル」）、定型戻りリンク（`> ← ...へ戻る`）、定型「基準日 2026-11-26」ブロックを削除し、タイトル直後から結論・本題に入れる構成にする。
  - 旧冒頭にあった必要な出典・正本リンク、基準日・判断日・計測日、関連ページ導線は本文の該当箇所および末尾の「出典・関連ページ」節へ確実に移設・維持する。
- **マーカー付き生成図表ブロックの埋め込み**:
  - 生成スニペット（`tmp/person-revision/bundle/<page-slug>.md`）の各 `<!-- wiki-assets:begin … -->`〜`<!-- wiki-assets:end … -->` ブロックを**改変せずそのまま**適切な節へ配置する（ブロック前後に空行を置く）。
  - 旧来の本体 repo `?raw=true` の Pareto PNG 参照（`Person-Detector-and-ECPose-Comparison.md`・`Top-Down-Pose-Model-Comparison.md`）は、新しく生成した `assets/<page-slug>[local path]` のアウトライン化 SVG＋原寸/互換 PNG リンク＋折りたたみテキスト表＋固定正本リンクへ置き換える（旧来の正本・再現スクリプトへのリンクが必要な場合は本文・出典に残す）。
- **非数値の長文表の階層化と Markdown 表残存基準**:
  - ユーザー判断履歴、モデル系列とライセンス・利用条件、入力・出力契約、構成と入出力、クリップ選定理由、実行不能・未計測の原因一覧など、セルが長く横幅を圧迫する非数値表は、`docs/wiki/README.md` に従って階層リスト・日付別項目・`<details>` 折りたたみ（空行ルール厳守）へ整理する。
  - 2〜4 列・おおむね 12 行以内・セル短め・390px 横スクロールなしの簡潔な索引・対応表・理由コード表（例: Issue/PR 対応表など）は Markdown 表として残してよい（残す場合は 390px で横スクロールしないことを確認）。
- **情報・解釈・注意事項の完全保全**:
  - 疑似GTの限界、未ラベル人物の precision/AP 未定義、未計測・変換異常・ライセンス除外・打ち切り・重みロードのみ、NVML 増分 vs 全体値の区別、B1/B2/B4/B8 の分離、既定モデルと `recipe=standard` の維持、FastHMR の速度見送り・目視未実施・定性評価のみ制約など、旧版 5 ページの全事実・全出典リンク・全 Issue/PR リンクを欠落なく維持する。
  - 既存の見出しアンカー（他ページからリンクされているもの）を壊さないよう確認する。

### 3. 公開前全検査（成果物: `tmp/person-revision/`）
1. **共通ツールによる厳密照合 (`verify`)**:
   ```bash
   .venv/bin/python -m tools.wiki_assets verify \
     --output tmp/person-revision/bundle \
     --wiki-page tmp/wiki/Person-Model-Comparison-Overview.md \
     --wiki-page tmp/wiki/Person-Model-Comparison-Methods.md \
     --wiki-page tmp/wiki/Person-Detector-and-ECPose-Comparison.md \
     --wiki-page tmp/wiki/Top-Down-Pose-Model-Comparison.md \
     --wiki-page tmp/wiki/Body-Recovery-FastHMR-vs-GVHMR.md
   ```
   全 5 ページ・全 34 組の SVG/PNG SHA-256、マーカー付きテキストブロック全文、公開禁止情報、`<details>` 空行、相対リンク、`alt` テキストがすべて PASS することを確認・記録する。
2. **情報欠落・リンク突合監査 (`tmp/person-revision/migration-audit.md` / `.json`)**:
   - 旧版 5 ページの全リンク・主要数値・判断記録・見出しアンカーが新版に維持されていることを自動/手動で突合し、欠落 0 件を確認する。
   - 5 ページ相互リンクと他 Wiki ページへの相対リンク（`Meiji-4cam-Benchmarks` 等）がすべて有効であることを確認する。
3. **ローカル Chromium 4 条件表示比較 (`tmp/person-revision/render-check.md` & `screenshots/`)**:
   - 896px / 390px（および 326px）× ライト `#ffffff` / ダーク `#0d1117` で 5 ページの変更前後を比較し、高さ・横スクロールブロック数・全 SVG 画像の表示・`<details>` 折りたたみ開閉・Mermaid 描画を確認する。
   - 代表スクリーンショットを採取して目視確認し、折り返し崩れ・見切れ・文字欠落がないことを確かめる。

## 報告・終了形式
- 最終メッセージに、改訂した 5 ページと `assets/` のファイル数・サイズ、`tools.wiki_assets verify` の結果、旧版→新版の表数・横スクロール・リンク/情報欠落突合結果、ローカル 4 条件表示比較と目視結果を簡潔にまとめ、末尾に次の 1 行を出力してください：
`PARENT_RESULT {"schema_version":1,"state":"done","wait":null}`
