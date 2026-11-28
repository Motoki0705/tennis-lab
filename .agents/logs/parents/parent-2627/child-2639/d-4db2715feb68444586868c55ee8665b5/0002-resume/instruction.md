PR #2727 の正式レビュー（round 1 / `parent-2627-25823aad53aa411fbbc658d0899fd7b8`）で以下の **major 指摘 2件** が入りました。2件とも採用し、最小限かつ正確に修正してください。

## 修正対象（全2件）

### 1. 球検出の 20 px 以内率の分母を明確化する
- 箇所: `paper/tennis_scene_architecture/chapters/ball_detection.tex`（110〜113行目付近または表11のキャプション）
- 指摘内容: 110行目で validation の採点対象 `117,358` 件と観測 `85,473` 件を併記した後、113行目で「全validationの20 px以内率」と記述しているため、掲載値 `1.66%`・`40.90%`・`44.23%` の分母が `117,358` なのか `85,473` なのかが曖昧になる。
- 修正方針: 実験設定文または表11のキャプションの一箇所で、全体の 20 px 以内率を含む位置指標はすべて観測 `85,473` 件（observed）を分母として集計したことを明確に記述する（冗長な留保を増やさず1箇所で簡潔に明記する）。`\ev` 根拠リンク・数値は変えず、必要に応じて `evidence/polish.md` の変更一覧へ反映する。

### 2. `evidence/polish.md` の `@TICK@` プレースホルダーを正規の Markdown バッククォートへ修正する
- 箇所: `paper/tennis_scene_architecture/evidence/polish.md`（全体、特に 61〜69 行目のコードブロックおよび正規表現表）
- 指摘内容: `@TICK@` のプレースホルダーがそのまま残っており、GFM パーサーでインラインコードとシェルコードブロックが崩れている。
- 修正方針: `evidence/polish.md` 内のすべての `@TICK@` を正規のバッククォート（`` ` `` および ```` ``` ````）へ置換し、`@TICK@` の残存が 0 件であること・シェルコードブロックの照合コマンドがそのままコピペ実行して通過することを確認する。

## 再検証・提出手順
1. 変更後に `make_tables.py --check`, `prepare_figures.py`, `check_evidence.py`, および `evidence/polish.md` に記載の照合コマンドをすべて実行し、終了コード 0 を確認する。
2. ステージ済み原稿で `SOURCE_DATE_EPOCH=1795817876 TZ=UTC bash paper/tennis_scene_architecture/build.sh --check-tracked` を空の `build/` から実行し、警告・Overfull・Underfull・文字欠落・未解決参照/引用・未追跡入力 0 件を確認する。
3. 変更が入ったページ（第7章・表11周辺およびページ繰りの影響範囲）を 150 dpi でレンダリングして目視確認し、ページ数・バイト数・新 PDF SHA-256 を `evidence/build.md` へ更新する（確認用 PNG は削除する）。
4. コミット後に再度空の `build/` から決定論的ビルドを行い、コミット前後の PDF バイト一致を確認して `evidence/build.md` のコミット後確認記録も整合させた上で push する（必要なら記録更新コミット後に再照合）。
5. PR #2727 本文の `## 実行証拠` を新 HEAD SHA・新 PDF SHA-256 で更新し、PR CI 完了を待たずに `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` で終了する。
