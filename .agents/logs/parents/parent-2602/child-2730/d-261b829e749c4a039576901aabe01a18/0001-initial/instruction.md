# #2602 残31ページ展開 担当 `pillar-ball-research`: Pillar III・VI（11ページ: Tier 2×7, Tier 3×4）の改訂と公開前検証

`docs/wiki/README.md` および `docs/wiki/rendering-comparison.md`（および `#2694` の `inventory.md`）に従い、担当する Pillar III・VI の **11 ページ** を改訂し、公開前全検査を完了してください。

## 厳守事項・境界
- **本番 GitHub Wiki への `git push` は行わないでください**（後段で統合担当 `primary` が全 31 ページをまとめて最新 Wiki `master` へ公開します）。
- **本体リポジトリのファイルは変更・commit・PR 作成しないでください。** Issue コメント投稿も行わず、成果物は worktree 内 `tmp/wiki/`（担当 11 ページのみ変更）と `tmp/rollout-ball-research/` に保存してください。
- GitHub Wiki（`https://github.com/Motoki0705/tennis-lab.wiki.git`）はこの worktree の `tmp/wiki/` に clone し、以下の担当 11 ページ**以外**は一切触らないでください（`Meiji-4cam-Benchmarks.md` には触れない）。

## 担当 11 ページ
- **Tier 2（7 ページ・本文の表改善＋冒頭定型欄廃止）**:
  - Pillar III（ボール 5 ページ）:
    1. `Ball-Detection-Overview.md`
    2. `Ball-2D-Detector-Baseline-and-Domain-Adaptation.md`
    3. `Ball-Refiner-2D-Failure-and-3D-Transition.md`
    4. `Ball-3D-Smoothing-and-Scene-Evaluation.md`
    5. `Ball-Long-Sequence-and-Architecture-Exploration.md`
  - Pillar VI（研究モデル・用語集 2 ページ）:
    6. `Research-Models-PLCS-BLCS-SLCS.md`
    7. `Glossary-and-Search-Traps.md`
- **Tier 3（4 ページ・冒頭定型欄廃止＋簡潔な既存参照構造を維持）**:
  - Pillar III:
    8. `Ball-Datasets-Annotation-and-Test-Contracts.md`
  - Pillar VI:
    9. `Research-and-Infra-Overview.md`
    10. `Installation-Storage-and-Outputs.md`
    11. `Knowledge-Library-and-Annotation.md`

## 参照すべき規約・棚卸し
- `docs/wiki/README.md`（執筆ガイドライン正本）
- `docs/wiki/rendering-comparison.md`（各ページの改訂・簡素適用理由）
- `[local path]`（各ページの出典移設先・全表の改訂案）

## 改訂要件
1. **全 11 ページ共通（冒頭定型メタデータ欄・定型戻り行の廃止）**:
   - 冒頭の「項目／内容」表（「想定読者」「検索キーワード」「正本ファイル」[local path]`Glossary-and-Search-Traps` の「主な正本パス」）と定型戻り行（`> ← ...へ戻る`）を削除し、タイトル直後から本題に入れる構成にする。
   - 「正本ファイル」[local path]、該当本文または末尾の「出典・関連ページ」節へ確実に移設する（1 リンクも落とさない）。
2. **Tier 2（7 ページ）の表改善**:
   - 横幅過多・長文セル・評価集合と採否判断が混在した表を、`docs/wiki/README.md` に従ってツリー・`<details><summary>` 折りたたみ（タグ前後・`<[local path]>` 後の空行必須）・階層リスト・項目別見出しへ整理する。
   - 数値比較表（例: `Ball-Long-Sequence-and-Architecture-Exploration` の 6 列比較、`Ball-3D-Smoothing-and-Scene-Evaluation` の LOVO 6 方式比較、`Ball-2D-Detector-Baseline-and-Domain-Adaptation` 等）は、同一評価集合・同一母数の数値だけを 3〜5 列以内の簡潔な小表にまとめ、長い採否判断・根拠は直下本文または箇条書きへ分離する。評価集合や比較禁止条件の違いを絶対に曖昧にしない。
   - `Glossary-and-Search-Traps.md` の長文用語対比表（最長セル 173 字）は、語別の定義・よくある誤解・参照先の階層リストまたは短表＋詳細リストへ見やすく再構成する。
3. **Tier 3（4 ページ）の簡素適用**:
   - 冒頭定型メタデータ表と定型戻り行の廃止・出典移設のみを行い、2〜4 列の簡潔な既存参照表はそのまま維持する（不要な書き換えをしない）。
4. **情報・数値・リンク・アンカーの完全保全**:
   - 旧版 11 ページのすべての数値・母数・比較禁止注記・判断経緯・全リンク・見出しアンカーを欠落なく維持する。

## 必須の公開前検証（成果物: `tmp/rollout-ball-research/`）
1. **共通検証 CLI の実行**: 全 11 ページに対して `.venv/bin/python -m tools.wiki_assets verify --wiki-page tmp/wiki/<Page>.md` を実行し、冒頭定型欄 0 件・`<details>` 空行・公開禁止情報 0 件・相対リンク検査が全件 PASS することを確認する。
2. **移行・リンク突合監査 (`tmp/rollout-ball-research/audit.md` / `.json`)**:
   - 旧版 11 ページの全リンク・主要数値・見出しアンカー・表数の変化（旧→新）を突合し、リンク・情報欠落 0 件を確認する。
3. **ローカル表示チェック (`tmp/rollout-ball-research/render-check.md`)**:
   - 代表 3 ページ（`Ball-Long-Sequence-and-Architecture-Exploration`, `Research-Models-PLCS-BLCS-SLCS`, `Glossary-and-Search-Traps`）について、ローカル Chromium（896px / 390px × ライト/ダーク）で変更前後の高さ・横スクロールブロック数・折りたたみ開閉・Mermaid 表示を確認し、スクリーンショットを目視確認する。

## 報告・終了形式
最終メッセージに、改訂した 11 ページのパス（`tmp/wiki/*.md`）、表数・横スクロールの変化、`tools.wiki_assets verify` 結果、旧版リンク・アンカー突合結果、代表ページ実表示結果を簡潔にまとめ、末尾に次の 1 行を出力してください：
`PARENT_RESULT {"schema_version":1,"state":"done","wait":null}`
