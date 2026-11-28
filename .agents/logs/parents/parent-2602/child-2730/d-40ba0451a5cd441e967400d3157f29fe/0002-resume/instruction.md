# #2602 残31ページ展開 統合担当 `primary`（第1ターン）: Pillar IV・V（9ページ: Tier 2×3, Tier 3×6）の改訂と公開前検証

`docs/wiki/README.md` および `docs/wiki/rendering-comparison.md`（および `#2694` の `inventory.md`）に従い、まずあなたが担当する Pillar IV・V の **9 ページ** を改訂し、公開前全検査を完了してください。
（※並行する補助担当 2 名 `pillar-scene-court`・`pillar-ball-research` が担当する 22 ページの統合・本番 Wiki への 31 ページ反映・全 38 ページ公開最終監査は、全員のドラフト完了後に継続指示 `--resume` で依頼します。）

## 厳守事項・境界
- **このターンでは本番 GitHub Wiki への `git push` は行わないでください。**
- **本体リポジトリのファイルは変更・commit・PR 作成しないでください。** Issue コメント投稿も行わず、成果物は worktree 内 `tmp/wiki/`（担当 9 ページのみ変更）と `tmp/rollout-ops-memory/` に保存してください。
- GitHub Wiki（`https://github.com/Motoki0705/tennis-lab.wiki.git`）はこの worktree の `tmp/wiki/` に clone し、以下の担当 9 ページ**以外**は一切触らないでください。

## 担当 9 ページ
- **Tier 2（3 ページ・本文の表改善＋冒頭定型欄廃止）**:
  - Pillar IV（AI 運用 2 ページ）:
    1. `AI-Ops-Overview.md`
    2. `Host-Ops-and-Recovery.md`
  - Pillar V（共有 Memory 1 ページ）:
    3. `Shared-Memory-Catalog.md`
- **Tier 3（6 ページ・冒頭定型欄廃止＋簡潔な既存参照構造を維持）**:
  - Pillar IV（5 ページ）:
    4. `Parent-Session-Lifecycle.md`
    5. `Issues-Adoption-and-Dispatcher.md`
    6. `Worker-Delegation-and-Waits.md`
    7. `Review-and-Merge-Gate.md`
    8. `Communication-Visibility-and-Audit-Logs.md`
  - Pillar V（1 ページ）:
    9. `Shared-Memory-and-AutoDream.md`

## 参照すべき規約・棚卸し
- `docs/wiki/README.md`（執筆ガイドライン正本）
- `docs/wiki/rendering-comparison.md`（各ページの改訂・簡素適用理由）
- `[local path]`（各ページの出典移設先・全表の改訂案）

## 改訂要件
1. **全 9 ページ共通（冒頭定型メタデータ欄・定型戻り行の廃止）**:
   - 冒頭の「項目／内容」表（「想定読者」「検索キーワード」「正本ファイル」）と定型戻り行（`> ← ...へ戻る`）を削除し、タイトル直後から本題に入れる構成にする。
   - 「正本ファイル」にあった全リンク・必要な出典情報は、該当本文または末尾の「出典・関連ページ」節へ確実に移設する（1 リンクも落とさない）。
2. **Tier 2（3 ページ）の表改善**:
   - `AI-Ops-Overview.md`: ロール・モデル設定や「人間が関与する 6 つのポイント一覧」（最長セル 141 字）の横並び長文表を、役割別／判断時点別の階層リストへ再構成する（カテゴリ詳細ページ索引は簡潔に維持可）。
   - `Host-Ops-and-Recovery.md`: 症状別復旧（最長セル 153 字）と保護仕組みの横長表を、緊急時にすぐ見つけられる症状別見出し・項目リスト（または短い索引＋症状別手順）へ改善する（unit 一覧の簡潔な 3 列参照表は維持可）。
   - `Shared-Memory-Catalog.md`: 4 層（workflow / decision / environment / gotcha）の長い 4 本の索引表（横方向の確認手掛かりが長い）を、層別の見やすい階層リストまたは `<details>` 折りたたみ（空行必須）＋短索引へ再構成し、全 25 件のエントリとリンク・確認手掛かりを完全維持する。
3. **Tier 3（6 ページ）の簡素適用**:
   - 冒頭定型メタデータ表と定型戻り行の廃止・出典移設のみを行い、2〜3 列の簡潔な既存参照表はそのまま維持する。
4. **情報・リンク・アンカーの完全保全**:
   - 旧版 9 ページのすべての記述・全リンク・見出しアンカー・Mermaid 図を欠落なく維持する。

## 必須の公開前検証（成果物: `tmp/rollout-ops-memory/`）
1. **共通検証 CLI の実行**: 全 9 ページに対して `.venv/bin/python -m tools.wiki_assets verify --wiki-page tmp/wiki/<Page>.md` を実行し、冒頭定型欄 0 件・`<details>` 空行・公開禁止情報 0 件・相対リンク検査が全件 PASS することを確認する。
2. **移行・リンク突合監査 (`tmp/rollout-ops-memory/audit.md` / `.json`)**:
   - 旧版 9 ページの全リンク・見出しアンカー・表数の変化（旧→新）を突合し、リンク・情報欠落 0 件を確認する。
3. **ローカル表示チェック (`tmp/rollout-ops-memory/render-check.md`)**:
   - Tier 2 の 3 ページ（`AI-Ops-Overview`, `Host-Ops-and-Recovery`, `Shared-Memory-Catalog`）について、ローカル Chromium（896px / 390px × ライト/ダーク）で変更前後の高さ・横スクロールブロック数・折りたたみ開閉・Mermaid 表示を確認し、スクリーンショットを目視確認する。

## 報告・終了形式
最終メッセージに、改訂した 9 ページのパス（`tmp/wiki/*.md`）、表数・横スクロールの変化、`tools.wiki_assets verify` 結果、旧版リンク・アンカー突合結果、Tier 2 3 ページ実表示結果を簡潔にまとめ、末尾に次の 1 行を出力してください：
`PARENT_RESULT {"schema_version":1,"state":"done","wait":null}`
