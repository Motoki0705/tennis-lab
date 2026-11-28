# #2730 統合公開・全38ページ本番Wiki最終監査（Turn 2）

親AI（Mateo）より、Tier 2・Tier 3 全31ページの統合・本番 Wiki 公開・全38ページ最終監査を依頼します。
先に公開した Tier 1 全6ページ（`Meiji-4cam-Benchmarks` コミット `405c4ee09d79e94a81e1a5393786933fd47f1909`、人物モデル比較5ページ＋68アセット＋一時比較撤去 コミット `99b66328f567570ac7cc3077d2bf32c3a4ca7ef2`）は本番 Wiki `master` に反映済みです。また補助担当2名（`pillar-scene-court` 11ページ、`pillar-ball-research` 11ページ）のドラフトと公開前検証も親レビューを通過しました。

## 1. 統合対象（計31ページ）

### primary（あなたの worktree 内 `tmp/wiki/` にある改訂済み 9 ページ）
- Tier 2 (3): `AI-Ops-Overview.md`, `Host-Ops-and-Recovery.md`, `Shared-Memory-Catalog.md`
- Tier 3 (6): `Parent-Session-Lifecycle.md`, `Issues-Adoption-and-Dispatcher.md`, `Worker-Delegation-and-Waits.md`, `Review-and-Merge-Gate.md`, `Communication-Visibility-and-Audit-Logs.md`, `Shared-Memory-and-AutoDream.md`

### pillar-scene-court（コピー元: `[local path]` の 11 ページ）
- Tier 2 (11):
  - `Tennis-Scene-Pipeline-Overview.md`
  - `Tennis-Scene-History-and-Optimizations.md`
  - `Component-Person-Detection-and-Tracking.md`
  - `Component-CourtSide-Association-Alignment.md`
  - `Component-3D-Reconstruction-and-Tooling.md`
  - `Court-Detection-Overview.md`
  - `Court-Model-Evolution-and-Backbones.md`
  - `Court-Geometry-Postprocessing-and-Video-Shared.md`
  - `Synthetic-3DGS-Alignment-and-Generation.md`
  - `Synthetic-Scaling-Diversity-and-Pair-Studies.md`
  - `NHT-Rendering-and-Foreground-Compositing.md`

### pillar-ball-research（コピー元: `[local path]` の 11 ページ）
- Tier 2 (7):
  - `Ball-Detection-Overview.md`
  - `Ball-2D-Detector-Baseline-and-Domain-Adaptation.md`
  - `Ball-Refiner-2D-Failure-and-3D-Transition.md`
  - `Ball-3D-Smoothing-and-Scene-Evaluation.md`
  - `Ball-Long-Sequence-and-Architecture-Exploration.md`
  - `Research-Models-PLCS-BLCS-SLCS.md`
  - `Glossary-and-Search-Traps.md`
- Tier 3 (4):
  - `Ball-Datasets-Annotation-and-Test-Contracts.md`
  - `Research-and-Infra-Overview.md`
  - `Installation-Storage-and-Outputs.md`
  - `Knowledge-Library-and-Annotation.md`

## 2. 実施手順

### Step A: 最新 Wiki `master` への統合と差分固定
1. 自分の `tmp/wiki/` にある改訂済み 9 ページを一時退避した上で、`tmp/wiki/` を `git reset --hard` / `git fetch origin` / `git pull --ff-only origin master` で最新 `99b66328f567570ac7cc3077d2bf32c3a4ca7ef2` に揃える（他の更新が入っていた場合は基点差分を確認し、31ページ以外を壊さないこと）。
2. 退避した 9 ページと、補助担当 2 名の worktree から計 22 ページ（合計 31 ページ）を `tmp/wiki/` に配置する。
3. `git -C tmp/wiki status --short` で **変更がこの 31 ファイルのみ** であることを厳密に確認する。
   - `Home.md`、`_Sidebar.md`、`_Footer.md`、Tier 1 の 6 ページ（`Meiji-4cam-Benchmarks.md` および人物比較 5 ページ）、`assets/**` は一切変更しないこと。
   - 一時比較ページ `_Preview-*` や `assets/_preview-style-comparison/` が存在しないことも確認すること。

### Step B: 公開前・全38ページ横断検証
1. 全 37 非Home ページに対して `.venv/bin/python -m tools.wiki_assets verify --wiki-page tmp/wiki/<Page>.md` を実行し、全 37/37 PASS を確認する（ログを `tmp/rollout-final/verify-all-37.log` に保存）。
   - ※もし 31 ページ統合後にクロスリンクのフラグメント不整合や微細な検証エラーが見つかった場合のみ、対象ページを最小修正して再検証し、修正内容と理由を監査レポートに明記する。
2. 全 38 ページ + `_Sidebar.md` + `_Footer.md` を横断検査し、`tmp/rollout-final/full-wiki-audit.json` に記録する:
   - 非Home 37 ページすべてで冒頭定型メタ表 0 件・定型戻りリンク（`> ← ...`）0 件・定型 `基準日` ブロック 0 件。
   - 全 38 ページで `<details><summary>` 空行違反 0 件。
   - 全 38 ページ + `_Sidebar.md` + `_Footer.md` で公開禁止情報（ローカル絶対パス・UNC・トークン・PII 等）0 件。
   - 全 38 ページ + `_Sidebar.md` + `_Footer.md` 内の相対 Wiki リンクおよび `#anchor`（ページ内・ページ跨ぎ双方）が、実在ページと実在見出し/カスタムアンカーへ解決すること。
   - 全アセット参照（`assets/...`）が `tmp/wiki/assets/` に実在すること（68 ファイル）。
   - 棚卸し基点 `b63668ad3d04dfa9f29f517cecd293b3c85b12fb` と統合後（38ページ）の全体統計（総表数、冒頭定型表数、本文表数、Tier別集計）を算出。

### Step C: 本番 Wiki へのコミット・push
1. `tmp/wiki/` で 31 ファイルをステージし、日本語コミットメッセージでコミットする。
   - 著者・コミッターは直近コミット（`405c4ee0` / `99b66328`）と同じ GitHub noreply（`Motoki0705 <132140099+Motoki0705@users.noreply.github.com>`）を使う。
   - `Co-Authored-By` は付けない。
2. `git -C tmp/wiki push origin master` で公開し、公開コミット SHA を記録する。
3. 本体リポジトリの worktree（`git status`）が完全にクリーン（追跡ファイルへの変更・未追跡ソースなし）であることを確認する。

### Step D: 本番公開後の全38ページ最終監査（HTTP + 実ブラウザ Chromium 4 条件）
1. **匿名 HTTP・SHA-256 全件検査**（`tmp/rollout-final/published-http.json`）:
   - 全 38 ページの公開 URL（`https://github.com/Motoki0705/tennis-lab/wiki/<Page>`）と raw Markdown が匿名 HTTP 200 かつローカル公開版とバイト一致することを確認。
   - `_Sidebar.md` からリンクされている全公開ページが HTTP 200 であることを確認。
   - 一時比較ページ 3 件が存在しないこと（302 / raw 404）を再確認。
2. **本番 Chromium 153 4 条件実表示監査**（1440×980 / 390×844 × ライト / ダーク、未ログイン状態、`tmp/rollout-final/published-metrics.json`・`tmp/rollout-final/screenshots-published/`）:
   - Tier 2 代表 8 ページ（全柱を網羅）:
     1. `Tennis-Scene-Pipeline-Overview`
     2. `Court-Model-Evolution-and-Backbones`
     3. `Synthetic-Scaling-Diversity-and-Pair-Studies`
     4. `Ball-Long-Sequence-and-Architecture-Exploration`
     5. `Research-Models-PLCS-BLCS-SLCS`
     6. `Glossary-and-Search-Traps`
     7. `AI-Ops-Overview`
     8. `Shared-Memory-Catalog`
   - Tier 1 / Tier 3 スポット確認 4 ページ:
     9. `Meiji-4cam-Benchmarks`
     10. `Person-Model-Comparison-Overview`
     11. `Installation-Storage-and-Outputs`
     12. `Parent-Session-Lifecycle`
   - 上記 12 ページ × 4 条件（計 48 条件）で、既定・折りたたみ展開時の横スクロール件数（ページ全体・本文ブロック別）、Mermaid 描画（iframe 内 SVG）、`<details>` の開閉、画像デコード、サイドバー遷移を実測し、代表スクリーンショットを保存して目視確認すること。
   - もし本番実表示で重大な崩れ（Markdown 記法崩れ、折りたたみ内未描画、Mermaid 構文エラー、画像欠損、明らかなリンク切れ等）が 1 件でも見つかった場合は、その場で原因を修正し、追加コミット・push して再検証すること。

### Step E: 最終監査レポート作成
- `tmp/rollout-final/full-wiki-audit.md` に以下を簡潔かつ網羅的にまとめる:
  - 公開コミット SHA（Tier 1 2件 + Tier 2/3 1件 + 修正があれば追加分）と差分範囲
  - 全 38 ページの棚卸し前後サマリー（冒頭定型表 37→0、定型戻り行 30→0、定型基準日 6→0、本文表数の推移）
  - 公開前横断検証（37/37 CLI PASS、リンク・アンカー・プライバシー・アセット全件照合）
  - 公開後匿名 HTTP 38/38 + サイドバー + 一時ページ不在の確認結果
  - 本番 Chromium 12 ページ × 4 条件（48 条件）の計測表と目視所見・スクリーンショット一覧
- 最終応答に、公開コミット SHA、全 38 ページ監査結果の要約、`tmp/rollout-final/full-wiki-audit.md` へのリンク、末尾 `PARENT_RESULT` JSON を出力する。
